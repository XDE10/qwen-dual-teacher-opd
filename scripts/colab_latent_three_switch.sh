#!/usr/bin/env bash
set -euo pipefail
cd /content/project2
export OPD_OUTPUT_ROOT=/content/runs USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

step=${1:?Specify a sealed compatible checkpoint number}
[[ "$step" =~ ^[1-9][0-9]*$ ]] && (( step <= 13 ))
old_root=/content/runs/latent_three_codefix_20261115/full
new_root=/content/runs/latent_three_poolfix_20261115/full
old_manifest=/content/project2/runs/latent_mopd_three/release_codefix_20261115/manifest.json
new_manifest=/content/project2/runs/latent_mopd_three/release_poolfix_20261115/manifest.json
old_run="$old_root/latent_mopd_three_latent"
new_run="$new_root/latent_mopd_three_latent"
old_job=/content/jobs/latent-three-codefix-62
new_job=latent-three-poolfix-62
exec 9>/content/latent-three-poolfix-switch.lock
flock -n 9
[[ ! -e "/content/jobs/$new_job" ]]
printf '%s  %s\n' \
  071898e06aa54ad0471f3c949628d36d92f735e929be26c859f3869ff38c0499 "$old_manifest" \
  e5f4c579f8a4514795e6ba80e1e4c2df034e724346bbfce7a2ea150d7d969252 "$new_manifest" | sha256sum -c

python - "$old_job" "$old_manifest" "$old_root" "$step" <<'PY'
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts.colab_job_v3 import check_drive, check_publish
from scripts.migrate_latent_three_checkpoint import migrate

job, old_manifest, old_root = map(Path, sys.argv[1:4])
step = int(sys.argv[4])
supervisor = psutil.Process(int((job / 'pid').read_text()))
command = supervisor.cmdline()
if command[-2:] != ['--supervise', str(job)] or Path(command[1]).name != 'colab_job_v3.py':
    raise SystemExit('Unexpected old job supervisor')

trainers = []
for child in supervisor.children(recursive=True):
    try:
        args = child.cmdline()
    except psutil.NoSuchProcess:
        continue
    if (args[1:4] == ['-m', 'src.training.latent_mopd_three', 'latent']
            and '--resume' in args and str(old_manifest) in args and str(old_root) in args):
        trainers.append(child)
if len(trainers) != 1:
    raise SystemExit('Expected exactly one resumed trainer in the original job')
trainer, = trainers

problems = [error for error in (check_drive('mydrive:opd_runs'), check_publish()) if error]
if problems:
    raise SystemExit('\n'.join(problems))
release_root = old_manifest.parent.parent
new_manifest = release_root / 'release_poolfix_20261115/manifest.json'
destination = Path('/content/runs/latent_three_poolfix_20261115/full/latent_mopd_three_latent')
checkpoint = old_root / 'latent_mopd_three_latent' / f'checkpoint-{step}'
if not destination.exists():
    migrate(checkpoint, old_manifest, new_manifest, destination)
else:
    from src.training.latent_mopd_three import verify_checkpoint_files
    from src.training.runtime_v2 import read, sha

    receipt = read(destination / 'migration.json')
    if (receipt['step'] != step or receipt['source_checkpoint'] != str(checkpoint)
            or receipt['source_checkpoint_manifest_sha256'] != sha(checkpoint / 'checkpoint_manifest.json')
            or receipt['new_release_sha256'] != sha(new_manifest)):
        raise SystemExit('Existing fork has a different migration receipt')
    verify_checkpoint_files(destination / checkpoint.name)
print(f'LATENT_THREE_MIGRATION_READY step={step}', flush=True)

def gpu_processes():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True,
    )
    return {int(line.strip()) for line in out.splitlines() if line.strip()}

if gpu_processes() - {trainer.pid}:
    raise SystemExit('Another compute process is present on the G4')
fd = os.pidfd_open(trainer.pid)
try:
    if not trainer.is_running() or trainer not in supervisor.children(recursive=True):
        raise SystemExit('Original trainer changed during preflight')
    signal.pidfd_send_signal(fd, signal.SIGTERM)
finally:
    os.close(fd)
trainer.wait(timeout=60)
deadline = time.monotonic() + 15
while gpu_processes() and time.monotonic() < deadline:
    time.sleep(0.5)
if gpu_processes():
    raise SystemExit('The G4 still has a compute process')
print('LATENT_THREE_ORIGINAL_TRAINER_STOPPED', flush=True)
PY

exec python scripts/colab_job_v3.py "$new_job" --drive mydrive:opd_runs --publish --sync-every 15 \
  --save full="$new_run" -- python -m src.training.latent_mopd_three latent \
  --manifest "$new_manifest" --output-root "$new_root" \
  --steps 62 --prompts 32 --responses 4 --max-new-tokens 16384 --rollout-batch 4 --resume
