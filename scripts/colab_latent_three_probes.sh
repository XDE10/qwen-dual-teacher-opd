#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPD_OUTPUT_ROOT=/content/runs USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

mode=${1:-}
manifest=${LATENT_THREE_MANIFEST:-/content/project2/runs/latent_mopd_three/release/manifest.json}
root=${LATENT_THREE_RUN_ROOT:-/content/runs/latent_three_g4}
case "$mode" in
  smoke) steps=6; prompts=1; cap=1024 ;;
  scale) steps=1; prompts=32; cap=16384 ;;
  joint16k)
    python -m scripts.probe_latent_mopd_three_joint --manifest "$manifest" \
      --checkpoint "$root/scale/latent_mopd_three_latent/checkpoint-1" \
      --output "$root/evidence/joint16k/probe.json"
    exit
    ;;
  *) printf 'Usage: %s {smoke|scale|joint16k}\n' "$0" >&2; exit 2 ;;
esac

out="$root/$mode"
evidence="$root/evidence/$mode"
python -m src.training.latent_mopd_three latent --manifest "$manifest" --output-root "$out" \
  --steps "$steps" --prompts "$prompts" --responses 4 --max-new-tokens "$cap" \
  --rollout-batch 4 --keep-checkpoints 1

python - "$mode" "$out/latent_mopd_three_latent" "$evidence" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path

mode, run, evidence = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
domains = ['math', 'code', 'logic'] * 2 if mode == 'smoke' else ['math']
responses = 4 if mode == 'smoke' else 128
assert [r['step'] for r in rows] == list(range(1, len(domains) + 1))
assert [r['domain'] for r in rows] == domains
for i, row in enumerate(rows):
    weight = 0.0 if i < 3 else 0.1
    assert row['token_weight'] == weight and row['latent_weight'] == 1.0 - weight
    assert row['responses'] == responses and row['optimizer_updates'] == 1
    assert all(math.isfinite(row[name]) for name in ('loss', 'gradient_norm', 'raw_rep_loss', 'token_loss'))
    assert row['gradient_norm'] > 0 and row['raw_rep_loss'] > 0
checkpoint = run / f'checkpoint-{len(domains)}'
evidence.mkdir(parents=True, exist_ok=True)
for name in ('metrics.jsonl', 'phase_metrics.jsonl', 'run_identity.json'):
    shutil.copyfile(run / name, evidence / name)
shutil.copyfile(checkpoint / 'checkpoint_manifest.json', evidence / 'checkpoint_manifest.json')
print(f'LATENT_THREE_PROBE_OK {mode}', flush=True)
PY
