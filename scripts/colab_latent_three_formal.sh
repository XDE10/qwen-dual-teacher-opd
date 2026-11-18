#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OPD_OUTPUT_ROOT=/content/runs USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export LATENT_THREE_MANIFEST=/content/project2/runs/latent_mopd_three/release_codefix_20261115/manifest.json
export LATENT_THREE_RUN_ROOT=/content/runs/latent_three_codefix_20261115
formal="$LATENT_THREE_RUN_ROOT/full"
run="$formal/latent_mopd_three_latent"
evidence="$LATENT_THREE_RUN_ROOT/evidence"
train_args=(latent --manifest "$LATENT_THREE_MANIFEST" --output-root "$formal"
  --steps 62 --prompts 32 --responses 4 --max-new-tokens 16384 --rollout-batch 4)

verify_steps() {
  python - "$run" "$LATENT_THREE_MANIFEST" "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

from src.losses.latent_opd import loss_weights
from src.training.latent_mopd_three import verify_checkpoint_files
from src.training.runtime_v2 import sha

run, manifest = map(Path, sys.argv[1:3])
last = int(sys.argv[3])
release = json.loads(manifest.read_text())
schedule = manifest.with_name("schedule.jsonl")
assert sha(schedule) == release["files"][schedule.name]["sha256"]
planned = [json.loads(line) for line in schedule.read_text().splitlines()][:last]
measured = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
assert len(planned) == len(measured) == last
for expected, row in zip(planned, measured, strict=True):
    assert all(row[key] == expected[key] for key in ("step", "domain", "sample_ids"))
    assert row["responses"] == 128 and row["optimizer_updates"] == 1
    token, rep = loss_weights("latent", row["domain"], (row["step"] - 1) // 3)
    assert (row["token_weight"], row["latent_weight"]) == (token, rep)
    assert all(math.isfinite(row[key]) for key in ("loss", "gradient_norm", "raw_rep_loss", "token_loss"))
verify_checkpoint_files(run / f"checkpoint-{last}")
print(f"LATENT_THREE_FORMAL_VERIFIED step={last}", flush=True)
PY
}

printf '%s\n' 'LATENT_THREE_STAGE smoke'
bash scripts/colab_latent_three_probes.sh smoke
printf '%s\n' 'LATENT_THREE_STAGE scale_formal_step_1'
python -m src.training.latent_mopd_three "${train_args[@]}" --stop-at-step 1
verify_steps 1

printf '%s\n' 'LATENT_THREE_STAGE joint16k'
python -m scripts.probe_latent_mopd_three_joint --manifest "$LATENT_THREE_MANIFEST" \
  --checkpoint "$run/checkpoint-1" --output "$evidence/joint16k/probe.json"

printf '%s\n' 'LATENT_THREE_STAGE formal_steps_2_to_62'
python -m src.training.latent_mopd_three "${train_args[@]}" --resume
verify_steps 62
