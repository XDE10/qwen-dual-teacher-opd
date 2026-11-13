#!/usr/bin/env bash
# Colab entry for the separate R1 / Latent-MOPD experiment.
set -euo pipefail

cd "$(dirname "$0")/.."
export OPD_OUTPUT_ROOT=/content/runs
export USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

variant=${1:-latent}
case "$variant" in
  token|latent|math20) ;;
  *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 2 ;;
esac

mode=${2:-full}
case "$mode" in
  smoke|full|smoke-then-full) ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
run_root="$OPD_OUTPUT_ROOT/r1_latent_observable"
mkdir -p "$run_root"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 30
  done
) &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

run_phase() {
  local phase=$1
  shift
  local output="$run_root/$phase"
  local -a resume=()
  if [[ -f "$output/latent_mopd_top16_$variant/run_identity.json" ]]; then resume=(--resume); fi
  python -m src.training.latent_opd "$variant" --output-root "$output" --responses 4 --rollout-batch 4 \
    --microbatch-tokens 24576 "$@" "${resume[@]}"
}

if [[ "$mode" != full ]]; then
  run_phase smoke --steps 3 --prompts 1 --max-new-tokens 1024
  python - "$run_root/smoke/latent_mopd_top16_$variant" <<'PY'
import json
import math
import sys
from pathlib import Path

from src.training.runtime_v3 import validate_checkpoint

run = Path(sys.argv[1])
identity = json.loads((run / 'run_identity.json').read_text())
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
assert [row['domain'] for row in rows] == ['math', 'code', 'math']
assert all(math.isfinite(row['loss']) and row['gradient_norm'] > 0 for row in rows)
validate_checkpoint(run / 'checkpoint-1', identity)
validate_checkpoint(run / 'checkpoint-3', identity)
print('Three-step GPU smoke passed', flush=True)
PY
fi
if [[ "$mode" != smoke ]]; then
  run_phase full --steps 62 --prompts 32 --max-new-tokens 16384
fi
