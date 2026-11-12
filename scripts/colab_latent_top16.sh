#!/usr/bin/env bash
# Colab entry for the separate R1 / Latent-MOPD experiment.
set -euo pipefail

cd "$(dirname "$0")/.."
export OPD_OUTPUT_ROOT=/content/runs
export USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

variant=${1:-token}
case "$variant" in
  token|latent|math20) ;;
  *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 2 ;;
esac

run_root="$OPD_OUTPUT_ROOT/r1_latent_top16_20261112"
run="$run_root/latent_mopd_top16_$variant"
mkdir -p "$run_root"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 30
  done
) &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

common=("$variant" --output-root "$run_root" --steps 62 --prompts 32 --responses 4
        --max-new-tokens 16384 --rollout-batch 16 --microbatch-tokens 24576)

if ! compgen -G "$run/checkpoint-*/checkpoint_manifest.json" >/dev/null; then
  python -m src.training.latent_opd "${common[@]}" --stop-at-step 1
fi
python -m src.training.latent_opd "${common[@]}" --resume
