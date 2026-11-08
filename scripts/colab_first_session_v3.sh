#!/usr/bin/env bash
# First GPU session for opd-v3, in one command:
#   environment -> model download -> 2-step smoke -> run summary -> teacher-fit probe.
# Writes one report folder on Drive. Moves an old smoke run aside instead of deleting it.
#   PROJECT_ROOT=/content/project2 OPD_OUTPUT_ROOT=/content/drive/MyDrive/project2/runs \
#     bash scripts/colab_first_session_v3.sh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/content/project2}"
export OPD_OUTPUT_ROOT="${OPD_OUTPUT_ROOT:-/content/drive/MyDrive/project2/runs}"
cd "$PROJECT_ROOT"
REPORT="$OPD_OUTPUT_ROOT/first_session_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$REPORT"
exec > >(tee -a "$REPORT/console.log") 2>&1
echo "== commit $(git rev-parse HEAD) on $(git branch --show-current)"
bash scripts/colab_bootstrap_v3.sh
echo "== prefetch (student + math/code/general teachers, about 48 GB)"
python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_smoke.json configs/project2_opd_v3_d3_general.json \
  | tee "$REPORT/prefetch.jsonl"
SMOKE="$OPD_OUTPUT_ROOT/opd_v3_smoke"
if [ -e "$SMOKE" ]; then mv "$SMOKE" "$SMOKE.previous_$(date -u +%s)"; fi
echo "== smoke: 2 mixed steps, 4+4 prompts, 64 new tokens"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_smoke.json --output-root "$OPD_OUTPUT_ROOT"
python -m scripts.summarize_run_v3 --run-dir "$SMOKE" | tee "$REPORT/smoke_summary.json"
cp "$SMOKE"/rollouts/*.jsonl "$REPORT/" 2>/dev/null || true
echo "== teacher-fit probe: 64+64 student rollouts, 3 teachers x 2 domains"
python -m scripts.probe_teacher_fit_v3 --config configs/project2_opd_v3_d4_dual.json --output "$REPORT/teacher_fit.json"
echo "== done. Send these two files: $REPORT/smoke_summary.json and $REPORT/teacher_fit.json"
