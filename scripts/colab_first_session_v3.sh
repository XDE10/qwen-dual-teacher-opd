#!/usr/bin/env bash
# First GPU session for opd-v3, in one command:
#   environment -> model download -> 2-step smoke -> run summary -> teacher-fit probe.
# Writes one report folder under $OPD_OUTPUT_ROOT. On exit, also after a failure, packs it into
# $OPD_ARCHIVE (default /content/opd_first_session.tar.gz), one fixed path for `colab download`,
# which fetches single files only. The probe still runs when the smoke fails. An old smoke run
# is moved aside, not deleted.
#   PROJECT_ROOT=/content/project2 OPD_OUTPUT_ROOT=/content/drive/MyDrive/project2/runs \
#     bash scripts/colab_first_session_v3.sh
# With google-colab-cli (e.g. from WSL2), start it through scripts/colab_job_v3.py: docs/colab_cli_wsl2.md
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/content/project2}"
export OPD_OUTPUT_ROOT="${OPD_OUTPUT_ROOT:-/content/drive/MyDrive/project2/runs}"
ARCHIVE="${OPD_ARCHIVE:-/content/opd_first_session.tar.gz}"
case "$OPD_OUTPUT_ROOT" in
  /content/drive/*)
    if ! grep -qs ' /content/drive ' /proc/mounts; then
      echo "Google Drive is not mounted at /content/drive (colab-cli: colab drivemount -s <session>)." >&2
      echo "Mount it, or set OPD_OUTPUT_ROOT=/content/runs and download $ARCHIVE before the runtime stops." >&2
      exit 1
    fi ;;
esac
cd "$PROJECT_ROOT"
REPORT="$OPD_OUTPUT_ROOT/first_session_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$REPORT"
pack() {
  local status=$? rc=0
  tar --warning=no-file-changed -czf "$ARCHIVE" -C "$(dirname "$REPORT")" "$(basename "$REPORT")" || rc=$?
  if [ "$rc" -gt 1 ]; then echo "== could not write $ARCHIVE (tar exit $rc)"; fi
  echo "== exit status $status; report folder $REPORT; archive $ARCHIVE"
}
trap pack EXIT
exec > >(tee -a "$REPORT/console.log") 2>&1
# Progress bars would fill console.log with carriage-return updates.
export HF_HUB_DISABLE_PROGRESS_BARS=1 PIP_PROGRESS_BAR=off
echo "== commit $(git rev-parse HEAD) on $(git branch --show-current)"
bash scripts/colab_bootstrap_v3.sh
echo "== prefetch (student + math/code/general teachers, about 48 GB)"
python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_smoke.json configs/project2_opd_v3_d3_general.json \
  | tee "$REPORT/prefetch.jsonl"
SMOKE="$OPD_OUTPUT_ROOT/opd_v3_smoke"
if [ -e "$SMOKE" ]; then mv "$SMOKE" "$SMOKE.previous_$(date -u +%s)"; fi
STATUS=0
echo "== smoke: 2 mixed steps, 4+4 prompts, 64 new tokens"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_smoke.json --output-root "$OPD_OUTPUT_ROOT" \
  || { STATUS=1; echo "== smoke FAILED; the teacher-fit probe still runs"; }
if [ -f "$SMOKE/metrics.jsonl" ]; then
  python -m scripts.summarize_run_v3 --run-dir "$SMOKE" | tee "$REPORT/smoke_summary.json" || STATUS=1
fi
for f in metrics.jsonl run_identity.json teacher_pool.json; do cp "$SMOKE/$f" "$REPORT/smoke_$f" 2>/dev/null || true; done
cp "$SMOKE"/startup_*.json "$SMOKE"/rollouts/*.jsonl "$REPORT/" 2>/dev/null || true
echo "== teacher-fit probe: 64+64 student rollouts, 3 teachers x 2 domains"
python -m scripts.probe_teacher_fit_v3 --config configs/project2_opd_v3_d4_dual.json --output "$REPORT/teacher_fit.json" \
  || STATUS=1
echo "== done. Send smoke_summary.json and teacher_fit.json (console.log if a step failed); all are in $ARCHIVE"
exit "$STATUS"
