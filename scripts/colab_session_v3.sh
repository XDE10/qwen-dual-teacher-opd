#!/usr/bin/env bash
# One Colab session of opd-v3 work: environment setup and model download, then the given steps in order.
# A failed step is recorded and the next step still runs; the exit status is 1 if any step failed.
# Run folders, evaluations and teacher outputs go under $OPD_OUTPUT_ROOT (default /content/runs, the VM
# disk). This session's console log, step list, probe report and environment records go to
# $OPD_OUTPUT_ROOT/sessions/<UTC time>/.
#   PROJECT_ROOT=/content/project2 bash scripts/colab_session_v3.sh probe train=d4_dual:30 eval=d0 teacher-outputs
# From colab-cli, start it with scripts/colab_job_v3.py, which saves the results to Drive and to the results
# branch (docs/colab_cli_wsl2.md).
#
# Steps:
#   probe                teacher-fit probe of configs/project2_opd_v3_d4_dual.json: KL matrix, the tokens that carry
#                        most KL, and per-token values of 6 rollouts per domain (teacher_fit.json, teacher_fit_tokens.jsonl)
#   train=NAME[:STOP]    train configs/project2_opd_v3_NAME.json; stop after step STOP if given. If the run folder
#                        already holds a sealed checkpoint, continue from the newest valid one; an unfinished folder
#                        without one is moved aside. Writes summary.json (with the stability gate) into the run folder.
#   eval=d0              greedy evaluation (HF engine) of the untrained student: math_dev, math500 and gsm8k_test are
#                        graded here; code_dev, humanevalplus and mbppplus become candidates for the local Docker sandbox
#   eval=NAME            the same for the newest checkpoint of run NAME
#   teacher-outputs      Math-7B on train_math (1024 new tokens) and Coder-7B on train_code (1536 new tokens), with
#                        token IDs, for the SeqKD-Mix baseline and the SFT warm start
#   restore=REMOTE:PATH  copy a folder saved on Drive (a run folder, teacher_outputs, ...) into $OPD_OUTPUT_ROOT
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/content/project2}"
export OPD_OUTPUT_ROOT="${OPD_OUTPUT_ROOT:-/content/runs}"
STUDENT=Qwen/Qwen2.5-1.5B-Instruct STUDENT_REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
MATH_TEACHER=Qwen/Qwen2.5-Math-7B-Instruct MATH_TEACHER_REV=ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
CODE_TEACHER=Qwen/Qwen2.5-Coder-7B-Instruct CODE_TEACHER_REV=c03e6d358207e414f1eca0bb1891e29f1db0e242
EVAL_SUITES=(math_dev math500 gsm8k_test code_dev humanevalplus mbppplus)

# An output root on a Drive mount point that is not mounted would silently fill the VM disk instead.
for mount in /content/drive /content/mydrive; do
  case "$OPD_OUTPUT_ROOT/" in
    "$mount"/*) grep -qs " $mount " /proc/mounts || {
      echo "$mount is not mounted; mount it or use OPD_OUTPUT_ROOT=/content/runs" >&2; exit 1; } ;;
  esac
done
cd "$PROJECT_ROOT" || exit 1
[ "$#" -gt 0 ] || { echo "usage: bash scripts/colab_session_v3.sh STEP [STEP ...] (see the header)" >&2; exit 2; }

bad() { echo "invalid step: $1" >&2; exit 2; }
for step in "$@"; do
  case "$step" in
    probe|teacher-outputs|eval=d0) ;;
    train=*)
      spec="${step#train=}"
      [ -f "configs/project2_opd_v3_${spec%%:*}.json" ] || bad "$step (no configs/project2_opd_v3_${spec%%:*}.json)"
      if [[ "$spec" == *:* ]] && ! [[ "${spec#*:}" =~ ^[1-9][0-9]*$ ]]; then bad "$step (STOP must be a positive integer)"; fi ;;
    eval=*) [ -f "configs/project2_opd_v3_${step#eval=}.json" ] || bad "$step (no configs/project2_opd_v3_${step#eval=}.json)" ;;
    restore=*) [[ "${step#restore=}" =~ ^[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]+$ ]] || bad "$step (expected REMOTE:PATH)" ;;
    *) bad "$step" ;;
  esac
done

SESSION="$OPD_OUTPUT_ROOT/sessions/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SESSION"
exec > >(tee -a "$SESSION/console.log") 2>&1
# Progress bars would fill the log with carriage-return updates; transformers need not import TensorFlow.
export HF_HUB_DISABLE_PROGRESS_BARS=1 PIP_PROGRESS_BAR=off USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== commit $(git rev-parse HEAD) on $(git branch --show-current); steps: $*"
OPD_ENV_DIR="$SESSION/environment" bash scripts/colab_bootstrap_v3.sh || { echo "== environment setup failed"; exit 1; }
echo "== model download (student, math, code and general teachers)"
python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_d4_dual.json configs/project2_opd_v3_d3_general.json \
  | tee "$SESSION/prefetch.jsonl" || { echo "== model download failed"; exit 1; }

run_dir() {
  echo "$OPD_OUTPUT_ROOT/$(python3 -c 'import json, os, sys; print(os.path.basename(json.load(open(sys.argv[1]))["output_dir"]))' \
    "configs/project2_opd_v3_$1.json")"
}

train_run() {
  local name="$1" stop="$2" run rc=0
  local args=(--config "configs/project2_opd_v3_$name.json" --output-root "$OPD_OUTPUT_ROOT")
  run="$(run_dir "$name")"
  if [ -n "$stop" ]; then args+=(--stop-at-step "$stop"); fi
  if compgen -G "$run/checkpoint-*/checkpoint_manifest.json" > /dev/null; then
    args+=(--resume-from latest)
  elif [ -e "$run" ]; then
    mv "$run" "$run.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "moved an unfinished $run aside (it has no sealed checkpoint)"
  fi
  python -m scripts.train_project2_opd_v3 "${args[@]}" || rc=$?
  if [ -f "$run/metrics.jsonl" ]; then
    python -m scripts.summarize_run_v3 --run-dir "$run" > "$run/summary.json" || rc=1
    python3 - "$run/summary.json" <<'PY' || rc=1
import json, sys
s = json.load(open(sys.argv[1]))
gate = s['gate']
print(f"== {s['run_dir']}: {s['steps']} steps, median {s['median_step_seconds']} s/step, "
      f"peak {s['overall_peak_gib']} GiB, gate pass={gate.get('pass')} failed={gate.get('failed')}")
PY
  fi
  return "$rc"
}

evaluate() {
  local name="$1"
  shift
  python -m scripts.generate_eval_v3 --engine hf --batch-size 128 "$@" --suites "${EVAL_SUITES[@]}" \
    --output-dir "$OPD_OUTPUT_ROOT/eval/$name"
}

eval_run() {
  local run checkpoint
  run="$(run_dir "$1")"
  checkpoint="$(ls -d "$run"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -z "$checkpoint" ]; then echo "no checkpoint in $run"; return 1; fi
  evaluate "$1-$(basename "$checkpoint")" --model "$STUDENT" --revision "$STUDENT_REV" --adapter "$checkpoint/student"
}

teacher_outputs() {
  local rc=0
  python -m scripts.generate_eval_v3 --engine hf --batch-size 128 --model "$MATH_TEACHER" --revision "$MATH_TEACHER_REV" \
    --suites train_math --max-new-math 1024 --save-token-ids --output-dir "$OPD_OUTPUT_ROOT/teacher_outputs/math7b" || rc=1
  python -m scripts.generate_eval_v3 --engine hf --batch-size 128 --model "$CODE_TEACHER" --revision "$CODE_TEACHER_REV" \
    --suites train_code --max-new-code 1536 --save-token-ids --output-dir "$OPD_OUTPUT_ROOT/teacher_outputs/coder7b" || rc=1
  return "$rc"
}

restore() {
  command -v rclone > /dev/null || { echo "rclone is not installed"; return 1; }
  rclone copy "$1" "$OPD_OUTPUT_ROOT/$(basename "$1")" --stats 1m --stats-one-line --stats-log-level NOTICE || return 1
  echo "restored $1 -> $OPD_OUTPUT_ROOT/$(basename "$1")"
}

do_step() {
  local spec
  case "$1" in
    probe) python -m scripts.probe_teacher_fit_v3 --config configs/project2_opd_v3_d4_dual.json \
             --output "$SESSION/teacher_fit.json" --dump-tokens 6 ;;
    train=*) spec="${1#train=}"
             if [[ "$spec" == *:* ]]; then train_run "${spec%%:*}" "${spec#*:}"; else train_run "$spec" ""; fi ;;
    eval=d0) evaluate d0 --model "$STUDENT" --revision "$STUDENT_REV" ;;
    eval=*) eval_run "${1#eval=}" ;;
    teacher-outputs) teacher_outputs ;;
    restore=*) restore "${1#restore=}" ;;
  esac
}

FAILED=()
for step in "$@"; do
  started=$(date +%s)
  echo "== step $step (started $(date -u +%H:%M:%SZ))"
  do_step "$step"
  rc=$?
  printf '{"step": "%s", "status": %d, "seconds": %d}\n' "$step" "$rc" "$(( $(date +%s) - started ))" >> "$SESSION/steps.jsonl"
  if [ "$rc" -ne 0 ]; then FAILED+=("$step"); echo "== step $step FAILED (exit $rc)"; fi
done
echo "== session done; failed steps: ${FAILED[*]:-none}; log $SESSION/console.log"
[ "${#FAILED[@]}" -eq 0 ]
