#!/usr/bin/env bash
# opd-v3 Linux/Colab setup for A100 (sm_80) or G4 RTX PRO 6000 Blackwell (sm_120).
# UNVERIFIED on GPU until the first smoke run. Downloads no model weights and starts no training.
#   PROJECT_ROOT=/content/project2 OPD_OUTPUT_ROOT=/content/drive/MyDrive/project2/runs \
#     INSTALL_VLLM=1 bash scripts/colab_bootstrap_v3.sh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/content/project2}"
export OPD_OUTPUT_ROOT="${OPD_OUTPUT_ROOT:-/content/drive/MyDrive/project2/runs}"
case "$OPD_OUTPUT_ROOT" in
  /content/drive/*) grep -qs ' /content/drive ' /proc/mounts || {
    echo "Google Drive is not mounted at /content/drive; mount it or set OPD_OUTPUT_ROOT=/content/runs" >&2; exit 1; } ;;
esac
cd "$PROJECT_ROOT"
test -f src/training/opd_v3.py
python -c 'import sys; v = sys.version_info[:2]; assert (3, 10) <= v < (3, 14), sys.version; print("python", sys.version)'
ENV_DIR="$OPD_OUTPUT_ROOT/environment-v3/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ENV_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv | tee "$ENV_DIR/gpu.csv"
nvidia-smi > "$ENV_DIR/nvidia-smi.txt"
# cu128 wheels contain sm_80 and sm_120 kernels; cu126 wheels lack sm_120 (Blackwell).
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/linux-cuda-v3.txt
if [ "${INSTALL_VLLM:-0}" = "1" ]; then python -m pip install -r requirements/linux-cuda-v3-eval.txt; fi
python -m pip check | tee "$ENV_DIR/pip-check.txt" || echo "pip check reported conflicts; see $ENV_DIR/pip-check.txt"
python -m pip freeze > "$ENV_DIR/pip-freeze.txt"
python - <<'PY' | tee "$ENV_DIR/torch.txt"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
major, minor = torch.cuda.get_device_capability(0)
arch = f'sm_{major}{minor}'
props = torch.cuda.get_device_properties(0)
print(torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), arch,
      f'{props.total_memory / 2**30:.1f} GiB', 'archs', torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f'torch build has no {arch} kernels'
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
print('bf16 matmul ok', float((x @ x).float().abs().mean()))
PY
python -m scripts.restore_official_benchmarks | tee "$ENV_DIR/benchmarks.txt"
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_opd_v3.py tests/test_v3_tools.py tests/test_release_v5.py \
  tests/test_eval_v3.py tests/test_opd_contract.py tests/test_losses_and_masks.py tests/test_opd_v2.py | tee "$ENV_DIR/cpu-tests.txt"
echo "Bootstrap completed: $ENV_DIR. Next: python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_smoke.json"
