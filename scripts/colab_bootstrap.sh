#!/usr/bin/env bash
# UNVERIFIED Linux/CUDA setup only. No model downloads, code execution or training.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/content/project2}"
export OPD_OUTPUT_ROOT="${OPD_OUTPUT_ROOT:-/content/drive/MyDrive/project2/runs}"
cd "$PROJECT_ROOT"
test -f src/training/opd_v2.py
python -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
mkdir -p "$OPD_OUTPUT_ROOT/environment-v2"
nvidia-smi > "$OPD_OUTPUT_ROOT/environment-v2/nvidia-smi.txt"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/linux-cuda.txt
python -m pip check | tee "$OPD_OUTPUT_ROOT/environment-v2/pip-check.txt"
python -m pip freeze > "$OPD_OUTPUT_ROOT/environment-v2/pip-freeze.txt"
python -m torch.utils.collect_env > "$OPD_OUTPUT_ROOT/environment-v2/torch-env.txt"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
# Force only tiny CPU tests here: no GPU training is started.
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_approved_training_inputs.py tests/test_training_entries_cpu.py tests/test_opd_contract.py tests/test_losses_and_masks.py tests/test_opd_v2.py tests/test_code_evaluation_v2.py -q | tee "$OPD_OUTPUT_ROOT/environment-v2/cpu-tests.txt"
echo 'Bootstrap completed. Real Qwen, A100 memory and resume acceptance remain 未验证.'
