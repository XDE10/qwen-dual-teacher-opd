# 从公开数据到三教师训练

首页的 [CPU 快速检查](../README.md#下载后先跑一次) 用小模型执行一次真实更新。本页使用项目固定的 1.5B 学生和数学、代码、逻辑教师。训练需要一张可见的 BF16 CUDA GPU。历史 16K 正式训练实测使用 RTX PRO 6000 Blackwell 96 GB。模型、下载数据和 checkpoint 需要额外磁盘空间。

## 1. 建立 CUDA 环境

在 Linux、WSL2 或 Colab 的 Python 3.12 环境中，从仓库根目录执行。单机有多张卡时通过 `CUDA_VISIBLE_DEVICES` 选择一张。以下用 CUDA 12.8 版 PyTorch。

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/linux-cuda-v3.txt
python -m pip check
python -c 'import torch; assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'
```

## 2. 下载、核对并准备训练题

安装 `curl`，保持对 Hugging Face 和 GitHub Releases 的网络访问。三个命令可在 CPU 上执行。

```bash
python -m src.training.latent_mopd_three audit
python -m scripts.restore_official_benchmarks
python -m scripts.prepare_latent_mopd_three build
```

`audit` 下载并核对四个固定模型版本的配置和 tokenizer，不下载权重。`build` 从固定版本的 DAPO-Math、OpenCodeReasoning 和公开评测集下载素材，并用 Reasoning Gym 0.1.19 生成逻辑题。生成 `runs/latent_mopd_three/candidate/` 后，先审核其中的清单和样本。审批命令会核验被审核的清单 SHA-256，并把审批人和说明写进发布清单。

```bash
sha256sum runs/latent_mopd_three/candidate/candidate_manifest.json
read -r -p '审核过的 SHA-256: ' candidate_sha256
read -r -p '负责人: ' approver
read -r -p '审批说明: ' approval_note
python -m scripts.prepare_latent_mopd_three approve \
  --candidate-sha256 "$candidate_sha256" --approver "$approver" --note "$approval_note"
```

`runs/latent_mopd_three/release/` 是完整批准发布。将数据准备和训练拆到不同机器时，原样传输整个 release 目录。

## 3. 短跑、正式训练和恢复

以下示例禁用项目私有网盘模型缓存，通过 Hugging Face 获取并校验四个模型的固定权重。首次运行会下载模型。换机器时可用受信任的缓存，清单和权重仍按原校验逻辑检查。

```bash
manifest=runs/latent_mopd_three/release/manifest.json
python -m src.training.latent_mopd_three latent \
  --manifest "$manifest" --model-drive '' --output-root runs/smoke \
  --steps 6 --prompts 1 --responses 1 --rollout-batch 1 --max-new-tokens 1024
```

六步覆盖三域纯 latent 权重及第一轮混合损失，输出 `runs/smoke/latent_mopd_three_latent/`。正式实验单独建输出目录。

```bash
python -m src.training.latent_mopd_three latent \
  --manifest "$manifest" --model-drive '' --output-root runs/formal
```

正式默认 62 步，每步一个域，32 题各生成 4 条，最多 16,384 个新 token。要先跑一段，可在正式命令加 `--stop-at-step 6`。接着移除该参数并加 `--resume`，保持其他启动参数相同。想延长已封存的完整预算时，走 [派生续训](latent_mopd_three_continuation.md)。

## 4. 验证结果

`metrics.jsonl` 每行对应一次完成的 AdamW 更新。检查最后一步号及 checkpoint 保存事件，再用 [三域开发集流程](latent_mopd_three_final_evaluation.md#运行开发集) 验证模型。[项目结构](README.md) 讲解脚本和张量执行顺序。
