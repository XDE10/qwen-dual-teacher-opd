# 三教师 Latent-MOPD · G4 三项探针

本次只运行六步短跑、32 × 4 正式规模单步和 16K 联合损失三个探针。使用同一个 G4 会话 `p2-latent-three-g4`，按顺序执行。正式 release 已记录项目负责人的书面批准。三教师身份、批准数据、损失、采样与优化器沿用本地通过验收的实现。

## 1. WSL 预先上传资产

先在 WSL 将小部署包和固定模型缓存放到同一账户的 `mydrive:`。部署包只有本次工作区覆盖文件、已批准的 release 和引导脚本需要核验的评测资产。模型缓存单独存储。下列命令可以在开 G4 前执行，Drive 已有相同文件时 `--checksum` 会跳过。

```zsh
cd /home/xde10/projects/qwen-dual-teacher-opd
rclone copyto "$PWD/runs/latent_mopd_three/g4/latent_three_g4_20261115.tar.gz" mydrive:opd_inputs/latent_three_g4_20261115.tar.gz --checksum &&
rclone copy "$PWD/runs/latent_mopd_three_assets/weights" mydrive:models --exclude '**/.cache/**' --checksum --progress
```

完成后再创建 G4。若侧边栏已创建同名 G4 会话，从状态检查开始。两个凭据命令把现有配置直接传到自己的 VM。从状态输出确认分配的是 G4。

```zsh
"$HOME/.local/bin/colab" new -s p2-latent-three-g4 --gpu G4
"$HOME/.local/bin/colab" status -s p2-latent-three-g4
"$HOME/.local/bin/colab" upload -s p2-latent-three-g4 "$HOME/.config/rclone/rclone.conf" /content/.colab-drive-rclone-upload.conf
"$HOME/.local/bin/colab" upload -s p2-latent-three-g4 "$HOME/.config/opd/gh_credentials" /content/gh_credentials
```

## 2. 首次在该 VM 部署

从 WSL 粘贴以下命令。它使用固定 Git 基线、覆盖本地送审后修复、核对部署包，然后引导 torch 2.8.0/cu128。首次短跑还会拉取并校验四个模型缓存，后两项在同一台 VM 复用。Disk 上会同时容纳约 17 GiB 的模型缓存和两份全参数快照，另有环境和临时文件开销。脚本各留一个最终检查点，短跑替换检查点时短暂保留前一份。

```zsh
"$HOME/.local/bin/colab" exec -s p2-latent-three-g4 --timeout 86400 <<'PY'
%%bash
set -euo pipefail
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
df -h /content
mkdir -p "$HOME/.config/rclone" /content/project2
install -m 600 /content/.colab-drive-rclone-upload.conf "$HOME/.config/rclone/rclone.conf"
chmod 600 /content/gh_credentials
command -v rclone >/dev/null || { apt-get update && apt-get install -y rclone; }
if [[ ! -d /content/project2/.git ]]; then
  git init /content/project2
  git -C /content/project2 remote add origin https://github.com/XDE10/qwen-dual-teacher-opd.git
fi
git -C /content/project2 config credential.helper 'store --file=/content/gh_credentials'
git -C /content/project2 fetch --depth 1 origin 4dee8bfbc554a22461d4d347f4973fc7fb7a016e
git -C /content/project2 checkout --detach FETCH_HEAD
rclone copyto mydrive:opd_inputs/latent_three_g4_20261115.tar.gz /content/latent_three_g4_20261115.tar.gz
printf '%s  %s\n' df927bae5d1c8f89a56614eb5b214c96a777a6fd65756e99d2a55e39db5550c6 /content/latent_three_g4_20261115.tar.gz | sha256sum -c
cd /content/project2
tar -xzf /content/latent_three_g4_20261115.tar.gz
sha256sum -c runs/latent_mopd_three/overlay-checksums.sha256
PROJECT_ROOT=/content/project2 OPD_OUTPUT_ROOT=/content/runs INSTALL_VLLM=0 bash scripts/colab_bootstrap_v3.sh
PY
```

## 3. 三条启动命令

每条命令从 WSL 执行。看到对应的 `LATENT_THREE_PROBE_OK` 和任务 `finished with exit status 0` 后再运行下一条。

### 六步短跑

每步 1 题 × 4 答，最多 1,024 新 token。覆盖三位教师各一次纯 Latent、一次联合更新。检查点保存在本 VM。

```zsh
"$HOME/.local/bin/colab" exec -s p2-latent-three-g4 --timeout 86400 <<'PY'
%run /content/project2/scripts/colab_job_v3.py latent-three-smoke --publish --drive mydrive:opd_runs --sync-every 5 --save evidence=/content/runs/latent_three_g4/evidence/smoke -- bash scripts/colab_latent_three_probes.sh smoke
PY
```

### 32 × 4 正式规模单步

从 Base R1 独立起步，32 题 × 4 答，最多 16,384 新 token，并发生成 4 条。只跑数学第 1 步，按论文调度使用纯 Latent。记录 128 条回答、一次优化器更新、用时、显存与 checkpoint-1。该检查点供下一项使用。

```zsh
"$HOME/.local/bin/colab" exec -s p2-latent-three-g4 --timeout 86400 <<'PY'
%run /content/project2/scripts/colab_job_v3.py latent-three-scale --publish --drive mydrive:opd_runs --sync-every 5 --save evidence=/content/runs/latent_three_g4/evidence/scale -- bash scripts/colab_latent_three_probes.sh scale
PY
```

### 16K 联合损失

加载上一步的真实学生和 AdamW 状态，三位教师常驻。用已批准逻辑样本的最长正式前缀和一条恰好 16,384 token 的固定合成回答，走 top-16 打分、0.1 token + 0.9 latent 联合损失和反传。为纯 Latent 首步尚未分配的 AdamW moments 预留等量显存。输出显存峰值、耗时、原始损失和梯度范数。优化器更新计数为 0，此项独立于正式训练计数。单长序列测量覆盖这一个联合微批的内存峰值。

```zsh
"$HOME/.local/bin/colab" exec -s p2-latent-three-g4 --timeout 86400 <<'PY'
%run /content/project2/scripts/colab_job_v3.py latent-three-joint16k --publish --drive mydrive:opd_runs --sync-every 5 --save evidence=/content/runs/latent_three_g4/evidence/joint16k -- bash scripts/colab_latent_three_probes.sh joint16k
PY
```

## 4. 给侧边栏的监督入口

每项的 job 名依次为 `latent-three-smoke`、`latent-three-scale`、`latent-three-joint16k`。在另一个 WSL 终端取日志，避免同个 kernel 排队。下面将变量换成当前 job 名。

```zsh
opd_g4_probe_job=latent-three-smoke
"$HOME/.local/bin/colab" download -s p2-latent-three-g4 "/content/jobs/$opd_g4_probe_job/output.log" "/tmp/$opd_g4_probe_job.log" &&
tail -n 40 "/tmp/$opd_g4_probe_job.log"
```

训练进度看 `rollout_progress`、`optimizer_done`、`checkpoint_saved`。16K 诊断看 `Candidate scoring`、`Joint objective and backward` 和最终 JSON。后台任务在终端断开或 Ctrl+C 后继续运行。Drive 和 `results` 分支回传 job 日志、检查点清单和指标文本，权重和优化器状态留在同一 VM。三项结束并确认保存日志后可释放 G4。

```zsh
"$HOME/.local/bin/colab" stop -s p2-latent-three-g4
```

三条命令的启动参数、部署压缩包、数据发布和 CPU 入口已本地验证。实际 CUDA 执行由负责人启动和侧边栏监督。
