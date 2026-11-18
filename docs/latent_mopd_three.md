# 三教师 Latent-MOPD 独立复现

实现以原论文 Table 1 的同家族 last-3 配置为目标。固定 DeepSeek-R1-Distill-Qwen-1.5B 学生和 JustRL、Archer2、Nemotron-Research-Reasoning-Qwen-1.5B 教师。Latent 与 token 代码路径共享输入顺序、rollout 预算与优化器；当前正式执行的是 Latent，匹配预算的 token-only 对照尚未运行。Latent 组另外对齐最后三层，每位教师前 10 次更新逐步切换到 token 损失。运行使用单卡 HF rollout、BF16 计算与 FP32 全参数 AdamW。

原作者的训练批次与提示词未公开。本实现从 DAPO-Math-17k、OpenCodeReasoning 和 Reasoning Gym 0.1.19 独立构建候选。DAPO 保留原始 Answer 行指令，代码使用题面，逻辑使用包内默认的 `<answer>` 指令。逻辑任务表和种子是本次重建的选择。数据排序由样本 ID 的固定哈希确定，训练中按清单顺序读取。学生 tokenizer 被传入 `enable_thinking=False`，固定模板仍会渲染末尾的 `<think>` 前缀。Rollout 与候选 log-prob 的归一化均使用审计通过的 151,665 个 token ID。

当前部署与中期结果见 [2026-11-18 状态报告](../reports/latent_mopd_three_status_20261118/README.md)，后续恢复见 [完整断点与延长训练](latent_mopd_three_continuation.md)。历史短跑与早期迁移命令的运行身份可能与当前断点不同；恢复正式任务时必须使用其封存版本。

- [主论文，Table 1 与附录 A/C](https://arxiv.org/html/2610.02381v1)
- [公开独立复现的提示词与长度过滤，附录 B](https://arxiv.org/html/2611.02746v2#A2)

## WSL 本地构建与验收

在项目 Python 环境安装 `pyarrow==21.0.0` 和 `reasoning-gym==0.1.19`。从仓库根目录执行。

```bash
python -m src.training.latent_mopd_three audit
python -m scripts.restore_official_benchmarks
python -m scripts.prepare_latent_mopd_three build
python -m pytest -q tests/test_opd_v3.py -k latent_mopd_three
```

候选清单在 `runs/latent_mopd_three/candidate/candidate_manifest.json`。它记录数据文件哈希、过滤计数和任务生成配置。`schedule.jsonl` 是锁定的 62 步顺序。项目负责人核对候选后执行发布，参数填入实际审查过的清单 SHA-256 和批准记录。

```bash
python -m scripts.prepare_latent_mopd_three approve \
  --candidate-sha256 "$candidate_sha256" --approver "$approver" --note "$approval_note"
```

## Colab 训练命令

把已批准的 release 目录作为整体传到 Colab 的 `/content/p2_latent_three_release`，使用项目标准 Colab 环境。从 Colab 仓库根目录运行短跑。

```bash
python -m src.training.latent_mopd_three latent \
  --manifest /content/p2_latent_three_release/manifest.json \
  --output-root /content/runs/latent_mopd_three_smoke \
  --steps 6 --prompts 1 --max-new-tokens 1024
```

短跑覆盖三个纯 Latent 步和三个混合损失步。正式预算试跑使用正式命令加 `--stop-at-step 6`，保留默认每步 32 题、每题 4 个回答和 16,384 token 上限。通过后加 `--resume` 接着跑。每个运行默认保留最近两个完整检查点。

续训从编号最高的检查点开始验证，跳过文件损坏或运行身份不匹配的快照。找到可用快照后归档较晚的检查点和训练日志。无匹配快照时退出。

```bash
python -m src.training.latent_mopd_three latent \
  --manifest /content/p2_latent_three_release/manifest.json --output-root /content/runs
python -m src.training.latent_mopd_three token \
  --manifest /content/p2_latent_three_release/manifest.json --output-root /content/runs
```

## 三域 dev 对照

GPU 环境另装 `reasoning-gym==0.1.19` 后执行。Base 省略 `--checkpoint`。逻辑评分读取冻结题面、参考答案和 verifier 元数据，避免生成器在不同 Python 进程里产出不同实例。代码候选传回 WSL 后用 `scripts.score_code_local_v3` 和 Docker 评分。论文 Table 1 的整套最终评测尚需独立验证。

指定 `--checkpoint` 时，入口先校验封存文件清单、学生权重及全部分片，再开始加载模型和生成。评测只读模型，优化器文件的字节校验留在续训入口。

```bash
python -m scripts.evaluate_latent_mopd_three_dev \
  --manifest /content/p2_latent_three_release/manifest.json \
  --checkpoint /content/runs/latent_mopd_three_latent/checkpoint-62 \
  --output-dir /content/runs/latent_mopd_three_dev/latent
```
