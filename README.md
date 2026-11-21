# 三教师 Latent-MOPD

在单张 GPU 上把数学、代码和逻辑教师的能力蒸馏到同一个 1.5B 推理模型。学生 **DeepSeek-R1-Distill-Qwen-1.5B** 在线采样自己的回答，然后由 JustRL、Archer2 和 Nemotron 按题目领域为同一条 token 轨迹打分。训练同时使用候选 token 的分布信号和最后三层的表示对齐。

本项目独立复现 [Latent-MOPD 同家族 last-3](https://arxiv.org/html/2610.02381v1#S4.T1)。本地开发集比较了 60、70、80 步，选用 **70 步**做最终九项评测。原始分数和改进评测流程后的分数见 [收尾报告](reports/latent_mopd_three_closeout_20261121/README.md)。论文 Base 和 62 步使用发表值。

## 从哪开始

- [构建数据、核对模型、启动训练](docs/latent_mopd_three.md)
- [读懂断点、归档与继续训练](docs/latent_mopd_three_continuation.md)
- [开发集选型与九项最终评测](docs/latent_mopd_three_final_evaluation.md)
- [按任务找脚本](scripts/README.md)

## 本地验证

在仓库根目录使用 Python 3.12 的 WSL 环境。轻量检查使用 `requirements/wsl2-cpu.txt`。

```bash
python -m pytest -q
python tools/verify_repository.py
python tools/verify_project2_release_v5.py
```

需要训练时，先生成候选池并核对清单。由负责人批准后，训练入口读取完整 release。每步固定一个领域，32 道题各采样 4 个回答，单个回答最多 16,384 tokens。

```bash
python -m src.training.latent_mopd_three --help
python -m scripts.prepare_latent_mopd_three --help
python -m scripts.evaluate_latent_mopd_three_dev --help
```

## 目录怎么读

- `src/training/` 训练循环、采样、运行身份和断点。`src/losses/` 实现 token 与表示损失。
- `src/data/` 管理已批准的输入和沙箱，`src/teachers/` 管理词表与输入契约，`src/evaluation/` 负责判分。
- `scripts/` 放可调用入口。`configs/` 和 `requirements/` 放已版本化的配置与依赖。
- `data/` 和 `provenance/` 保留数据清单及校验需要的冻结证据。大型模型、运行生成和断点存放在运行归档。
- `docs/` 说明用法，`reports/` 存版本化结果，`tests/` 和 `tools/` 核验实现与证据。

旧 Qwen2.5 双教师 OPD、R1 双域验证和历史交接按 [文档索引](docs/README.md) 查找。历史代码入口和数据验证仍可独立使用。
