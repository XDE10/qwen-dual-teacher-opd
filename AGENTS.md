# 项目二协作说明

从 [README](README.md)、[三教师实现与运行](docs/latent_mopd_three.md) 和 [最新评测报告](reports/latent_mopd_three_closeout_20261121/README.md) 开始。当前交付是 DeepSeek-R1-Distill-Qwen-1.5B 的三教师 Latent-MOPD 独立复现，最终评测使用本地 70 步。旧 Qwen2.5 双教师 OPD-v3 与 R1 双域试验各有 [历史入口](docs/README.md)。

## 改动前核对的契约

- 模型、数据和续训都以运行清单为准。训练只读负责人已批准的样本。候选、开发集与测试集分别留档，冻结的数据、批准记录和 SHA-256 按原版本保存。代为批准时保留负责人书面决定与时间。
- 学生模板只渲染一次，同一组 `input_ids` 和 `attention_mask` 传给教师。共同支持由词表审计给出。三教师训练使用连续 token ID 0–151664。
- 恢复训练时使用该断点实际封存的源码、配置、依赖和输出路径。正式三教师训练器 SHA-256 为 `c68797a2063ad5df7bf1d9e187ad94c0c37bbbf12894b26a14a518052fa29eb7`。延长预算要记录父身份与排程，保留权重、优化器和 RNG。
- 开发集用于选训练配置和 checkpoint。最终评测按提示词、样本数、提取器和判题器记录协议，保留原始分和诊断重测分。代码评测在 Docker 沙箱中执行生成程序。
- 在 WSL2 中运行 `python -m pytest -q`、`python tools/verify_repository.py` 和 `python tools/verify_project2_release_v5.py`。报告中记录实际运行命令与环境。
- 大模型、GPU 日志、断点及生成结果存放在独立运行归档。本机的大型文件与缓存优先放 D 盘。Git 提交携带代码、清单和摘要。`results` 数据分支独立管理。训练与评测发布时优先使用已配置的 P 盘归档。

[脚本索引](scripts/README.md)说明当前入口。[文档索引](docs/README.md)收录早期试验、交接和迁移台账。
