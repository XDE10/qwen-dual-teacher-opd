# 项目二协作说明

从 [README](README.md)、[项目结构](docs/README.md) 和 [评测报告](reports/latent_mopd_three_closeout_20261121/README.md) 开始。当前主线是三教师 Latent-MOPD，最终评测使用本地 70 步。

- 模型、数据和续训以运行清单为准。训练只读负责人已批准的样本。候选、开发集与测试集分别留档。批准记录和 SHA-256 按原版本保存。
- 学生模板只渲染一次，同一组 `input_ids` 和 `attention_mask` 传给教师。共同支持是连续 token ID 0–151664。
- 恢复用断点实际封存的源码、配置、依赖和输出路径。正式训练器 SHA-256 为 `c68797a2063ad5df7bf1d9e187ad94c0c37bbbf12894b26a14a518052fa29eb7`。延长预算记录父身份和排程，保留权重、优化器与 RNG。
- 开发集用于选配置和 checkpoint。终评记录提示词、样本数、提取器和判题器，原分与重测分分别留档。生成代码在 Docker 沙箱中执行。
- 从仓库根目录执行 `OMP_NUM_THREADS=2 python -m pytest -q` 和 `python tools/verify_repository.py`。旧版 v5 发布验证脚本及完整资料位于 [精简前快照](https://github.com/XDE10/qwen-three-teacher-latent-mopd/tree/61f021c1aad6d31415b2de4f905153363f11374b/tools)。
- 模型、大型日志、断点及生成结果使用独立运行归档。本机大文件优先放 D 盘。正式训练和评测优先使用已配置的 P 盘归档。Git 保存源码、必要的小型运行输入和结果摘要。
