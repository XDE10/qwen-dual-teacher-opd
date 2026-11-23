# 断点与派生续训

一次训练分段保存完整 checkpoint。训练状态包括学生权重、AdamW 矩和步号、参数组、Python/NumPy/Torch/CUDA RNG、运行身份及排程位置。归档再保存生成参数、指标、版本锁、源码和文件清单。这样恢复时可以继续上一次优化器更新，推理评测则只需可验证的学生权重。

## 怎样恢复

1. 定位原运行的封存源码、批准数据、模型和完整断点。
2. 核对清单中每个文件的哈希，从编号最高的完整且同身份的 checkpoint 恢复。
3. 原身份内继续训练时保留配置、排程和输出目录。改训练总步数时另建派生运行，记录父 checkpoint 及运行身份哈希。排程只在父前缀后追加，已用样本保持原顺序。
4. 新端点先完整落盘，再校验云端权重与恢复材料，最后写入并读回归档回执。

[训练器](../src/training/latent_mopd_three.py)负责保存、校验、恢复和选择可续跑断点。[续训入口](../scripts/continue_latent_mopd_three.py)派生训练预算，[归档入口](../scripts/archive_latent_three_checkpoint.py)验证回传。仓库通用续训入口默认按每五步划分保存端点，显式结束步也保存。恢复本项目已有运行时，以该运行包记录的入口、参数和环境为准。

```bash
python -m scripts.continue_latent_mopd_three --help
python -m pytest -q tests/test_latent_three_continuation.py
```

同一运行预算内中断时，复用 [训练说明](latent_mopd_three.md#3-短跑正式训练和恢复) 中的原命令加 `--resume`。仓库中的派生入口绑定了项目实际封存的 62 步父包、池修复版清单和特定源码哈希。运行它前先取得对应归档。新建 release 和短跑照常使用训练入口恢复。

## 本项目怎样用

三教师运行在 Colab 经历重连与冷恢复，开发评测获得了 60、70、80 步的推理快照。选中的 70 步权重用于最终评测。推理快照可评测的事实和某个端点能恢复完整优化器是两项不同证据。运行归档按端点保留，实际恢复记录见[阶段报告](https://github.com/XDE10/qwen-three-teacher-latent-mopd/blob/61f021c1aad6d31415b2de4f905153363f11374b/reports/latent_mopd_three_status_20261118/README.md)和[收尾结果](../reports/latent_mopd_three_closeout_20261121/README.md)。
