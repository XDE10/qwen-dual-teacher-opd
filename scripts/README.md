# 命令入口

从仓库根目录用 `python -m <模块> --help` 查看参数。CPU 验证看 [首页](../README.md#下载后先跑一次)。

- 数据准备和审批用 [`scripts.prepare_latent_mopd_three`](prepare_latent_mopd_three.py)，官方代码基准用 [`scripts.restore_official_benchmarks`](restore_official_benchmarks.py) 恢复。
- 正式训练用 [`src.training.latent_mopd_three`](../src/training/latent_mopd_three.py) 的 `latent` 子命令。小预算与正式启动示例见 [训练说明](../docs/latent_mopd_three.md)。
- 开发集生成用 [`scripts.evaluate_latent_mopd_three_dev`](evaluate_latent_mopd_three_dev.py)，代码候选用 [`scripts.score_code_local_v3`](score_code_local_v3.py) 在 Docker 中判分。
- [`scripts.continue_latent_mopd_three`](continue_latent_mopd_three.py) 和 [归档模块](archive_latent_three_checkpoint.py) 用于已封存实验，所需运行包见 [续训说明](../docs/latent_mopd_three_continuation.md)。

`train_project2_opd_v3.py` 的路径与内容由三教师运行身份引用。完整的早期实验和相应入口可在 [精简前快照](https://github.com/XDE10/qwen-three-teacher-latent-mopd/tree/61f021c1aad6d31415b2de4f905153363f11374b/scripts) 查阅。
