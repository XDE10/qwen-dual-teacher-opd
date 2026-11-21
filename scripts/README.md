# 脚本入口

命令从仓库根目录执行。用 `python -m <模块> --help` 查看参数。

## 三教师 Latent-MOPD

- 数据构建和负责人批准见 [`prepare_latent_mopd_three`](prepare_latent_mopd_three.py)。
- 学生训练入口是 [`src.training.latent_mopd_three`](../src/training/latent_mopd_three.py)，`latent` 与 `token` 分支共用训练框架。
- 固定三域开发集见 [`evaluate_latent_mopd_three_dev`](evaluate_latent_mopd_three_dev.py)。代码候选交给 [`score_code_local_v3`](score_code_local_v3.py) 和 Docker。
- [`continue_latent_mopd_three`](continue_latent_mopd_three.py) 派生训练预算，[归档脚本](archive_latent_three_checkpoint.py) 按断点写回执。正式作业恢复以原运行包为准。
- `colab_latent_three_*.sh`、`probe_latent_three_*.py` 与迁移脚本的适用版本见 [训练说明](../docs/latent_mopd_three.md) 和 [断点说明](../docs/latent_mopd_three_continuation.md)。

## 历史入口与共享工具

- Qwen2.5 双教师训练用 [`train_project2_opd_v3`](train_project2_opd_v3.py)，旧数据发布和验证路径见 [v5 说明](../reports/project2_release_v5_20261108/README.md)。
- [`generate_eval_v3`](generate_eval_v3.py) 生成数学答案和代码候选，[官方基准恢复脚本](restore_official_benchmarks.py) 核对下载数据。
- 比较器、数据修复、早期上卡命令和固化实验入口见 [旧脚本索引](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc4644daf46beebffffb8c6759cd5a9e95c4f1b7/scripts/README.md)。
