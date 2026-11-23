# 主线运行输入

- `project2_release_v5_20261108/release/` 保留三教师准备和开发评测实际读取的数学/代码 dev 和原始清单。两份 JSONL 按旧清单校验哈希。完整 v5 训练发布及审批记录可在 [精简前快照](https://github.com/XDE10/qwen-three-teacher-latent-mopd/tree/61f021c1aad6d31415b2de4f905153363f11374b/data/project2_release_v5_20261108) 查阅。
- `evaluation/reviewed_comparators.json` 按题目身份给代码 dev 选择判题规则，评测运行时会与 dev 文件哈希核对。
- `math/math500_test.jsonl` 参与构建三教师数据时的重合题过滤。GSM8K 测试文件供共享判分模块的基准入口读取。
- 官方 EvalPlus 数据由 [恢复入口](../scripts/restore_official_benchmarks.py) 下载、解压并核对哈希。三域训练候选及批准后的 release 在 `runs/` 中生成。

当前模型成绩在 [收尾报告](../reports/latent_mopd_three_closeout_20261121/README.md)。完整生成、权重和优化器留在运行归档。
