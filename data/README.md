# data/ 说明

本目录只保留**项目二当前发布、历史迁移基线与评测/源池所需文件**。原始采集数据、候选池与项目一/三数据已在 2026-11-08 重排中删除；删除台账见 [docs/migration/deletion_ledger.json](../docs/migration/deletion_ledger.json)，重排前快照见 [docs/migration/inventory_before_reorg.json](../docs/migration/inventory_before_reorg.json)。

## 保留内容

| 路径 | 内容 | 性质 |
| --- | --- | --- |
| `code_training_repair_20261025_v4/release/` | 当前 **v4 修复发布**：math/code 的 train/dev jsonl + `manifest.json` + `reserved_dev_ids_never_train.json` | **当前执行数据**（根 `configs/project2_opd_v4_*.json` 绑定） |
| `portable_release/` | 2026-10-24 迁移时的历史发布基线（同 6 个文件） | **冻结基线**，逐字节保留；默认执行不用它 |
| `project2_release_v5_20261108/candidate/` | **v5 候选**：v4 去掉与评测/dev 重合的 133 条数学、12 条代码，数学按同 level 补 133 条（`training_approved=false`），并附数学参考答案 sidecar | **待负责人批准**；批准后才生成 `release/`，opd-v3 实验配置指向它 |
| `code_benchmarks/official/` | HumanEval+ v0.1.10、MBPP+ v0.2.0 官方文件 | 不入库（gitignore）；由 `scripts/restore_official_benchmarks.py` 恢复并按 sha256 校验 |
| `tokenizers/` | 4 个 tokenizer 快照：`Qwen2.5-1.5B-Instruct`、`Qwen2.5-7B-Instruct`、`Qwen2.5-Coder-7B-Instruct`、`Qwen2.5-Math-7B-Instruct` | 模板/词表身份证据，离线可用 |
| `math/` | 数学源池与最终评测集：`math_train.jsonl`（7184）、`math_dev.jsonl`（256）、`math_quarantine.jsonl`（60）、`math_source_test.jsonl`（5000）、`math500_test.jsonl`（500）、`gsm8k_train.jsonl`（7473）、`gsm8k_test.jsonl`（1319） | 源池/评测，不代表训练批准 |
| `math_manifest.json` | 数学数据来源、revision、去重与文件 sha256 记录 | 溯源 |

## 当前 v4 发布（`code_training_repair_20261025_v4/release/`）

- 数学：**2500 train / 256 dev**；代码：**2500 train / 200 dev**（train/dev 中 function 与 stdin/stdout 各 1250 / 100）。
- `manifest.json` 的 `approved_training_files` = `project2_math_train.jsonl`、`project2_code_train.jsonl`；`approved_evaluation_files` = `project2_math_dev.jsonl`、`project2_code_dev.jsonl`；`protected_dev_ids_file` = `reserved_dev_ids_never_train.json`。
- v4 相对 `portable_release` 只替换代码训练文件 1 行：移除与 dev 同题的 `code/vfc_121257`，加入 `code/vfc_96076`；其余 2499 行、数学 train/dev、代码 dev、保护 ID 均逐字节不变。旧发布 manifest/批准/SHA 未修改。
- 每个批准行仍要通过文件 SHA、逐行 `training_approved`、唯一 ID、模型输入白名单与 dev 隔离检查。代码发布证据在 [reports/code_training_repair_20261025_v4/](../reports/code_training_repair_20261025_v4/README.md)。

## 批准边界（必须遵守）

- 训练/评测范围**只认 manifest** 的 `approved_training_files` / `approved_evaluation_files` 与逐行批准；候选、隔离、AI 审核通过都不自动等于批准。
- dev 只用于评测/调参，**不得**进训练。
- **不得手改批准字段或哈希**来绕过检查；新数据版本必须独立取证、保留旧发布。
- 数据文件本身不含"可以训练"的隐含授权；模型输入只走代码里的白名单视图（参考/答案/测试/审核字段不进入 OPD 问题提示）。

## 原始采集数据与台账

重排删除了 **130 个顶层项**：台账合计 4225 个文件 / 4,380,095,420 字节（约 4.38 GB），其中原始采集/候选/中间数据与历史报告等为 4093 个文件，另 132 个为项目一/三的源码与测试文件（`screening_sources/`、`code_v2_sources/`、`topup_v6/`、`project2_code/`、`strict_v8/` 等）；逐项文件与字节数以 [docs/migration/deletion_ledger.json](../docs/migration/deletion_ledger.json) 为准，完整范围见重排说明。需要追查某个旧数据文件时：

1. 查 [docs/migration/path_mapping.json](../docs/migration/path_mapping.json)：可能是 `kept`（仍在 `data/`）、`mirrored`（在 `provenance/`）或 `moved`。
2. 查 [docs/migration/deletion_ledger.json](../docs/migration/deletion_ledger.json)：本次删除清单与统计。
3. 查 [docs/migration/inventory_before_reorg.json](../docs/migration/inventory_before_reorg.json)：重排前每个文件的 `[bytes, sha256]`，用于核对从外部副本恢复的文件是否原件。
4. 被冻结发布清单锚定的证据文件在 [provenance/](../provenance/README.md) 中按原字节镜像，校验器会回退读取。

## 校验

```bash
D:/anaconda3/envs/genai-test/python.exe tools/verify_code_repair_v4.py
D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py
```

- `verify_code_repair_v4.py` 校验 v4 manifest 文件 SHA、输入证据、受保护文件、批准行、接口配额与逐行字节保留。
- `verify_repository.py` 校验重排后布局、`path_mapping` 与 `provenance` 镜像。
- 代码评测必须走 Docker 沙箱，数据集代码不在宿主执行；校验通过不等于 GPU 训练或能力验收。
