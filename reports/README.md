# reports/ 说明

本目录保存**项目二的版本化审核与验收记录**。历史实验报告（项目一/三、strict_v8、topup_v6、data_revision_v4 等）已在 2026-11-08 重排中删除；删除台账见 [docs/migration/deletion_ledger.json](../docs/migration/deletion_ledger.json)，重排前快照见 [docs/migration/inventory_before_reorg.json](../docs/migration/inventory_before_reorg.json)。

## 保留内容

| 路径 | 内容 | 性质 |
| --- | --- | --- |
| `code_training_repair_20261025_v4/` | **当前 v4 发布验收证据**：准入决策、来源核验、受保护文件、执行/测试日志、发布摘要、沙箱环境、被拒绝候选 | **当前验收证据** |
| `project2_release_v5_20261108/` | v5 候选审计：全部 5456 条 prompt 的预算复核、decontamination_v5 命中证据、补题统计、筛除历史说明与批准步骤（`README.md`、`audit.json`） | **当前审计**（v5 已批准） |
| `g4_first_session_20261109/` | 第一次上卡（Colab G4）：环境、2 步 smoke 的耗时与显存、教师适配度探针的 KL 矩阵，以及据此做的配置改动；原始文件在 `results` 分支 | **GPU 实测记录**（只到 smoke 与探针） |
| `g4_session2_lr_sweep_20261109/` | 第二次上卡：三个学习率各 30 步完整批次的吞吐、显存、KL/熵/长度/EOS/在线准确率，配对比较，学生熵下降的成因分析，关卡 v1→v2，学习率决定；原始文件在 `results` 分支 | **GPU 实测记录**（训练遥测，没有评测分数） |
| `g4_opd_v3_experiments_20261112/` | 第三至第六次上卡与本地 Docker 评测汇总：D0、SFT（`SeqKD-Mix`）、D3（`7B` 单通用教师）、D4（`Math-7B+Coder-7B` 含 `3e-5`、`1e-4`、近端教师 `prox07`）以及先验对齐双教师 `d4_math14b_lr1e-4`（`14B-Instruct+Coder-7B`）的 80 步训练与 6 大评测集完整结果（`README.md`、`evaluation_matrix.json`） | **GPU 正式训练与完整评测记录** |
| [`wsl_evalplus_crash_fix_20261113/`](wsl_evalplus_crash_fix_20261113/README.md) | WSL Docker 中 EvalPlus 0.3.1 崩溃诊断、资源策略及两道 MBPP 样本的验证 | **评测工程验证**（完整榜单未重评分） |
| `handover_20261024_v3/` | 2026-10-25 v3 审核与实施记录：200 题比较器逐项决策（192 reviewed / 8 blocked）、去重、Docker 环境、v3 配置与测试日志 | **历史审核记录**（v3 的 2499 池已被 v4 取代） |
| 根目录当前审核 json/md（15 个文件） | tokenizer 对齐审计（含 14B 扩展审计）、提示词契约审计、去重、人工审核包、代码接口/官方基准重叠/沙箱/CPU 环境、技术审查复核记录等 | **当前审核依据** |

## 当前 v4 验收证据（`code_training_repair_20261025_v4/`）

- `README.md`：补题发布说明、来源与准入决策、实际检查与拒绝记录。
- `admission_decision.json`：v4 准入决策与证据文件哈希；`source_verification.json`：源 parquet 行级核验。
- `verification_summary.json`：`tools/verify_code_repair_v4.py` 的输出（文件/批准/接口配额/逐行字节保留）。
- `protected_before.json`：旧发布受保护文件在改动前快照。
- `pytest_admission.log`、`pytest_all_cpu.log`：本轮准入与合并 CPU 测试日志。
- `sandbox_environment.json`：Docker 环境记录（源用例与独立 oracle 检查）。
- `rejected/`：本轮拒绝候选（如 `code/vfc_90438`、`code/vfc_121103`）的完整证据。
- `project2_opd_v4_*.json`：v4 配置副本；根 `configs/` 为执行版本。

这些是**有限源测试 + 独立小规模 oracle + 静态审核**证据，不是穷尽正确性证明，也不构成 GPU 训练或能力验收。

## v3 审核记录（`handover_20261024_v3/`）

- `reviewed_comparators.json`：200 个内部 dev 的逐题比较器决策（192 reviewed、8 blocked），被 `evaluate_code_v2` 学生入口使用。
- `training_exclusions.json`：v3 的采样排除 sidecar（v4 manifest 把它列为输入证据）。
- `code_evaluation/`、`docker_environment.json`：参考解自检与 Docker 环境。
- `decontamination_review_20261025.json`：去重审查结果。
- `delivery_hashes.json`、`protected_before.json`、`verification_summary.json`：交付身份与保护项。

v3 只说明当时的 2499 有效代码池与审核状态；当前执行数据是 v4 的 2500/2500，不得用 v3 数量描述当前。

## 冻结发布证据 vs 当前验收证据

- **冻结发布身份**：被 `data/portable_release/manifest.json`、v4 `release/manifest.json` 等清单用 sha256 锚定的证据文件；旧路径已在重排中删除的，镜像在 [provenance/](../provenance/README.md)，由校验器回退读取。它们**不允许改写**。
- **当前验收证据**：本目录的执行日志、审核 json/md 与 `verification_summary.json`。它们是本地记录，可追加新版本，但不能覆盖或修改旧版本；合法差异要单独披露，不能靠刷新哈希"修好"。

## 使用规则

- 报告必须区分「代码已实现」「CPU 测试通过」「真实模型跑通」「能力提升」；未运行的检查保持未验证。
- CPU 测试、2 步 smoke 与 tiny 模型恢复测试不代表真实 Qwen / GPU / 能力结果。
- 不覆盖旧报告；新增版本用新目录/文件，并记录运行命令、环境、数据/模型身份与差异。
- 代码评测必须走 Docker 沙箱，数据集代码不在宿主执行；无 Docker 时停止并报告。

## 复现/校验

```bash
D:/anaconda3/envs/genai-test/python.exe -m pytest -q
D:/anaconda3/envs/genai-test/python.exe tools/verify_code_repair_v4.py
D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py
```

`verify_code_repair_v4.py` 会重写 `reports/code_training_repair_20261025_v4/verification_summary.json`；该文件是校验输出，不是新的训练结果。
