# scripts/ 入口索引

所有命令从**项目根目录**执行。默认 `python` 是 Python 3.8，**不可用**；请用 `D:/anaconda3/envs/genai-test/python.exe`（Python 3.12），或先 `conda activate genai-test` 再使用 `python`。

本目录共 **35 个文件**（33 个 `.py` + 2 个 `.sh`），按职责分 7 组；第 0 组是 2026-11-08 新增的 opd-v3 / v5 数据 / 评测 v3 入口。标注「不能直接重跑」的脚本依赖本次重排已删除的原始采集/基准数据，只作**方法与历史记录参考**；删除明细见 [docs/migration/deletion_ledger.json](../docs/migration/deletion_ledger.json)。

## 0. opd-v3 / v5 数据 / 评测 v3（7，当前）

| 文件 | 说明 |
| --- | --- |
| `train_project2_opd_v3.py` | opd-v3 训练入口（MOPD 式混合批次双教师）；配置 `configs/project2_opd_v3_*.json`；设计见 `docs/opd_v3_mopd.md` |
| `generate_eval_v3.py` | 评测协议 v3：vLLM（`--engine hf` 回退）greedy 生成；数学同时用 Math-Verify 和保守判分器评分；代码只写 `*.candidates.jsonl` 交本地 Docker；`--self-check` 测判分器覆盖率 |
| `restore_official_benchmarks.py` | 从 EvalPlus 官方 release 恢复 HumanEval+ v0.1.10 / MBPP+ v0.2.0，按 `inventory_before_reorg.json` 的 sha256 校验后才写入 |
| `build_project2_release_v5.py` | 生成 v5 **候选**：重查 prompt 预算、decontamination_v5、去掉重合行、从封存池按同 level 补数学；不批准任何东西 |
| `approve_project2_release_v5.py` | **仅负责人执行**：绑定候选与审计报告的摘要，生成 `release/`（保留行逐字节不变，永不覆盖） |
| `prefetch_models_v3.py` | 每个 Colab 会话先按配置下载固定 revision 的学生/教师权重 |
| `colab_bootstrap_v3.sh` | A100 / G4（sm_120）环境引导：cu128 torch 2.8.0、架构检查、恢复基准、跑 CPU 测试；可选装 vLLM |

## 1. 训练入口（2）

| 文件 | 说明 |
| --- | --- |
| `train_project2_opd.py` | **当前 v2 CLI 入口**，内部 `from src.training.opd_v2 import cli`。示例：`D:/anaconda3/envs/genai-test/python.exe -m scripts.train_project2_opd --config configs/project2_opd_v4_smoke.json --output-root runs/opd_smoke` |
| `train_project2_opd_v1.py` | 历史 v1 入口（`src.training.opd.cli`），仅追溯，不推荐新实验 |

## 2. 评测（8）

| 文件 | 说明 | 可重跑性 |
| --- | --- | --- |
| `evaluate_approved_math.py` | 固定 greedy 数学评测，封存 JSONL 与模型身份；使用 `math_verifier.check()` 轻量 API | 可（需模型与 `data/math/`） |
| `evaluate_code_v2.py` | 内部 dev 代码评测，显式 comparator 注册表 + Docker-only；默认 `--data data/portable_release/project2_code_dev.jsonl` | 可（需 Docker） |
| `evaluate_evalplus_v2.py` | 官方 HumanEval+/MBPP+ 完整解准备，可选调用封存 Docker EvalPlus | **不能直接重跑**：依赖已删除的 `data/code_benchmarks/official/*.jsonl` |
| `code_eval_worker_v2.py` | Docker-only 候选 worker v2；宿主不执行数据集代码 | 由评测脚本调用 |
| `code_eval_worker_v3.py` | v3 有限测试比较器（非完整规格证明） | 由评测脚本调用 |
| `code_sandbox_worker.py` | 容器专用 worker；**Windows 宿主禁止 import/执行** | 仅容器内 |
| `check_generation_config_colab_v2.py` | 只读 pinned generation/config 元数据，不下载权重 | 可 |
| `estimate_opd_memory_v2.py` | 解析式显存/计算上界估算，**不是 GPU 实测** | 可 |

## 3. 审计（2）

| 文件 | 说明 |
| --- | --- |
| `audit_project2_prompt_contract.py` | 离线 pinned-tokenizer 契约审计；数学池已改指 `data/code_training_repair_20261025_v4/release/project2_math_train.jsonl`，报告 `reports/project2_prompt_contract_audit.json` 为重排后合法重生成 |
| `revalidate_tokenizer_alignment.py` | 四模型 tokenizer 对齐审计的离线重放，不加载权重/不联网 |

## 4. v4 主线数据修复/验收（7）

| 文件 | 说明 | 可重跑性 |
| --- | --- | --- |
| `prepare_code_training_repair_v4.py` | 只读 reserve inventory + 单个替换 prompt 的准入检查 | **不能完整重跑**：依赖已删的 `data/topup_v6_20261024/`、`data/project2_code/`、`data/code_benchmarks/`（部分源路径有 `path_mapping`/`provenance` 回退） |
| `export_code_training_repair_v4.py` | 受证据门控地发布一个新数据版本，旧发布保持只读 | 读存在的 `data/portable_release/` 与 v4 release；**重跑会重写/新增发布，非必要不执行** |
| `validate_code_training_repair_v4.py` | CPU Docker 验证：全源测试、独立 oracle、错误程序 | 可（需 Docker）；会写 `reports/code_training_repair_20261025_v4/`，非发布/验收流程不要跑 |
| `verify_code_repair_source_v4.py` | 仅按 ID 列定位源 parquet 行，不读参考解 | **不能直接重跑**：依赖已删的 `data/screening_sources/` |
| `build_data_human_review_pack.py` | 生成只读、确定性的人工审核包 | **不能直接重跑**：依赖已删的 `data/screening_sources/`、`data/project2_code*/` 与项目一数据 |
| `audit_official_code_benchmarks.py` | 对下载的官方发布做补充去重审计 | **不能直接重跑**：依赖已删的 `data/code_benchmarks/official/` |
| `decontaminate_project2_v2.py` | 冻结阈值去重扫描，不改批准/不删数据 | **不能完整重跑**：依赖已删的 `data/code_benchmarks/official/` 等 |

## 5. 历史 code-v2/v3 数据工程（8，方法记录）

以下脚本均依赖本次重排删除的原始采集（`data/code_v2_sources/`、`data/screening_sources/`、`data/topup_v6_*`、`data/project2_code*/` 等），**不能直接重跑**，只用于理解当时的数据工程方法。

| 文件 | 说明 |
| --- | --- |
| `apply_code_v2_input_budget.py` | OPD 输入预算门控与确定性训练提示回填 |
| `seal_code_v2_pilot.py` | 显式审核与回归检查后封存 pilot |
| `report_code_v2.py` | 从封存 pilot/批量产物生成可读执行记录 |
| `build_code_v3.py` | 版本化移除精确 wrapper，仅从已审 AceCode 池替换 |
| `resume_code_v3_dev.py` | 仅在受限容器测试通过后补完 v3 dev |
| `export_code_function_cache.py` | 从 pinned 全量 parquet 导出可复现的小型 function 行缓存 |
| `prepare_code_v2_stdio_extension.py` | 从未训练过的 v1 提示扩展 dev 候选，不覆盖既有划分 |
| `quarantine_code_literal_conflicts.py` | 有界静态诊断，不执行参考程序 |

## 6. 环境（1）

| 文件 | 说明 |
| --- | --- |
| `colab_bootstrap.sh` | Linux/Colab 候选引导，**未验证**；依赖 `requirements/linux-cuda.txt`。用法见根 [README.md](../README.md) 的训练入口示例 |
