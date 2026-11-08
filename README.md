# 项目二：数学与代码双教师在线策略蒸馏

> 一句话定位：让 1.5B 学生模型在数学与代码两个领域，用自己的在线生成轨迹接受领域专业教师（数学 7B / 代码 7B）的 **token 级反向 KL** 监督，即 On-Policy Distillation（OPD），并比较领域收益、跨域遗忘与教师成本。

**当前状态（2026-11-08，opd-v3 分支）**

- **opd-v3（MOPD 式混合批次双教师 OPD）已实现，CPU 契约测试通过**：单步内混合 math/code，按领域路由常驻教师；批量 rollout；只算回答位置的 logits；exact / top-k / PG 三种损失；分层采样；软/硬运行身份。设计、显存规划（A100 80GB / G4 96GB）与实验矩阵见 [docs/opd_v3_mopd.md](docs/opd_v3_mopd.md)。真实 Qwen 在 GPU 上**尚未运行**；**D0～D4 均未完成**，不得写成已有正式训练结果。
- **当前数据为 v5（2026-11-08 按负责人指示批准）**：数学 2500 train / 256 dev；代码 2488 train（function 1247 / stdin 1241）/ 200 dev。相对 v4 去掉了与 MATH-500 / GSM8K test / MATH test / dev 重合的 133 条数学题、与代码 dev 重合的 12 条代码题，数学按同 level 补齐。见 [reports/project2_release_v5_20261108/README.md](reports/project2_release_v5_20261108/README.md)。v4 逐字节保留，仅作溯源；所有 opd-v3 配置都指向 v5。
- 评测协议 v3：vLLM（可回退 HF）greedy 生成；数学同时报 Math-Verify 与保守判分器（保守判分器对 MATH-500 参考答案自判只认 389/500）；代码仍只在本地 Docker 执行。
- 代码评测在 **Docker 沙箱**中运行，数据集代码不在宿主执行；目标环境无可用 Docker 时停止，不得静默回退宿主子进程。
- 历史 CPU 测试、2 步 smoke 与 tiny 模型恢复测试只证明工程链路，不能替代真实 Qwen 前向、显存、梯度与能力验收。

## 快速开始

所有命令都从项目根目录执行。**默认 `python` 是 Python 3.8，不可用**；请使用 `D:/anaconda3/envs/genai-test/python.exe`（Python 3.12），或先 `conda activate genai-test` 再使用 `python`。

```bash
cd "D:/Users/Mr.Z/Desktop/项目二_双教师在线蒸馏"
D:/anaconda3/envs/genai-test/python.exe --version

# 全部项目二测试（23 个测试文件）
D:/anaconda3/envs/genai-test/python.exe -m pytest -q

# v4 发布离线验收（stdlib，无需模型/网络）
D:/anaconda3/envs/genai-test/python.exe tools/verify_code_repair_v4.py

# 仓库级校验：目标布局、path_mapping、provenance 镜像 sha256
D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py

# 语法编译检查
D:/anaconda3/envs/genai-test/python.exe -m compileall -q src scripts tests tools

# opd-v3 / v5 数据 / 评测 v3
python -m scripts.restore_official_benchmarks          # 恢复 EvalPlus 官方文件（sha256 校验）
python tools/verify_project2_release_v5.py              # v5 候选（批准后也检查 release/）
python -m scripts.generate_eval_v3 --self-check --suites math500 math_dev gsm8k_test   # 判分器覆盖率
```

Linux/Colab（A100 或 G4）：`bash scripts/colab_bootstrap_v3.sh` → `python -m scripts.prefetch_models_v3 --config ...` → `python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_smoke.json` → `python -m scripts.probe_teacher_fit_v3 ...`，完整顺序与异源教师的处理见 [docs/opd_v3_mopd.md](docs/opd_v3_mopd.md)。

依赖记录在 `requirements/cpu.txt`（本机 CPU 实测版本，**不是** CUDA 锁定）和 `requirements/linux-cuda.txt`（Linux/Colab 候选，**未验证**）。

## 目录结构（重排后实际布局）

```
README.md  AGENTS.md  .gitignore  pytest.ini
configs/            # 5 个 json：project2_opd_smoke.json（v1 基线）、
                    #   project2_opd_v4_{smoke,math_short,code_short}.json（当前）、
                    #   project2_prompt_contract.json（从 project_blueprints/configs/ 移入）
src/                # data/ evaluation/ losses/ teachers/ training/，只保留项目二代码
scripts/            # 项目二训练/评测/数据工程入口 + colab_bootstrap.sh（分类见 scripts/README.md）
tests/              # 项目二测试（23 个文件）
tools/              # verify_code_repair_v4.py + verify_repository.py
data/               # portable_release/、code_training_repair_20261025_v4/、
                    #   tokenizers/、math/、math_manifest.json
reports/            # code_training_repair_20261025_v4/、handover_20261024_v3/
                    #   + 根目录当前审核 json/md
provenance/         # 57 个被冻结清单 sha256 锚定的历史证据镜像 + README.md
docs/               # plan/（3 个项目二历史方案/契约/共用规范）
                    # handover/v2/（v2 交接记录）
                    # migration/（迁移清单、path_mapping.json、
                    #   inventory_before_reorg.json、legacy_tools/、重排说明）
                    # README.md（docs 索引）
requirements/       # cpu.txt、linux-cuda.txt
```

- `docs/migration/legacy_tools/` 放已退役的 `verify_migration.py` 与 `verify_handover_v2.py`，仅作历史工具保留，**不是**当前入口。
- `docs/migration/path_mapping.json` 记录被移动/镜像证据的旧路径→新位置与 sha256；`provenance/README.md` 解释镜像机制。
- `scripts/` 下 39 个入口按训练/评测/审计/数据工程/环境分组，见 [scripts/README.md](scripts/README.md)；哪些依赖已删除的原始数据、不能直接重跑，也在该文件说明。

## 数据与模型身份

固定身份如下，训练/评测/报告都不得混用：

| 角色 | 模型 | revision |
| --- | --- | --- |
| 学生 | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 数学教师 | `Qwen/Qwen2.5-Math-7B-Instruct` | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |
| 代码教师 | `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |

- 共同 token 支持为 **151665 个 ID（0～151664）**；必须从词表审计读取上界并验证连续性，不能用 `min(vocab_size)` 直接算 KL，也不能把支持集外的概率质量当作不存在。
- 词表/模板身份证据：`reports/tokenizer_alignment_report.json`、`configs/project2_prompt_contract.json`、`reports/project2_prompt_contract_audit.json`；本地 tokenizer 快照在 `data/tokenizers/`。

## 数据与批准边界

当前执行配置绑定 **v4 修复发布** `data/code_training_repair_20261025_v4/release/manifest.json`；`data/portable_release/manifest.json` 是历史迁移基线，仍逐字节保留。

- 批准训练：`approved_training_files` = 数学 train、代码 train。
- 批准评测：`approved_evaluation_files` = 数学 dev、代码 dev；dev 不得用于训练。
- 保护 ID：`reserved_dev_ids_never_train.json`。
- v4 相对基线只替换代码训练文件中的 1 行（移除与 dev 同题的 `code/vfc_121257`，加入 `code/vfc_96076`），其余 2499 行、数学 train/dev、代码 dev 与保护 ID 均逐字节不变；旧发布 manifest/批准/SHA 不修改。

准入要求：

- 数据范围以 **manifest 的 `approved_training_files` / `approved_evaluation_files`** 和**逐行批准**为准；候选、隔离、AI 审核通过都不自动等于批准。
- 所有批准行仍要经过文件 SHA、逐行 `training_approved`、唯一 ID、模型输入白名单与 dev 隔离检查。
- **不得手改批准字段或哈希**来绕过检查；新数据版本必须独立取证并保留旧发布。

## 训练 / 评测 / 数据工程入口

训练（未来命令，当前未在真实 GPU 执行）：

```bash
export OPD_OUTPUT_ROOT=/content/drive/MyDrive/project2/runs
PROJECT_ROOT=/content/project2 bash scripts/colab_bootstrap.sh

D:/anaconda3/envs/genai-test/python.exe -m scripts.train_project2_opd \
  --config configs/project2_opd_v4_smoke.json --output-root "$OPD_OUTPUT_ROOT"
```

- `scripts/train_project2_opd_v3.py`：**opd-v3 入口**（`src/training/opd_v3.py`），配置为 `configs/project2_opd_v3_*.json`（smoke、d1_math、d2_code、d3_general、d4_dual、d4_blocked、d4_pcgrad、d4_pg、d4_topk）。
- `scripts/train_project2_opd.py`：v2 CLI 入口（内部直接调用 `src/training/opd_v2.py` 的 `cli()`），保留用于溯源。
- `scripts/train_project2_opd_v1.py`：历史 v1 入口，仅追溯，不推荐新实验。
- 全部 39 个 `scripts/` 入口的分组与可重跑性见 [scripts/README.md](scripts/README.md)。
- `configs/project2_opd_v4_smoke.json` 为 3 步 math/code/math；`*_v4_math_short.json` / `*_v4_code_short.json` 是单域短程配置；`project2_opd_smoke.json` 是 v1 基线原样保留。
- 数据流：学生模板渲染一次提示 → 学生生成轨迹 → **同一 `input_ids` / `attention_mask` 原样交给冻结教师** → 共同支持上 FP32 重归一化后计算 response-token reverse KL → 更新学生。教师不得重新套自己的聊天模板或 decode 后重新 tokenize。

评测：

- `scripts/evaluate_approved_math.py`：固定 greedy 数学评测。
- `scripts/evaluate_code_v2.py`、`scripts/evaluate_evalplus_v2.py`、`scripts/code_eval_worker_v3.py`、`scripts/code_sandbox_worker.py`：代码评测与 Docker 沙箱 worker。
- 最终评测计划为数学 MATH-500 / GSM8K、代码锁定版本 EvalPlus 的 HumanEval+/MBPP+；内部 dev 与官方最终测试分开。当前只接通了部分入口，不能称完整混合域评测已完成。

数据工程（v4）：

- `scripts/prepare_code_training_repair_v4.py`、`scripts/export_code_training_repair_v4.py`、`scripts/validate_code_training_repair_v4.py`、`scripts/verify_code_repair_source_v4.py`（参数见各脚本 `--help`）。
- `scripts/audit_project2_prompt_contract.py`：提示词契约审计。

## 验证链

```bash
D:/anaconda3/envs/genai-test/python.exe -m pytest -q                    # 当前 189 passed / 1 skipped（含 opd-v3、v5 数据、评测 v3、预热与探针）
D:/anaconda3/envs/genai-test/python.exe tools/verify_code_repair_v4.py   # v4 发布身份/逐行批准/接口配额
D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py       # 新布局 + provenance 镜像/映射
D:/anaconda3/envs/genai-test/python.exe -m compileall -q src scripts tests tools
```

opd-v3 分支实测（2026-11-08，Linux / Python 3.11 / torch 2.15 CPU / transformers 4.57.6 / peft 0.19.0）：`pytest -q` = **189 passed / 1 skipped**；重排时的基线是 144 passed / 1 skipped（18.7s）。唯一 skip 是 `tests/test_code_v2_dedup.py::test_current_final_tests_and_all_legacy_dev_are_protected`：它依赖的原始 benchmark 源文件（`data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl`、`MbppPlus-v0.2.0.jsonl`、`data/project2_code/dev_verification.jsonl`）已随重排删除，测试内以 `pytest.mark.skipif` 显式跳过并注明原因，不是失败。

- `verify_code_repair_v4.py` 校验 v4 manifest 文件 SHA、输入证据 SHA、受保护文件、批准行与接口配额，并写出 `reports/code_training_repair_20261025_v4/verification_summary.json`；当前 `errors=[]`，60 条受保护文件中 8 条原位、52 条经 `path_mapping.json` 回退到 `provenance/`。
- `verify_repository.py` 校验重排后台账、`docs/migration/path_mapping.json` 与 `provenance/` 镜像的可达性和 sha256，并报告重排前即缺失的条目；当前 `layout_ok=true`、70 条映射 sha 全匹配。
- 校验通过只说明**数据/文件身份与目录结构**成立，不证明 CUDA 训练、教师真实前向或能力提升。

## provenance、docs 与历史查询

- `provenance/`：冻结发布清单仍要求校验、但旧路径已在重排中删除的证据文件，按原字节镜像保存（sha256 不变）。映射见 `docs/migration/path_mapping.json`；机制见 `provenance/README.md`。
- `docs/plan/`：3 篇 2026-10 冻结的历史方案/契约/共用规范，内部链接指向重排前路径，可能失效。
- `docs/handover/v2/`：v2 本地修订与 Colab 交接记录（历史交付，不代表 GPU 验收）。
- `docs/migration/`：重排说明 `reorg_20261108.md`、迁移清单、`path_mapping.json`、重排前快照 `inventory_before_reorg.json`、退役工具与删除台账 `deletion_ledger.json`（130 个顶层项 / 4225 个文件 / 4,380,095,420 字节）。查旧文件/旧目录先查这里。
- 原 `modern_genai` 仓库及其 `.venv` 已清理，不能引用其路径；本项目目录可独立迁移。

## 未完成事项（不要夸大）

- 真实学生/教师 GPU 前向、显存峰值、吞吐、词表外概率质量与 checkpoint 恢复**均未验收**。opd-v3 的这些项目只在 tiny 模型上通过了 CPU 测试。
- D0（原始学生）、D1/D2（数学/代码单教师）、D3（通用单教师）、D4（专业双教师路由）的正式配置与结果都还没有。
- 代码任务的多解题语义 checker、官方 EvalPlus 完整接线、各领域准确率与跨域退化统计仍需补齐。
- 所有阶段都应分别陈述「代码已实现」「CPU 测试通过」「真实模型跑通」「最终能力提升」，未运行的检查保持未验证。

## 给 AI/协作者的入口

先读 [AGENTS.md](AGENTS.md)，再读本 README、`configs/` 与 `data/code_training_repair_20261025_v4/release/manifest.json`；按任务读取 `src/`、`scripts/` 和版本化报告。不需要先遍历历史报告或 provenance 全量文件。文档索引见 [docs/README.md](docs/README.md)。
