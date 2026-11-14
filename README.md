# 项目二：数学与代码双教师在线策略蒸馏

> 一句话定位：让 1.5B 学生模型在数学与代码两个领域，用自己的在线生成轨迹接受领域专业教师（数学 7B / 代码 7B）的 **token 级反向 KL** 监督，即 On-Policy Distillation（OPD），并比较领域收益、跨域遗忘与教师成本。

**当前状态（2026-11-12，opd-v3 分支）**。完整实验报告与评测矩阵见 [reports/g4_opd_v3_experiments_20261112/README.md](reports/g4_opd_v3_experiments_20261112/README.md)；本地 Kiro IDE 交接说明见 [docs/handover/kiro_ide_onboarding.md](docs/handover/kiro_ide_onboarding.md)。

- **opd-v3（MOPD 式混合批次双教师 OPD）已实现，CPU 契约测试通过（190 passed / 1 skipped）**：单步内混合 math/code，按领域路由常驻教师；批量 rollout；只算回答位置的 logits；exact / top-k / PG 三种损失；分层采样；软/硬运行身份。设计、显存规划（A100 80GB / G4 96GB）与实验矩阵见 [docs/opd_v3_mopd.md](docs/opd_v3_mopd.md)。
- **G4 上已完成 6 轮上卡训练、checkpoint 续训与全套贪心评测（含本地 Docker 代码沙箱打分）**（详见 [reports/g4_opd_v3_experiments_20261112/README.md](reports/g4_opd_v3_experiments_20261112/README.md)）：
  - **D0 基线（`1.5B-Instruct`）**：`math_dev` 58.20%（保守 46.88%）、`MATH-500` 55.00%（保守 44.40%）、`GSM8K` 73.92%（保守 70.13%）、`code_dev` 39.00%（剔除 8 题阻断后 40.62%）、`HumanEval` 51.22% / Plus 46.95%、`MBPP` 64.55% / Plus 55.29%。
  - **SFT 基线（`sft_v3_seqkd_mix`）**：`math_dev` 58.59%（保守 49.22%）、`code_dev` 38.50%，表明离线序列蒸馏对已指令微调学生的增益有限。
  - **D3 通用单教师（`Qwen2.5-7B-Instruct` @ 80，`lr=1e-4`）**：`math_dev` 61.33%、`MATH-500` **58.80%**、`GSM8K` 77.26%、`MBPP` **67.72% / Plus 59.52%**，但 `HumanEval` 仅 54.27% / Plus 50.61%、`code_dev` 40.00%。
  - **D4 专职双教师（`Math-7B` + `Coder-7B`）**：在 `3e-5 @ 80` 和 `1e-4 @ 30` 下同时提升数学（`MATH-500` 58.00% / 56.60%）与代码（`HumanEval+` 54.88% / 56.10%）；但在 `1e-4 @ 80` 下因 `Math-7B` 先验偏远（初始 KL 0.369）出现数学长题截断退化（`MATH-500` 54.80%）；近端教师消融 `d4_prox07_lr1e-4`（$\alpha=0.7$）将 `HumanEval` 推高至 **62.80% / Plus 57.32%**、`code_dev` 推高至 **43.50%**。
  - **D4 先验对齐双教师（`d4_math14b_lr1e-4`：`Qwen2.5-14B-Instruct` + `Qwen2.5-Coder-7B-Instruct`）**：80 步训练以**零失败、零警告**通过 Gate v2（显存峰值 70.39 GiB，中位 96.6 s/步）；在 `@50` 实现 **6 大评测集 100% 全面超越 D0**（`GSM8K` **78.09%** 全项目最高、`MATH-500` 56.80%、`code_dev` 43.00%、`HumanEval` 61.59% / Plus 54.88%、`MBPP` 65.08% / Plus 55.82%），在 `@80` 取得 `math_dev` **60.94%**、`MATH-500` **56.80%**、`GSM8K` **77.26%**、`HumanEval+` **55.49%**，彻底消除后期数学退化。
  - **未运行项说明**：单域教师配置（`d1_math*`、`d2_code*`）及部分附加消融配置（`d4_blocked*`、`d4_pcgrad`、`d4_pg`、`d4_topk`、`d4_warmstart*`、`d4_aligned_lr1e-4`）仅完成配置与 CPU 契约测试，未在 GPU 上执行。
- **当前数据为 v5（2026-11-08 按负责人指示批准）**：数学 2500 train / 256 dev；代码 2488 train（function 1247 / stdin 1241）/ 200 dev。相对 v4 去掉了与 MATH-500 / GSM8K test / MATH test / dev 重合的 133 条数学题、与代码 dev 重合的 12 条代码题，数学按同 level 补齐。见 [reports/project2_release_v5_20261108/README.md](reports/project2_release_v5_20261108/README.md)。v4 逐字节保留，仅作溯源；所有 opd-v3 配置都指向 v5。
- 评测协议 v3：vLLM（可回退 HF）greedy 生成；数学同时报 Math-Verify 与保守判分器（保守判分器对 MATH-500 参考答案自判只认 389/500）；代码仍只在本地 Docker 执行。
- 代码评测在 **Docker 沙箱**中运行，数据集代码不在宿主执行；目标环境无可用 Docker 时停止，不得静默回退宿主子进程。
- 历史 CPU 测试、2 步 smoke 与 tiny 模型恢复测试只证明工程链路，不能替代真实 Qwen 前向、显存、梯度与能力验收。

## R1 / Latent-MOPD 双域训练

这轮探索使用 `DeepSeek-R1-Distill-Qwen-1.5B` 学生、`JustRL-DeepSeek-1.5B` 数学教师和 `Archer2.0-Code-1.5B-Preview` 代码教师。三个版本由 [训练入口](src/training/latent_opd.py) 锁定，权重缓存在 `mydrive:models/`。训练使用 v5 已批准的数学和代码池。D0–D4 的模型、配置和结果沿用上面的 OPD v3 记录。

- [损失实现](src/losses/latent_opd.py) 依据 [Latent-MOPD 附录 A、C](https://arxiv.org/html/2610.02381v1)。每个回答位置选择学生的 top-16 候选，师生打分在有效词表上归一化，优势按整步的领域内奖励标准差缩放。`token` 只使用这套 token 损失，`latent` 加最后三层对齐并按教师计数各淡出 10 步，`math20` 只把数学淡出延长到 20 步。
- 正式配置为单卡 LoRA、数学和代码逐步交替、62 步、每步 32 题 × 4 个回答、温度 1.0、生成上限 16,384 token、AdamW 固定学习率 `1e-5`。论文主实验包含逻辑域并使用 8 个 worker。本实验的并行生成分批和已批准数据属于双域改编。
- WSL2 执行 `~/.venvs/opd/bin/python -m pytest -q`，196 项通过、1 项跳过，包含 top-16 损失与梯度的数值对照。此前 L4 短跑用全词表 reverse KL 和 128 token 上限，验证了三条训练路径及旧实现的 checkpoint 续跑。[L4 原始记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/results/latent_smoke/20261112T065855Z)
- G4 首次 top-16 正式负载尝试于 2026-11-12 15:50（新加坡时间）启动。32 条并发生成期间观测到显存约 93.9 GiB，最近 20 秒平均 GPU 利用率约 97%。复查时会话已结束，Drive 保存的步级日志为空且无 checkpoint，退出原因未确认。正式步耗时、训练能力提升和评测分数仍待验证。[G4 启动与中断证据](https://github.com/XDE10/qwen-dual-teacher-opd/tree/results/latent_top16_token/20261112T075031Z)

2026-11-12 20:57（新加坡时间）的 G4 重试由负责人关闭会话，复查 Drive 时没有可恢复的 checkpoint。[重试中断记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/results/latent_top16_token_retry/20261112T125718Z)

2026-11-12 21:12（新加坡时间）在 G4 会话 `p2-top16-restart` 按相同配置从第 1 步重开纯 token。最后一次同步在 21:33，保存的 `metrics.jsonl` 为空，Drive 中没有 checkpoint，Colab 已无该会话。运行未完成，终止原因未确认。[最后一次同步记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/results/latent_top16_token_retry/20261112T131202Z)

[Colab 启动脚本](scripts/colab_latent_top16.sh) 的新运行目录为 `/content/runs/r1_latent_observable`。`latent smoke-then-full` 先跑 3 步验证数学、代码、混合损失和 checkpoint，验证通过后启动 62 步 Latent。正式配置保留每步 128 条回答与 16K 上限，每次并发 4 条。`phase_metrics.jsonl` 记录各批进度、长度、截断、耗时和 CUDA 显存。压缩的 `rollout_spool` 文件按运行身份和样本验证，恢复后可复用已经落盘的生成批次。首步结束即保存优化器 checkpoint。

```bash
python scripts/colab_job_v3.py latent_mopd_main --drive mydrive:opd_runs --publish --sync-every 2 --save runs=/content/runs/r1_latent_observable -- bash scripts/colab_latent_top16.sh latent smoke-then-full
```

2026-11-12 23:27（新加坡时间）在 G4 会话 `p2-latent-main` 启动 Latent。3 步短跑通过，覆盖纯隐藏层更新和 top-16 加隐藏层的联合更新，第 1、3 步 checkpoint 通过哈希校验并同步到 Drive。随后自动启动 16K 正式配置。11 月 13 日复查时已完成 16/62 步，正式 checkpoint 已保存至第 10 步，正式评测待执行。当前实现的 WSL2 全量测试为 197 项通过、1 项跳过。[本轮运行记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/results/latent_mopd_main/20261112T152744Z)

**2026-11-14 11:38（新加坡时间）快照**：正式训练累计完成 50/62 个 optimizer updates，第 51 步数学已生成 104/128 条。最新发布日志记录了 checkpoint-50 保存与 Drive 同步。训练启动版本为 `d3c0242de47040853a353f5f61b21842eb540624`。R1 双域 dev/test 和匹配预算的 Pure top-16 正式结果待验证。[固定结果快照](https://github.com/XDE10/qwen-dual-teacher-opd/tree/f0e98c6867d2004564983657eb296d957c3aa1aa/latent_mopd_main/20261113T220228Z)

首次从 checkpoint-20 恢复时，第 22 步复用了旧轨迹。当前缓存键没有学生权重指纹，第 22 步的严格 on-policy 性质无法追溯确认。从 checkpoint-40 恢复时，仅尝试复用直接由该 checkpoint 生成的第 41 步轨迹，第 42 步起重新生成。在后续训练中，缓存与生成策略的绑定仍需完善。

代码评测采用版本化的 EvalPlus 资源策略。默认 `vms-headroom-v1` 已在固定 Docker 镜像中验证两道 MBPP 样本，完整数据集重评分和评分入口的完整性验证仍待完成。[验证记录](reports/wsl_evalplus_crash_fix_20261113/README.md)

## 快速开始

所有命令都从项目根目录执行。

**当前开发环境是 WSL2（Ubuntu，zsh）**，仓库在 `~/projects/qwen-dual-teacher-opd`，Python 环境是 `~/.venvs/opd`（依赖 `requirements/wsl2-cpu.txt`）。初始化和验证命令见 [docs/handover/kiro_ide_onboarding.md](docs/handover/kiro_ide_onboarding.md) 第 3 节：

```zsh
~/.venvs/opd/bin/python -m pytest -q
~/.venvs/opd/bin/python tools/verify_repository.py
```

下面是原 Windows 环境的记录：**默认 `python` 是 Python 3.8，不可用**；请使用 `D:/anaconda3/envs/genai-test/python.exe`（Python 3.12），或先 `conda activate genai-test` 再使用 `python`。

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

Linux/Colab（A100 或 G4）：`bash scripts/colab_bootstrap_v3.sh` → `python -m scripts.prefetch_models_v3 --config ...` → `python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_smoke.json` → `python -m scripts.probe_teacher_fit_v3 ...`，完整顺序与异源教师的处理见 [docs/opd_v3_mopd.md](docs/opd_v3_mopd.md)。在 WSL2（zsh）里用 Colab CLI（google-colab-cli 0.7.5）时，开卡、rclone 挂 Drive、克隆、长任务的命令见 [docs/colab_cli_wsl2.md](docs/colab_cli_wsl2.md)；任务结束时结果自动存到 Drive 的 `opd_runs/`，日志和 JSON 推到 `results` 分支。

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
- `scripts/` 下 44 个入口（另有 `scripts/docker/` 下的 EvalPlus 镜像 Dockerfile）按训练/评测/审计/数据工程/环境分组，见 [scripts/README.md](scripts/README.md)；哪些依赖已删除的原始数据、不能直接重跑，也在该文件说明。

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

- `scripts/train_project2_opd_v3.py`：**opd-v3 入口**（`src/training/opd_v3.py`），配置为 `configs/project2_opd_v3_*.json`（smoke、d1_math、d2_code、d3_general、d4_dual、d4_blocked、d4_pcgrad、d4_pg、d4_topk、d4_warmstart，学习率探测用的 d4_dual_lr1e-5、d4_dual_lr1e-4，以及近端教师消融 d4_prox07）。`--stop-at-step N` 在第 N 步存 checkpoint 后停下，`--resume-from latest` 从最新的有效 checkpoint 继续。
- `scripts/train_project2_opd.py`：v2 CLI 入口（内部直接调用 `src/training/opd_v2.py` 的 `cli()`），保留用于溯源。
- `scripts/train_project2_opd_v1.py`：历史 v1 入口，仅追溯，不推荐新实验。
- 全部 44 个 `scripts/` 入口的分组与可重跑性见 [scripts/README.md](scripts/README.md)。
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

## 已验证与未运行事项边界（不要夸大）

- **已在 G4 GPU 与本地 Docker 完成验证**：
  - 2 步 smoke、教师适配度探针（`reports/g4_first_session_20261109/`）；
  - 三个学习率（`1e-5`、`3e-5`、`1e-4`）各 30 步完整批次训练及开发集配对评测（`reports/g4_session2_lr_sweep_20261109/`）；
  - D0 基线、SFT（`sft_v3_seqkd_mix`）、D3（`d3_general_lr1e-4`）、D4 专职双教师（`d4_dual`、`d4_dual_lr1e-4`、`d4_prox07_lr1e-4`）以及 D4 先验对齐双教师（`d4_math14b_lr1e-4`：`14B-Instruct` + `Coder-7B`）的 80 步训练、硬身份 checkpoint 续训、GPU 贪心评测生成，以及本地 Docker 沙箱代码评测（`code_dev` + 官方 EvalPlus v0.3.1 `HumanEval+` / `MBPP+`），详见 [reports/g4_opd_v3_experiments_20261112/README.md](reports/g4_opd_v3_experiments_20261112/README.md)。
- **仅实现配置并通过 CPU 测试、未在 GPU 运行**：
  - 单域单教师配置 `d1_math` / `d1_math_lr1e-4`、`d2_code` / `d2_code_lr1e-4`；
  - 附加消融配置 `d4_blocked` / `d4_blocked_lr1e-4`、`d4_pcgrad`、`d4_pg`、`d4_topk`、`d4_warmstart` / `d4_warmstart_lr1e-4`、`d4_aligned_lr1e-4`。
- 所有阶段都应分别陈述「代码已实现」「CPU 测试通过」「真实模型跑通」「最终能力提升」，未运行的检查保持未验证。

## 给 AI/协作者的入口

先读 [AGENTS.md](AGENTS.md)，再读本 README、`configs/` 与 `data/code_training_repair_20261025_v4/release/manifest.json`；按任务读取 `src/`、`scripts/` 和版本化报告。不需要先遍历历史报告或 provenance 全量文件。文档索引见 [docs/README.md](docs/README.md)。
