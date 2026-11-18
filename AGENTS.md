# AGENTS.md — 项目二：数学与代码双教师在线策略蒸馏

## 当前路线补充（2026-11-18）

当前在跑的是负责人批准的 **R1三教师Latent-MOPD独立复现**；下文Qwen2.5双教师OPD-v3及其模型身份属于历史路线。接手先读 [最新三教师状态](reports/latent_mopd_three_status_20261118/README.md)、[实现与命令](docs/latent_mopd_three.md)、[完整断点与派生续训](docs/latent_mopd_three_continuation.md)，再按需查旧交接。

- 三教师训练池是每域5952条已批准输入，使用冻结的 `release_poolfix_20261115`，不把候选或dev混进训练。固定模型revision和发布哈希见最新状态报告。
- 现有62步任务、部署包和17个身份绑定源码保持固定。仓库训练器须与线上 `c68797a2063ad5df7bf1d9e187ad94c0c37bbbf12894b26a14a518052fa29eb7` 版本一致；不要因为main更新就替换活跃VM代码、改源文件清单或重新安装依赖。
- 延长预算使用独立派生身份，继承优化器和RNG，保留父断点；禁止修改原身份或跳过数据/源码校验。续训与归档入口已实现，80步真实GPU恢复仍未验收。
- checkpoint-28的8K greedy dev、思考内答案单边补回、最终论文采样评测分别报告；最终测试不用于挑步数。代码仍只能在Docker评分。所有自由选择的大文件、缓存与导出优先放D盘。
- 当前正式任务仍只到62；本次代码交付不授权额外启动训练。先确认负责人当前指定的目标与实际运行证据。

适用范围：本仓库根目录及其全部子目录。给后续 AI 协作者和人类协作者。

## 0. 先读什么

1. 本文件。
2. [docs/handover/kiro_ide_onboarding.md](docs/handover/kiro_ide_onboarding.md)：本地 Kiro IDE 交接，包括当前进度、WSL2 初始化、下一次上卡的命令和决策规则。
3. [README.md](README.md)：当前状态、目录树、命令与验证链。
4. 按任务读 [configs/](configs) 与当前数据 manifest
   [data/project2_release_v5_20261108/release/manifest.json](data/project2_release_v5_20261108/release/manifest.json)（v5；v4 只作溯源）。
5. 需要历史背景时再读 [docs/README.md](docs/README.md)。不要先遍历 `provenance/`、历史报告或 `docs/handover/v2/`。

## 1. 项目范围与身份

- 项目二 = **数学与代码双教师 token-level 在线策略蒸馏（OPD）**：学生在线生成，冻结的领域教师对**同一 token 轨迹**给出分布，学生按 response-token reverse KL 更新。不做熵融合，不做多卡。
  - opd-v2（历史实现，保持不变）：教师按领域分时加载，每个 step 只含一个领域。
  - **opd-v3（当前，2026-11-08 设计变更）**：参照 MOPD，每个 step 混合两个领域并按领域路由；80/96 GB 卡上两位教师常驻显存，小显存卡用 `cpu_swap`。设计见 [docs/opd_v3_mopd.md](docs/opd_v3_mopd.md)。
- 固定身份（禁止混用）：
  - 学生 `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  - 数学教师 `Qwen/Qwen2.5-Math-7B-Instruct` @ `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`
  - 代码教师 `Qwen/Qwen2.5-Coder-7B-Instruct` @ `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- 共同 token 支持固定为 **151665 个 ID（0～151664）**，必须从词表审计读取并验证连续性；不得用 `min(vocab_size)` 计算 KL。
- 本项目独立于旧 `modern_genai` 仓库与其他项目目录；旧环境已清理，不能引用其 `.venv` 或绝对路径。

## 2. 硬约束（违反即结论错误）

- **token IDs 只能由学生模板渲染一次**，并原样（完整 `input_ids` / `attention_mask`）交给教师；教师不得重新套自己的 chat template，也不得 decode 后再 tokenize 学生前缀。教师侧不得引入自己的后缀/系统提示。
- **明确区分已跑完与未运行的实验配置**。G4 GPU 与本地 Docker 上已完成 D0、SFT（`sft_v3_seqkd_mix`）、D3（`d3_general_lr1e-4`）、D4 专职双教师（`d4_dual`、`d4_dual_lr1e-4`、`d4_prox07_lr1e-4`）以及 D4 先验对齐双教师（`d4_math14b_lr1e-4`：`Qwen2.5-14B-Instruct` + `Qwen2.5-Coder-7B-Instruct`）的 80 步训练与 6 套评测集（详见 [reports/g4_opd_v3_experiments_20261112/README.md](reports/g4_opd_v3_experiments_20261112/README.md)）；单域对照组（`d1_math*`、`d2_code*`）与附加消融配置（`d4_blocked*`、`d4_pcgrad`、`d4_pg`、`d4_topk`、`d4_warmstart*`、`d4_aligned_lr1e-4`）仅完成 CPU 契约测试，未在 GPU 上运行。
- **数据逐行批准**。范围只认 manifest 的 `approved_training_files` / `approved_evaluation_files` 与逐行 `training_approved`；候选 ≠ 批准，AI 审核 ≠ 人工金标。dev 不得进训练。
- **不得手改批准字段或哈希**来绕过检查。新数据版本必须独立取证、保留旧发布；不能用批量重算 SHA 的方式"修好"检查。
- **代码评测必须走 Docker 沙箱**，数据集代码不在宿主执行；无可用 Docker 时停止并报告，不得回退宿主子进程。Colab 没有 Docker：GPU 端只生成候选 JSONL（`scripts/generate_eval_v3.py`），由本地 Docker 执行。
- **数据批准只能由项目负责人决定**：`scripts/build_project2_release_v5.py` 只生成候选，`scripts/approve_project2_release_v5.py --approver <姓名>` 记录负责人的决定。AI 协作者只有在负责人明确书面指示后才能代为执行，并且必须在 `--note` 里写明指示的原文、时间和"由 AI 代为执行"。不得手改候选中的 `training_approved`。v5 已于 2026-11-08 按负责人指示批准（见 `data/project2_release_v5_20261108/release/manifest.json` 的 `approval`）。
- **数学主口径用 Math-Verify，同时报保守判分器**：保守判分器对 MATH-500 参考答案自判只认 389/500，不能单独作为主分数。
- 新增/修改 opd-v3 配置时，必须保持 `tests/test_opd_v3.py::test_experiment_configs_validate_and_differ_only_in_declared_factors` 通过：对照组之间只允许出现声明过的差异。
- **续训依赖运行身份**：`src/training/runtime_v3.py` 的 `SOURCES` 列出的源文件、run 的配置文件、`output_dir`（因此 `OPD_OUTPUT_ROOT` 固定 `/content/runs`）和 torch/transformers/peft 版本，任何一项变了，已有 checkpoint 就不能续训。第二次上卡的 checkpoint 写于 `bc02083`；在要续跑的 run 跑完之前不要改这些文件，必须改时先说明会作废哪些 checkpoint。
- **只用 dev 做选择**：挑学习率、checkpoint、配置只看 math_dev / code_dev；MATH-500、GSM8K test、HumanEval+、MBPP+ 只用于最终报告。在线准确率是训练题上 T=1 的遥测，不能当作评测结果。
- 代码能力看实际源码，状态看 README 与版本化报告；报告必须区分「已实现」「CPU 通过」「真实模型跑通」「能力提升」，未运行的检查保持未验证。

## 3. 目录职责

| 目录 | 职责 |
| --- | --- |
| `configs/` | v1 基线、v4 smoke/math_short/code_short、提示词契约，以及 13 个 opd-v3 配置（`project2_opd_v3_*.json`，含 2 个学习率探测配置和近端教师消融 `d4_prox07`）和 1 个 sft-v3 配置；命令从项目根执行 |
| `src/data/` | 批准输入加载、哈希/白名单、代码沙箱、数据工程实现 |
| `src/teachers/` | 学生提示渲染与"原样 token IDs 直传教师"契约、词表审计 |
| `src/losses/` | 共同支持上的 reverse KL 数值参考与分块实现 |
| `src/training/` | OPD v2 训练循环、采样、运行时/checkpoint、v1 历史实现 |
| `src/evaluation/` | 数学验证器、代码比较器/注册表、EvalPlus 接入 |
| `scripts/` | 训练/评测/数据工程入口；Colab 用 `colab_bootstrap_v3.sh`、`colab_first_session_v3.sh`、`colab_session_v3.sh`、`colab_job_v3.py`、`publish_results_v3.py`；本地分析与打分用 `compare_runs_v3.py`、`summarize_run_v3.py`、`score_code_local_v3.py`（GPU 上未验证；Colab CLI 用法见 [docs/colab_cli_wsl2.md](docs/colab_cli_wsl2.md)） |
| `tests/` | 项目二测试（23 个文件）；`python -m pytest -q` |
| `tools/` | `verify_code_repair_v4.py`（v4 发布离线验收）、`verify_repository.py`（仓库级布局/映射/镜像校验） |
| `data/` | 当前 v5 发布、v4 发布（溯源）、历史 `portable_release`、tokenizer 快照、数学源池/评测集；说明见 [data/README.md](data/README.md) |
| `reports/` | v4 验收证据、v3 审核记录、当前审核 json/md；说明见 [reports/README.md](reports/README.md) |
| `provenance/` | 冻结清单 sha256 锚定的历史证据**镜像**（内容不变），不参与运行时读取 |
| `docs/` | 历史方案/交接/迁移记录；索引见 [docs/README.md](docs/README.md) |
| `requirements/` | `wsl2-cpu.txt`（WSL2 测试环境，已实测）、`linux-cuda-v3.txt` / `linux-cuda-v3-eval.txt`（Colab GPU）、`cpu.txt`（原 Windows 环境记录）、`linux-cuda.txt`（v2 候选，历史） |

## 4. 删除/重构后如何查历史

重排（2026-11-08）删除了项目一/三代码与测试、原始采集数据、历史 reports、根目录学习材料等。查历史按以下顺序：

1. [docs/migration/reorg_20261108.md](docs/migration/reorg_20261108.md)：这次重排删了什么、移了什么、已知缺口。
2. [docs/migration/path_mapping.json](docs/migration/path_mapping.json)：旧路径 → 新位置 + 期望/实际 sha256（`mirrored` / `moved` / `kept` / `missing_before_reorg`）。
3. [provenance/README.md](provenance/README.md)：被冻结发布清单锚定、必须字节可校验的证据镜像机制。
4. [docs/migration/deletion_ledger.json](docs/migration/deletion_ledger.json)：删除台账（130 个顶层项 / 4225 个文件 / 4,380,095,420 字节）；统计删除文件数与字节数请引用它，不要凭记忆编造。
5. [docs/migration/inventory_before_reorg.json](docs/migration/inventory_before_reorg.json)：重排前文件快照（size + sha256），用于核对来源。
6. [docs/plan/](docs/plan/README.md) 与 [docs/handover/v2/](docs/handover/v2/README.md)：冻结的历史方案与 v2 交接；内部链接指向重排前路径，可能失效。
7. `docs/migration/legacy_tools/`：已退役的 `verify_migration.py` 与 `verify_handover_v2.py`，仅历史参考，**不是**当前入口。

注意：`data/project3_audit/geometry3k_intergps_links.jsonl` 在**重排前**即缺失，映射表标记 `missing_before_reorg`；不要为它伪造替代文件。原 `modern_genai` 仓库已清理，恢复已删文件只能依赖其他现存副本 + `inventory_before_reorg.json` 的 sha256 核对。

## 5. 验证命令

当前环境是 WSL2（venv `~/.venvs/opd`，依赖 `requirements/wsl2-cpu.txt`，见交接文档第 3 节）：

```zsh
~/.venvs/opd/bin/python -m pytest -q
~/.venvs/opd/bin/python tools/verify_repository.py
~/.venvs/opd/bin/python tools/verify_project2_release_v5.py
```

原 Windows 环境的命令：

```bash
# 全部项目二测试
D:/anaconda3/envs/genai-test/python.exe -m pytest -q

# v4 发布离线验收
D:/anaconda3/envs/genai-test/python.exe tools/verify_code_repair_v4.py

# 仓库级校验：目标布局、path_mapping、provenance 镜像 sha256（新增）
D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py

# v5 数据候选 / 批准后的 release 校验
D:/anaconda3/envs/genai-test/python.exe tools/verify_project2_release_v5.py

# 语法检查
D:/anaconda3/envs/genai-test/python.exe -m compileall -q src scripts tests tools
```

- 用 `D:/anaconda3/envs/genai-test/python.exe`（Python 3.12）或 `conda activate genai-test` 后 `python`；默认 `python` 是 3.8，不可用。
- 命令从项目根目录执行；`scripts/colab_bootstrap.sh` 仅用于 Linux/Colab 候选环境，未验证。
- 校验通过≠GPU 验收。任何"通过"结论都要注明实际运行了哪条命令、在什么环境。

## 6. 文档/代码变更规则

- 保留来源与验证记录；固定模型、数据、划分与评测身份。
- 数据、manifest、批准范围、既有 SHA 默认不可改；如需合法修订，独立记录新版本与差异，不刷新旧证据。
- 修改文档时保留历史说明；指向历史文档的链接要注明其路径是重排前路径。
- 新报告不要覆盖旧报告；用版本化目录/文件，并说明未运行的检查。

## 7. Colab 运行与结果回传

- 负责人在 WSL2（Ubuntu，zsh）里用 google-colab-cli 0.7.5 操作 Colab，项目在 `~/projects/qwen-dual-teacher-opd`。给本机的命令必须能在 zsh 里直接粘贴：不写行尾 `#` 注释，读密码用 `printf ... && read -rs 变量`（不用 `read -p`），带 `*?[]` 的参数加引号。`colab console` 里是 VM 上的 bash。步骤见 [docs/colab_cli_wsl2.md](docs/colab_cli_wsl2.md)。
- VM 上的运行输出放 `/content/runs`；Drive 用 rclone 远端 `mydrive:`。长任务用 `scripts/colab_job_v3.py` 启动，结束时复制到 `mydrive:opd_runs/<任务名>/<UTC 时间>/`，并用 `scripts/publish_results_v3.py` 把日志和 JSON 推到 `results` 分支的同名目录。
- 负责人不能往 Kiro Web 上传文件。读运行结果先看 `results` 分支（`git fetch origin results`），不要让负责人下载后再上传。
- `results` 分支只放结果，不合并进任何代码分支；不往里放权重、checkpoint 或凭据。
- 代码题的候选在 GPU 上生成，在本地 WSL2 的 Docker 里用 `scripts/score_code_local_v3.py` 打分，分数推到 `results` 分支的 `code_scores/`。内部 dev 分数与官方 EvalPlus 分数分开报告。
- 本地 Kiro IDE 的默认终端设为 WSL，agent 的命令也跑在 zsh 里，同样遵守上面的 zsh 规则。复杂逻辑写成脚本，用 `bash` 或 `python` 调用，不在命令里嵌套多层引号；每次 `colab exec` 都加 `--timeout 86400`。
- Colab 引导脚本会卸载预装的 torchvision、torchaudio、torchao（与 torch 2.8.0 不兼容）。不要把它们加回任何依赖文件，评测走 HF，不设 `INSTALL_VLLM`。
