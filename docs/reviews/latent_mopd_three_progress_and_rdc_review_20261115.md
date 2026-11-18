# 项目二三教师 Latent-MOPD 进展与 6Pro 审查

2026-11-15，新加坡时间。Colab 与归档状态核对截至 17:40 左右。

本项目以 Latent-MOPD Table 1 的同家族三教师 last-3 为目标，使用公开论文、公开固定模型和公开题源做独立复现。已在 G4 完成六步短跑、两个正式规模优化器更新，以及一次 16K 固定轨迹联合损失探针。现有会话已不在活动列表，Drive 上能定位到原任务的 `checkpoint-1`。新版数据已发布到本地，云端曾报告从该检查点完成迁移准备。修复版续训、三教师正式 62 步和 Base／Token-only／Latent 配对评测均无完成证据。

## 给 6Pro 的请求与 RDC 入口

请先使用 RDC（[$remote-desktop-commander](app://asdk_app_6a057d268ebc81919918d37eec718425)） 连接我的本地机器，用 `list_directory`、`read_file` 或 `read_multiple_files` 阅读实际项目文件，再开始判断。需要跑只读核对命令时，用 RDC 的 `start_process` 进入 WSL。项目在 Linux 文件系统，请查看实际工作区、未提交的三教师实现和被 Git 忽略的 `runs/`。仅查看 GitHub main 会遗漏本次实现和发布数据。

- WSL 路径为 `/home/xde10/projects/qwen-dual-teacher-opd`。
- Windows 的 UNC 路径为 `\\wsl.localhost\Ubuntu\home\xde10\projects\qwen-dual-teacher-opd`。如果 RDC 的文件读取无法打开 UNC，可用其进程工具执行下方只读命令。
- 请从 `AGENTS.md`、当前报告和下方列出的训练、数据、迁移、评测入口开始读。任何路径打不开，请明确列出路径和错误，以及因此无法验证的判断。
- 审查授权限于读文件、读取归档和低成本本地检查。训练切换由项目负责人指定的侧边栏处理。审查时请保持运行状态、发布数据、Drive 和 Git 分支原样。凭据不属于审查材料。仓库规定代码题的候选程序只能在 Docker 沙箱里执行。

```powershell
wsl.exe -d Ubuntu -e bash -lc 'cd /home/xde10/projects/qwen-dual-teacher-opd && git status --short --branch && git rev-parse HEAD && sed -n "1,145p" AGENTS.md'
wsl.exe -d Ubuntu -e bash -lc 'cd /home/xde10/projects/qwen-dual-teacher-opd && sha256sum runs/latent_mopd_three/{release_codefix_20261115,release_poolfix_20261115}/manifest.json'
```

请按优先级给出具体判断，尤其回答三个问题。**以当前实测吞吐是否值得继续投入，哪条恢复路径能保留已有工作，以及在什么证据出现之前不能声称复现出了能力提升。** 对每个问题请引用实际文件和函数、数据行、日志时间或论文章节。发现证据冲突时，把两个版本都列出来，并说明采用哪一份及理由。

## 实验目标与历史结果

代码基线是 `main@4dee8bfbc554a22461d4d347f4973fc7fb7a016e`。其 README 记录上一轮双教师实验。当前三教师代码和数据还在本地工作区，首次 G4 用固定基线加覆盖包部署。GitHub `results` 是结果归档分支。工作区另有历史审查稿，文档里的“未批准”“未上卡”属于此前的状态。

上一轮是数学和代码双教师、LoRA、各更新 31 次的单卡探索。相同的 v5 dev、HF greedy、8,192 新 token 和同一判分条件下，数学 Math-Verify 从 **170/256（66.41%）降到 162/256（63.28%）**，代码从 **70/200（35.00%）升到 78/200（39.00%）**。排除两边同样的 8 道阻断题，代码从 36.46% 升到 40.63%。这组结果缺匹配预算的 Token-only 对照，也缺逻辑教师。 [历史双教师逐题归档](https://github.com/XDE10/qwen-dual-teacher-opd/tree/36312e2212cc66d7cf76be1887aaa686340a6f9a/code_scores/r1_dev62-20261114T104404Z)

本轮的对照应当是同一 Base R1 初始化下的 Base、三教师 Token-only 和三教师 Latent。原论文 Table 1 为三教师组合提供依据，原作者逐题训练清单与完整实现此前未取得。Replay 论文是独立交叉参考，其本地联合实验九项均分为 **37.57／52.01／52.31**。在该协议中，Latent 相对 Base 为 +14.74 pp，相对 Token-only 为 +0.30 pp，部分评测模板由其重建。 [Latent-MOPD Table 1 和附录 A/C](https://arxiv.org/html/2610.02381v1) · [Replay Table 8 和附录 C](https://arxiv.org/html/2611.02746v2)

## 固定配方与当前实现

| 角色 | 模型 | 固定 revision |
| --- | --- | --- |
| 学生 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| 数学 | `hbx/JustRL-DeepSeek-1.5B` | `0637e4096c789c67f9eecbe8355e0bdeddede1c2` |
| 代码 | `Fate-Zero/Archer2.0-Code-1.5B-Preview` | `db1e1ff9f3c36f7177f3896358ce26edeb62ce56` |
| 逻辑 | `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` ProRL v2 | `a8e647c7a46f04e84ce732d6fa3fa627ba343b8d` |

- 62 次优化器更新，数学、代码、逻辑按纯域块轮转，更新数为 21／21／20。每步 32 道题、每题采样 4 条，正式共 1,984 道题和 7,936 条回答。温度 1.0，回答上限 16,384 token。当前实现提示词上限 2,048 token。
- 单 G4、HF 自回归，每批同时生成 4 条。学生全参数 FP32 AdamW，BF16 autocast，学习率 1e-5，三位教师冻住并常驻。由学生模板生成一次 token ID，师生对同一轨迹计算 token 和 hidden-state 项。当前实现限制采样在审计通过的连续 151,665 个 ID，候选 log-prob 的分母包含完整 LM head。
- Token 项用学生 top-16、候选归一化和整步奖励标准差。Latent 项用最后三层，教师各自按前 10 次更新 crossfade。全局第 1–3 步是纯 Latent，第 4–30 步是联合，第 31–62 步是纯 token。训练不使用正确答案验收奖励。程序按 8 组逻辑 worker、每组 16 条回答和 24,576 token 的 packing 预算在一张卡上执行。
- 固定 R1 tokenizer 仍渲染末尾的 `<think>\n`。原文的 `enable_thinking=False`、本地模板实际输出和日后评测的 `</think>` 提取需一起审查。四个模型缓存已由 CPU 路径放到 Drive，通过 MD5、字节数及固定模型源校验。参考 `runs/latent_mopd_three/g4/drive_upload_verification.json`。

## GPU 已验证的范围与耗时

Colab 原会话为 `p2-latent-three-g4`，原任务为 `latent-three-codefix-62`，启动归档时间为 `20261115T054641Z`。G4 实际是 RTX PRO 6000 Blackwell Server Edition，97,887 MiB，Torch 2.8.0+cu128、CUDA 12.8、sm_120。BF16 matmul 通过。 [归档固定提交](https://github.com/XDE10/qwen-dual-teacher-opd/tree/a8bf2517d5fe6d3274ba05df013e295e27d9da0f/latent-three-codefix-62/20261115T054641Z)

| 已运行项 | 实际结果 | 证据范围 |
| --- | --- | --- |
| 6 步短跑 | 1 题 × 4 答／步，回答上限 1,024。三位教师各完成一轮纯 Latent、一轮联合更新，损失与梯度有限 | 覆盖 CUDA 更新和短序列联合路径 |
| 正式第 1 步数学 | 128 条，1,297,513 个回答 token，20 条达到上限。105.19 分钟，峰值 CUDA reserved 42.88 GiB。15:37:13 更新完成，15:38:11 封存 `checkpoint-1` | 正式 32×4 采样和纯 Latent 参数更新已通过 |
| 固定轨迹 16K 联合探针 | 用第 1 步 checkpoint，逻辑前缀 703 token，固定响应恰为 16,384 token，权重 0.1 token／0.9 latent。有限损失 −0.07248、梯度范数 6.70158、峰值 reserved 41.25 GiB，计时段约 3.18 秒，优化器更新为 0 | 覆盖单条长联合微批的打分和反传 |
| 正式第 2 步代码 | 128 条，1,339,641 个回答 token，12 条达到上限。103.11 分钟，峰值 reserved 48.66 GiB，17:22:28 完成更新 | 有步级结果。按原计划第 2 步未单独封存 |
| 正式第 3 步逻辑 | 最后一份归档有 `rollout_start` | 尚无该步优化器更新记录 |

两个正式步的 rollout 分别耗时 102.16 和 100.32 分钟，约占步耗时 **97%**。若仅把这两步的均值外推到 62 步，约 **107.6 GPU 小时**，还要另计断点、同步和域间长度差异。这是成本量级示例，实际总耗时和 CU 尚不能确定。请优先审计 HF 逐 token decode、FP32 学生在 `autocast(cache_enabled=False)` 下的重复类型转换、KV cache 及解码并发。改后端、精度、批大小或采样前，先给出对固定策略和同协议对照影响的验证方法。 [正式 phase metrics](https://github.com/XDE10/qwen-dual-teacher-opd/blob/a8bf2517d5fe6d3274ba05df013e295e27d9da0f/latent-three-codefix-62/20261115T054641Z/full/phase_metrics.jsonl)

## 数据版本与全池复核

三域题源是固定版本 DAPO-Math-17k、OpenCodeReasoning、Reasoning Gym 0.1.19。每域候选 5,952 道，正式 62 步用数学 672、代码 672、逻辑 640。逻辑正式排程覆盖 37 类，参考解 640 道通过对应的验证器。数学 672 道原始排程均定位到源参考整数答案。评测保护输入含冻结 v5 math/code dev、逻辑 dev/test 和若干公开 benchmark 题面。去重和隔离使用 ID、规范化文本、3-gram Jaccard、13-gram/LCS 和短串包含规则，覆盖范围由清单记录。

下表各目录都位于 `runs/latent_mopd_three/`，哈希是各自 `manifest.json` 的 SHA-256。

| 发布目录 | 变更及用途 | SHA-256 |
| --- | --- | --- |
| `release` | 原批准版本，保留作溯源 | `44928d79025aecf4d909d6fabe426982573cc642754803bfb098c20501b31ce9` |
| `release_codefix_20261115` | 正式 G4 旧任务读取的版本。正式排程中 4 道交互题和 1 道需外部文件的题等量换掉 | `071898e06aa54ad0471f3c949628d36d92f735e929be26c859f3869ff38c0499` |
| `release_statementfix_20261115` | 修复正文缺失的 9 道正式代码题。全部来自 Aizu，旧题只剩输入输出样例 | `1c113b2c4640091123eb47dc100e89d151ca9d27d58da8a1b7ed83347eec3cb0` |
| `release_poolfix_20261115` | 当前最新本地版。在上一版基础上再替换 119 道，其中代码 90 道含原来剩余的 55 道样例残片，数学 29 道缺必要图示或条件。逻辑不变 | `e5f4c579f8a4514795e6ba80e1e4c2df034e724346bbfce7a2ea150d7d969252` |

全池复核采用 17,856 道结构扫描及命中项人工复核。新题来自固定源文件，代码保持平台和原始难度标签，数学替代优先选有整数参考答案的几何题。全部重新验证 tokenizer 前缀，冻结评测文件的哈希一致。最后一版相对上一版影响 5 个训练槽位。**相对原 G4 启动版共改变 14 个槽位，并且第 1–13 步的样本和实际 token IDs 完全一致。第 14 步才有第一处不同。** 请从正式加载入口验证，程序根据各 `*_train.jsonl` 行序组装排程，`schedule.jsonl` 用来封存和核对。

现有质量校验覆盖题干、题源、长度、批准和重合。代码源难度标签混用不同标度，换题保留标签并不提供师生难度校准。训练前没有在同一批本地训练题上对 Base R1 与对应教师做分层正确率测量，代码训练题这一份输入也缺带出处的可执行测试。Base R1 四次答错本身不构成坏题判据。请说明是否值得用小样本量化教师优势和难度覆盖，以及如何保证代码测试的来源和沙箱口径。

## 检查点与切换记录

- 原任务默认保存第 1 步、第 10 步和之后每 10 步，以及最后一步。归档证明第 2 步更新已完成，但目前 Drive 原任务目录只找到 `checkpoint-1`。检查点清单 SHA-256 为 `10f3be926c88e5aadcb351ce1017ab9e6b9067278438c0d0050ffca486701b51`，与 16K 探针记录吻合。Drive 可列出 `engine.pt`、两份模型分片和配置，整体约 18.12 GiB。当前审查前置核对读回了清单及文件列表，完整对象的重新哈希由实际恢复入口完成。
- 原运行身份包含数据清单哈希、代码源、tokenizer、采样契约、固定依赖和输出目录。普通 `--resume` 在清单变化后会拒绝旧检查点。迁移脚本先核验旧检查点和运行身份，逐步比对已完成的 ID 与 token 前缀，再创建独立续训目录。新封存记录两个清单和原检查点哈希，学生权重及 AdamW／RNG payload 保留原字节。WSL 用小 Qwen2 和实际两份发布验证了正式恢复入口可读，已应用更新的数据前缀有差异时会拒绝。
- 云端准备任务 `latent-three-poolfix-prepare/20261115T092744Z` 先读回约 18.12 GiB，随后报 `ValueError: Empty tokenizer directory`。第二次 `20261115T093035Z` 报 `LATENT_THREE_MIGRATION_READY step=1 next_step=2` 并以 0 退出。其 `command.json` 的 `save` 为空，Drive 当前可查到准备日志，还未找到新版检查点归档或新版训练任务产物。请把云端准备通过、产物持久化、新版开始采样三个阶段分别取证。
- 约 17:40，`colab status` 无法定位原会话，`colab sessions` 返回服务器无活动会话。原 `PUBLISHED.json` 的 `running` 是 17:23:37 写入的周期快照。暂无原任务文件 `job/exit_code` 和终止原因。现有切换启动脚本针对**原 VM 仍在且旧训练进程存在**的情况。若另开 VM，应审查从 Drive 恢复 `checkpoint-1`、保留绝对运行路径及源码、先复建并校验 tokenizer/audit，再迁移和从第 2 步重算的冷恢复顺序。若发现更晚完整封存，以实际清单和载入证据为准。

## RDC 核对文件与只读归档入口

下列路径均相对于上述 WSL 项目根目录。

| 审查内容 | 本地文件 |
| --- | --- |
| 仓库权限、历史实验、旧结果边界 | `AGENTS.md`、`README.md`、`docs/handover/kiro_ide_onboarding.md` |
| 实际训练、mask、词表与 HF rollout | `src/training/latent_mopd_three.py`、`src/losses/latent_opd.py`、`src/losses/mopd_v3.py`、`src/training/rollout_v3.py`、`src/training/sampling_v2.py` |
| 数据来源、自检、逐道修复 | `scripts/prepare_latent_mopd_three.py`、`src/data/decontamination_v5.py`、`runs/latent_mopd_three/release_poolfix_20261115/`、`runs/latent_mopd_three/selection_audit_20261115/loader_verification_poolfix_20261115.json` |
| G4 入口、运行身份与 checkpoint 迁移 | `scripts/colab_latent_three_formal.sh`、`scripts/probe_latent_mopd_three_joint.py`、`scripts/colab_job_v3.py`、`src/training/runtime_v3.py`、`scripts/migrate_latent_three_checkpoint.py`、`scripts/colab_latent_three_switch.sh` |
| 实际部署与切换命令 | `runs/latent_mopd_three/g4/deploy_codefix_20261115.ipy`、`runs/latent_mopd_three/g4/switch_to_poolfix_20261115.sh` 及旁边的压缩包校验文件 |
| 本地测试与评测现状 | `runs/latent_mopd_three/local_verification_20261115/full_tests.txt`、`tests/test_opd_v3.py`、`scripts/evaluate_latent_mopd_three_dev.py`、`scripts/score_code_local_v3.py` |
| 先前审查与论文 PDF | `docs/reviews/latent_mopd_three_progress_20261115.md`、`docs/reviews/latent_mopd_three_review_prompt_6pro_20261115.md`、`reports/argon_latent_mopd_audit_20261115/` |

归档可以用 RDC 在 WSL 运行下列只读命令。原任务的训练步级日志、探针、环境和运行身份位于 GitHub results。准备任务日志和权重存放在 Drive。

```bash
cd /home/xde10/projects/qwen-dual-teacher-opd
git show a8bf2517d5fe6d3274ba05df013e295e27d9da0f:latent-three-codefix-62/20261115T054641Z/full/metrics.jsonl
git show a8bf2517d5fe6d3274ba05df013e295e27d9da0f:latent-three-codefix-62/20261115T054641Z/evidence/joint16k/probe.json
rclone lsf mydrive:opd_runs/latent-three-codefix-62/20261115T054641Z/full/checkpoint-1 --recursive
rclone cat mydrive:opd_runs/latent-three-poolfix-prepare/20261115T092744Z/job/output.log
rclone cat mydrive:opd_runs/latent-three-poolfix-prepare/20261115T093035Z/job/output.log
```

## 请逐项给出审查结论

1. **吞吐、CU 与恢复优先级。** 用真实 phase 记录找耗时主因，区分观测和假设。请给出一项最小的等效提速探针、可接受的耗时或成本条件，以及达到该条件后才值得重开 G4 的理由。检查是否需要让检查点保存更密，估算额外磁盘和同步量。确认冷恢复是否会再次缺 tokenizer、错误匹配身份或丢掉已封存的优化器和 RNG。对 `MIGRATION_READY` 之后没有产物的情况给出证据判据。
2. **论文目标与训练数学。** 从 `read_schedule → run → HFRolloutEngine → score/objective → optimizer.step` 跟踪 math、code、logic 三域。核对共同 token 支持和 LM head 分母、rollout 与评分策略一致性、最后三层的预测位置和终层 RMSNorm、EOS/padding mask、奖励 std、跨微批聚合与 8-worker 等效条件。按论文原文指出已证实的差异和最小可验证假设。保留三教师身份和 62 步目标，任何配方修改请单独说明对复现和对照的影响。
3. **数据完整性、泄漏和难度。** 比对旧 G4 版和最新版的清单、真实源样本和已用训练前缀；独立抽查新替换的 14 个正式槽位和缺图／题干病例，判断是否还有同类缺陷或平台、任务偏差。核验逻辑的冻结题面与 verifier。说明如何用同一批有可信判分的训练题比较 Base 与对应教师，并分别报告题型、截断和正确率差。冻结 dev/test 继续用于各自的验证与报告用途。
4. **评测闭环。** 内部三域入口使用 HF greedy，默认 16K，数学报 Math-Verify 与保守判分，逻辑用对应 verifier，代码候选回本地 Docker 执行。当前有入口和 CPU 契约证据，未找到本轮 Base、Token-only、Latent 的成对实测分数。请核对无 `</think>`、截断和阻断题的分母，以及 256 数学／200 代码／200 逻辑 dev 的实际可比性。再列出论文 AIME24、AIME25、Minerva、LCB-e、MBPP、MBPP+、GYM、BBH、MuSR 九项缺哪些评测资产。分别给出相对 Base 的总涨幅与相对 Token-only 的 Latent 增量如何报告。
5. **交付判断。** 开头分别给 `GO / 条件 GO / NO-GO / 缺证`，覆盖零 CU 核验、冷恢复、最低成本提速探针和正式两组 62 步。问题按 P0/P1/P2 排，给源码位置、触发条件、后果、最小修复及验收结果。最后列出你实际用 RDC 读过的文件、跑过的命令、没拿到的证据和建议的执行顺序。

已有本地全量测试记录为 204 passed、1 skipped。新的迁移路径另做了本地恢复验证，未收到修复版完整的 GPU 训练与评测结果。项目目前可报告的是三教师方法的独立实现、真实 G4 工程验证和已测到的成本；能力提升需要对照分数支持。
