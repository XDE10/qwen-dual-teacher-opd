# 三教师 Latent-MOPD 复现审查与实施方案

- 日期：2026-11-14
- 本地仓库：`~/projects/qwen-dual-teacher-opd`
- 审查基线：本地 `main`，HEAD `4dee8bf`（与 `origin/main` 一致）
- 论文：Fang et al., *LATENT-MOPD: Latent Multi-Teacher On-Policy Distillation*, arXiv:2610.02381v1
- 目标：以现有 R1 1.5B 双域 Latent-OPD 代码为起点，完成论文同家族三教师 Math/Code/Logic 复现；不能把现有双教师结果冒充论文复现。

## 0. 一页结论

**结论：项目有可复用的算法原型，但还不是论文复现。** 现有 `src/losses/latent_opd.py` 与 `src/training/latent_opd.py` 已实现同家族双教师的 top-16 token 监督、最后三层 hidden-state 对齐、教师本地 crossfade、冻结教师和 62 步训练框架。最重要的工作不是从零搭训练框架，而是将其升级为真正的三域/三教师训练，并补齐数据、采样协议、恢复身份和成套对照。

当前已核实的最新结果（不是论文结果）：
- 训练：R1 1.5B + JustRL 1.5B（Math）+ Archer2 1.5B（Code），G4 完成 62 个 optimizer updates，Math/Code 各 31 步；checkpoint-62 的 manifest 校验通过，训练正常退出。
- 配对 dev（HF greedy，8,192 token 上限）：Math-Verify 从 66.41% 降到 63.28%（-3.13pp）；Code 全部 200 题从 35.00% 升至 39.00%（+4.00pp；剔除 8 道 blocked 后 +4.17pp）。数学截断 80→85，代码 98→97。
- 所以目前证据显示**代码方向有增益、数学方向负迁移**；既没有双域均 +5pp，也没有匹配预算的 Pure top-16 正式对照，更没有第三位 Logic 教师。
- 这轮不能直接写成“复现 Latent-MOPD”或“数学、代码双域显著提升”。可以写成“实现并完成双教师 Latent-OPD 原型，观察到跨域收益不对称”，并明确实验设置。

### 优先决策
1. **冻结现有双教师结果与代码身份，不覆盖旧 run。** 当前 `main` 是干净的（仅存在若干未跟踪审查文档和 `.agents/`），训练源码身份及历史结果均应保留。
2. **建立独立三教师分支和新数据版本。** 精确沿用论文同家族教师面板：Math=JustRL、Code=Archer2、Logic=Nemotron-Research-Reasoning-Qwen-1.5B；新增 Logic 数据必须经过独立来源、去重、manifest 与人工批准，不能把现有 v5 manifest 偷改成三域。
3. **先跑三教师 3-step GPU smoke + 1-step/单批梯度核验，再跑 62-step 正式实验。** 与论文同样采用 62 个全局更新、32 prompts/update、4 responses/prompt、temperature=1.0、16,384 max tokens、AdamW lr=1e-5、每教师 10-step crossfade、domain-pure updates。
4. **同一预算对照必须有 Base、Token-only、Latent-MOPD。** 先锁定三教师 Token-only 62-step，再跑 Latent-MOPD 62-step；否则无法把 latent 贡献从普通多教师 token distillation 中分离。
5. **简历成果以可复现的对照结果为准，不以论文表格数字为准。** 论文 Table 1 的 Norm=1.05、九项 benchmark 结果是作者实验，不能移植到本项目。

## 1. 论文方法：复现时不可删减的部分

### 1.1 核心计算图
每个 prompt 有固定领域标签 (d\in\{Math,Code,Logic\})。学生先生成自己的 response \hat y；对每个 response prefix，学生和该领域冻结教师在同一 token IDs 上做 forward：
- **Token channel**：学生 top-16 候选 (V_{k,t})（(k=16)）；学生和教师的 log-prob 都在完整共同词表上归一化；候选权重在 top-16 内重新归一化；reward 为加权的 teacher-vs-student log-prob 差；按该领域 rollout batch 的 reward 标准差缩放（不减均值）；使用 stop-gradient advantage 更新学生 log-prob。
- **Latent channel**：对齐学生与同一个 routed teacher 的 hidden states；同家族默认使用最后三个层级，identity map；先对 hidden feature 维做 L2 normalization，再按 hidden width 缩放平方误差；教师完全冻结，梯度只回学生。
- **Routing**：同一条 response 的 token 与 hidden-state supervision 必须来自同一教师。禁止三个教师对同一条 prompt 做平均或把不同教师的两个通道混搭。
- **Batching**：每个 optimizer update 只含一个 domain；Math/Code/Logic 更新按固定轮转。不要在同一 optimizer step 中混合三种教师的 latent gradients。
- **Crossfade**：每个教师单独计数 (u_d)，α(u)=min(1,u/T_w)，β(u)=max(0,1-u/T_w)。同家族 (T_w=10)。注意 u 是该教师已完成的更新数，首个更新 u=0，因此首步 token weight=0、latent weight=1；第 11 次该教师更新起 latent=0。
- **推理**：最终只部署学生模型；教师和投影器都不参与 inference。

### 1.2 论文默认同家族配置（论文 §3、§4.1、Appendix A/C）
| 项目 | 论文同家族默认 |
|---|---|
| 学生 | DeepSeek-R1-Distill-Qwen-1.5B |
| Math teacher | JustRL-DeepSeek-1.5B |
| Code teacher | Archer2.0-Code-1.5B |
| Logic teacher | Nemotron-Research-Reasoning-Qwen-1.5B（ProRL） |
| 模型关系 | 同一 DeepSeek-R1-Distill-Qwen-1.5B lineage；Qwen2；同宽同 tokenizer |
| latent target | 最后 3 个 depth-matched layers；identity projection |
| token support | student top-16；full-vocabulary log-prob normalization |
| λ | 1（same-family） |
| crossfade | 每教师 10 次更新 |
| 更新 | 62 global optimizer steps；domain-pure；Math→Code→Logic |
| 每步数据 | 32 prompts × 4 student responses |
| 预算 | 1,984 prompt presentations；7,936 responses |
| rollout | temperature 1.0；最多 16,384 response tokens；无 verifier reward |
| optimizer | AdamW；lr=1e-5；betas=(0.9,0.999)；weight_decay=0.01；eps=1e-8 |
| 数据池 | 17,856 prompts；每域 5,952 |
| 最终 checkpoint | 固定 step 62；论文不挑选中间最优 checkpoint |

论文 Table 1：同家族 Latent-MOPD(last 3) 在九项任务超过 token-only、representation-only、uniform averaging；五项超过最强单教师；Norm=1.05。论文 Figure 3/F 指出 all-layer representation-only 的 interleaved optimizer updates 会崩溃，而 domain-pure 更新稳定。论文 Table 2 是另一种 7B cross-family setting，不应与本项目 1.5B same-family 数字混淆。
## 2. 本地项目审计：已有能力、差距与风险

### 2.1 实际读取并核验的文件
- `AGENTS.md`、`README.md`、`docs/GPT6Pro_项目二进展审查与改进提示词_20261114.md)
- `src/losses/latent_opd.py`、`src/training/latent_opd.py`、`src/training/teachers_v3.py`
- `src/training/sampler_v3.py`、`src/training/latent_progress.py`、`src/training/runtime_v3.py`
- `src/losses/mopd_v3.py)、`scripts/colab_latent_top16.sh`
- `reports/g4_opd_v3_experiments_20261112/README.md)、`evaluation_matrix.json`、`reports/review_of_ultrathink_opd_report_20261111.md)
- `configs/project2_opd_v3_d4_dual.json)、`runs/latent_opd_assets/tokenizer_audit.json)
- 当前 main commit：`4dee8bf`；前序训练实现固定 commit：`d3c0242de47040853a353f5f61b21842eb540624`。

### 2.2 已实现且可复用
| 组件 | 当前代码事实 | 判断 |
|---|---|---|
| 模型加载 | 学生 R1-Distill-Qwen-1.5B；JustRL 与 Archer2 固定 revision；TeacherPool 冻结教师 | 已有双教师基础 |
| tokenizer audit | 三个现有模型共同支持 151,665 token IDs，且要求连续 | 三教师必须扩成四方审计（student+3 teachers） |
| rollout | 学生生成自身轨迹；同一 prompt token IDs 原样传教师；batch 按 response rows 计算 | 主链可复用 |
| token loss | top-16 student support、full-support log-probs、reward std scaling、masked response positions | 数学形式与论文 Appendix C 接近 |
| latent loss | `F.normalize` 后 feature-wise squared error；最后三层；教师 no_grad | 核心 latent 机制已存在 |
| crossfade | `loss_weights` 使用教师本地 completed_updates；默认 10 次 | 核心调度已存在 |
| domain-pure | `train_step` 每次只选一个 `opd.DOMAINS[step % len(DOMAINS)]` | 扩三域后可实现，但需改所有 domain 假设 |
| checkpoint | runtime identity、checkpoint closed-set manifest、哈希校验、optimizer/scheduler/trainer state | 有强身份机制；三域配置须成为新 identity |
| telemetry/spool | 每批 rollout 有独立压缩缓存、SHA、phase metrics | 可复用；恢复时的生成策略指纹仍不完整 |

### 2.3 必须修正或补齐
1. **只有两域**：`src/training/latent_opd.py` 的 `TEACHERS` 只有 math/code；`opd.DOMAINS`、sampler、pool loader、prefix 构造、progress、smoke assertions、metrics 汇总均按双域写死或依赖双域列表。不能只在 TEACHERS 字典中追加 logic 就认为完成。
2. **没有 Logic 数据**：v5 approved release 只有 math/code。当前数学 train 2,500、代码 train 2,488；没有 5,952 条逻辑训练 prompt，也没有 Logic verifier/冻结样本清单。论文 Logic 域用 Reasoning Gym；必须新增数据发布，不能混用旧 manifest 或直接从公开 benchmark 测试集抽题训练。
3. **采样协议不同**：`StratifiedSampler` 会按 strata shuffle，并以随机 offset 合并顺序；论文 Appendix A 明确禁用 data shuffling，按 Math→Code→Logic 顺序处理 domain-pure blocks。为了论文对照，新增一个 frozen ordered sampler（或配置化 `shuffle=False`），并记录排序 digest；保留现有 stratified 模式作为项目改编实验，不要静默替换旧 run。
4. **目前的双域交替不是三域轮转**：当前 smoke 断言 `['math','code','math']`，新三域 smoke 必须断言 `['math','code','logic','math','code','logic']`；62 步必须是 Math 21、Code 21、Logic 20。
5. **teacher-local clock 应显式持久化**：现在由 `step // len(DOMAINS)` 隐式推算。三域时理论上仍正确，但建议 checkpoint 记录每个 teacher 的 completed_updates，恢复时验证总和=global step，并防止未来不规则调度时 crossfade 计数错误。
6. **恢复缓存键**：`RunProgress.rollouts` 的 key 当前绑定 run_id、step、domain、indices、prefixes，不包含生成模型同步后的实际权重指纹。历史审查已确认首次恢复时第 22 步复用轨迹的严格 on-policy 性质不可追溯。新三域版本应在 `GenerationCopy.sync_from` 后计算/记录 student generation weights fingerprint，并将它纳入 spool key；旧格式只读兼容，不能伪装成已验证的严格 on-policy。
7. **跨域损失和权重**：当前 `hidden_alignment` 是对层、token、hidden feature 求均值，隐含 λ=1；对同家族论文设定合理。新增逻辑教师前必须确认隐藏维、层数、final norm 位置和 token IDs 一致，不要对 Logic 使用跨家族 projector 方案。
8. **论文复现的数据顺序和重复数**：每域 5,952 条 pool，但 62 步仅消耗 21/21/20 × 32 个 prompt presentations；要固定并保存 1,984 个实际 presentation IDs，防止把 pool size 当作训练实际使用样本数。

### 2.4 现有双域结果的正确解释
| 证据 | 结果 | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| 62-step 训练 | 31 Math + 31 Code updates；checkpoint-62 manifest 校验通过 | 真实模型训练流程完成 | 不代表论文三教师复现 |
| Math-Verify dev | Base 66.41% → Latent 63.28%（-3.13pp） | 当前 latent+crossfade 双域方案在该设置下数学负迁移 | 不能说数学提升 |
| Conservative math dev | 53.13% → 50.39%（-2.73pp） | 另一判分口径方向一致 | 不能取代 Math-Verify 主口径 |
| Code dev 200 | 35.00% → 39.00%（+4.00pp） | 代码方向有初步增益 | 不能推断 HumanEval+/MBPP+ 同幅度提升 |
| Code dev 192 | 36.46% → 40.63%（+4.17pp） | 排除 8 道 blocked 题后方向仍一致 | 不能与 200 题口径混报 |
| truncation | Math 80→85；Code 98→97 | 输出截断未改善，数学甚至略恶化 | 不能把全部差异归因于算法能力 |
| Pure top-16 | 尚无匹配预算正式成绩 | 当前不能隔离 latent channel 的增量 | 不能将 Latent−Base 说成 latent-specific gain |

**最重要的反例**：Latent−Base 同时改变了 teacher token target、hidden-state loss、crossfade、训练更新数与生成轨迹；它只测“整套方案相对 Base 的总效果”。要声称 hidden-state supervision 有贡献，必须再比较同样三教师、同样采样、同样 62 updates 的 Token-only 与 Latent-MOPD。

### 2.5 三个足以推翻乐观结论的替代解释
- **长度/截断混杂**：数学截断从 80 增至 85。即便 latent 对部分问题有帮助，8K dev 上限也可能截掉有效推导；需要固定 8K 与 16K 的配对敏感性分析，报告平均输出长度、EOS 率和截断数。
- **样本量波动**：Math dev 256、Code dev 192/200；+4pp 对应约 8 个代码题，单次 greedy 的逐题差异可能受少数样本影响。报告 paired bootstrap/discordant pairs，不用单一比例差异宣称统计显著。
- **恢复时轨迹复用**：第 22 步 spool 缓存缺少 student weight fingerprint，不能严格确认轨迹由当前 student policy 生成。新 run 必须解决，不得将这一疑点从历史记录中抹去。
## 3. 三教师模型与开源资源选择

### 3.1 首选：严格按论文同家族教师面板
| 领域 | 模型 | 参数/架构（HF 元数据） | 选择理由 | 注意事项 |
|---|---|---|---|---|
| Student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 约 1.777B，Qwen2 | 与论文同家族学生 | 当前本地 pin 为 `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`；论文未在 PDF 中给出同一 revision，需声明是复现采用的固定 revision |
| Math | `hbx/JustRL-DeepSeek-1.5B` | 约 1.777B，Qwen2；base_model 指向 R1-Distill-Qwen-1.5B | 论文 Appendix A 指定的数学教师 | 当前 pin `0637e4096c789c67f9eecbe8355e0bdeddede1c2` |
| Code | `Fate-Zero/Archer2.0-Code-1.5B-Preview` | 约 1.777B，Qwen2 | 论文 Appendix A 指定的代码教师 | 当前 pin `db1e1ff9f3c36f7177f3896358ce26edeb62ce56` |
| Logic | `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` | 约 1.777B，Qwen2；HF tags 指向同一 R1-Distill-Qwen-1.5B lineage | 论文 Appendix A 指定的逻辑教师（ProRL） | HF 元数据标注 `cc-by-nc-4.0`；非商业许可，适合个人研究/作品集展示但需保留许可声明；必须 pin immutable revision |

这是当前最稳妥的首选，因为它最大程度保留论文的 same-family 条件：相同 backbone、隐藏宽度、层数、tokenizer 和最终预测接口，直接使用 identity mapping，无需 projector。不要先选 7B cross-family 三教师：论文 Table 2 是单独实验配置，默认只对齐最后一层并使用共享 linear projector，不能与 Table 1 同家族结果混合比较。

**实施前的模型准入门槛**（Logic teacher 未过关不得开始正式训练）：
1. 对 student + 3 teachers 分别记录 HF repo ID、immutable commit SHA、license、config/tokenizer 文件 SHA256。
2. 检查 `model_type=qwen2`、`num_hidden_layers=28`、`hidden_size=1536`、`vocab_size >= 151665`；对比全部 tokenizer 的 token→ID 映射，要求零差异且 ID 0..151664 连续。
3. 验证 EOS/PAD/BOS/特殊 token 与 chat template。训练仍只用学生模板 tokenize 一次，再把同一 input_ids/attention_mask 传给教师；不得让教师重新套模板或二次 tokenize。
4. 对同一批 8–16 条固定前缀，逐教师运行 forward，确认 logits、last hidden state、last-three hidden-state shapes 均匹配预期且有限。
5. 记录每个模型的 license 到 `MODEL_CARD.md`，特别标注 Logic teacher 的非商业条款；不把模型权重上传到自己的公开仓库。

### 3.2 训练数据来源
论文 Appendix A/B 给出的来源：
- Math：DAPO-Math-17k prompts。
- Code：OpenCodeReasoning prompts（只用问题；数据集中的参考解不是训练 imitation target）。
- Logic：Reasoning Gym 的程序化任务 prompts；论文使用冻结生成样本，训练池 5,952 prompts。
- 每域 pool 5,952，共 17,856；训练 62 updates 实际只 presentation 1,984 prompts，4 responses/prompt，共 7,936 student responses。训练和评测题目严格隔离。

本项目当前 v5 release 只批准 Math/Code（2,500 / 2,488），所以不能直接复用为三域训练集。推荐新建独立 `project2_release_v6_three_domain`，不覆盖 v5：
- Math 与 Code 尽量复用 v5 已批准的训练样本，另外补齐到 5,952 的新增候选；旧样本来源/ID/hash 原样继承，新增样本另做去重和人工批准。
- Logic 从 Reasoning Gym 0.1.19 的固定 generator registry + 固定 seed 生成；冻结具体任务配置、参数、seed、question text、verifier metadata 和每条 sample_id。不能仅记录 seed，因为 generator registry/version 变化会改变实例。
- 对三域 train/dev/test 做 exact hash、规范化文本、近重复和 benchmark overlap 检查；dev/test 永不进入训练。
- Logic dev 至少保留一个固定 200–256 题的 in-loop/dev 子集；另留独立最终逻辑测试（如 GYM/BBH/MuSR 的冻结实例）。训练 verifier 不给奖励，只用于审计/最终评测。
- 负责人需按项目既有流程明确批准新增文件；AI 不得自行篡改 `training_approved`、manifest SHA 或批准记录。

### 3.3 外部代码库与项目可借鉴性
| 项目 | 可用价值 | 不应误用之处 |
|---|---|---|
| [官方 Latent-MOPD 仓库](https://github.com/fangzy96/Latent-MOPD) | 论文作者明确给出的官方地址 | 截至本次检查，README 只有 “Code is coming soon.”；没有可直接 clone 的算法实现。因此不能声称已找到官方实现或直接复现官方代码 |
| [Open-MOPD](https://github.com/BytedTsinghua-SIA/Open-MOPD) | 公开的完整多教师 on-policy pipeline；三教师 domain routing、训练/评测 launcher、verl 实现、HF 模型与数据发布；可借鉴多教师配置、数据/实验组织、训练日志和评测流程 | 它研究的是 capability imbalance / optimization budget，不含 Latent-MOPD 的 hidden-state alignment 和 per-teacher latent crossfade；SmolLM3-3B 模型面板也不同。不要整套替换现有 R1 训练架构 |
| [OPRD](https://github.com/ShenzhiYang2000/OPRD) | 有 on-policy representation distillation 实现、hidden-state pairing、projector/bridge 代码，可用于核对 representation extraction 和训练接口 | OPRD 是单教师研究，工程基于 LlamaFactory/verl 改造；本地已有更轻的 R1/HF rollout pipeline，不建议整体迁移 |
| [Awesome-OPD](https://github.com/kissshhot/Awesome-OPD) | 追踪 OPD/MOPD/Latent-OPD 后续方法、基线与评测论文 | 是文献索引，不是可运行复现代码 |
| [H-OPD](https://github.com/buptyqx/H-OPD) | 可作为异构/多级 OPD 相关研究线索 | 与论文主算法不是等价实现，先读论文再判断是否纳入 related work |

**技术选择**：保留本地 HF + PyTorch + LoRA + 自定义 loss pipeline；只从 Open-MOPD 借鉴三域训练/实验组织，从 OPRD 借鉴 hidden-state API 和 projector 对照。这样能让简历项目的核心贡献是自己实现 Latent-MOPD，而不是“把一个大型框架跑起来”。

### 3.4 资源可行性
三位 1.5B BF16 教师权重大约各 3.3 GiB，三教师常驻相对当前双教师约增加 3.3 GiB（仅权重粗估，不含运行时、激活和 KV cache）。G4 有 95 GiB，但本地历史 top-16 rollout 曾观测到约 93.9 GiB；因此不能仅凭参数量断言 resident 一定安全。
- 首轮 smoke 必须记录模型加载后、rollout、teacher forward、student backward 的 allocated/reserved/peak VRAM。
- 如果 resident 峰值安全余量不足（建议至少 6–8 GiB），启用 domain-pure 的 teacher CPU staging / swap：同一 update 只把当前 teacher 放 GPU，其他教师留 CPU；切换发生在 optimizer update 边界。
- 不允许通过 4-bit/8-bit 量化教师来“先跑起来”，除非单独做模型身份/精度消融；论文基准是原始 checkpoint，不应混淆。
- 正式预算按现有 R1 run 估计：8 个额外 update 约 4.6 G4 GPU-hours，折合约 34.5 分钟/update；三域 62-step 训练粗估 35–45 G4 GPU-hours，Logic 输出长度和 teacher swap 可能扩大范围。该估算不是已测值，正式 smoke 后更新。
## 4. 三教师实现蓝图（保持旧实验可复现）

### 4.1 分支与身份隔离
建议从当前 `main@4dee8bf` 建立新分支，例如 `feat/latent-mopd-three-teacher`。不要把三教师实验直接追加到旧的 `latent_mopd_top16_latent` 输出目录；不要改写旧的 `run_identity.json`、checkpoint manifest 或 v5 release。
- 新代码版本：`latent-mopd-3teacher-v1`。
- 新配置：`configs/latent_mopd_3teacher_same_family.json`。
- 新数据发布：`data/project2_release_v6_three_domain/release/manifest.json`。
- 新输出：`runs/latent_mopd_3teacher/<run_id>/`；Colab VM 使用 `/content/runs/latent_mopd_3teacher/<run_id>`，Drive 用独立 `mydrive:opd_runs/latent_mopd_3teacher/<run_id>/`。
- 新运行身份必须绑定：student/teacher repo+revision、全部 config/tokenizer hashes、v6 manifest hash、prompt contract、domain order、sampler mode/order digest、loss source hashes、generation contract、torch/transformers/peft pins、rollout policy fingerprint。
- 旧双教师代码和历史结果保持可读；必要时只抽取通用 helper，不得为兼容新 run 改旧 checkpoint 的 hard identity。

### 4.2 代码改造范围
| 文件/模块 | 改造点 | 验收标准 |
|---|---|---|
| `src/training/latent_opd.py` | 新建三域专用训练入口或升级为配置驱动；加入 Logic teacher；移除所有硬编码双域假设；输出每域计数 | 62 步域序列严格 21/21/20；同一 response 两个 loss 通道 teacher_id 一致 |
| `src/losses/latent_opd.py` | 保持 top-16 + latent loss；将 TOP_K、layer count、lambda、Tw 读入显式 config；token-only 路径必须完全跳过 hidden-state forward | 解析梯度对照、loss 数值对照和 no-grad teacher 测试通过 |
| `src/training/teachers_v3.py` 或新 `latent_teacher_pool.py` | 3 个 teacher 固定 revision；resident/cpu_swap 可选；每次 domain update 只取一个 teacher | 三教师全 frozen；GPU active teacher 与 domain route 一致；无隐式多教师前向 |
| `src/training/sampler_v3.py` 或新 sampler | 新增 `ordered_frozen` 模式，保持 manifest 文件顺序；记录每域 order SHA；旧 stratified sampler 保留 | 重启前后 sample IDs 完全一致；62-step presentation IDs 可审计 |
| `src/training/latent_progress.py` | spool key 增加生成策略权重 fingerprint；写入 tokenizer/model/revision/domain/prompt IDs 和 rollout config；旧 spool key 标记为 legacy | 修改 student generation weights 后旧 spool 必须 miss；同 checkpoint + 同 policy 才能 hit |
| `src/training/runtime_v3.py` 或新 runtime schema | 新 identity schema，不更改旧 schema；包含 teacher-local counters、ordered sampler digest、spool policy version | 旧 checkpoint 拒绝续入新 schema；新 checkpoint 能完整 resume |
| `scripts/colab_latent_top16.sh` | 新建独立 `colab_latent_mopd_3teacher.sh`；smoke 后才启动 full；自动释放监控进程 | smoke 失败不启动正式 run；每步 metrics/checkpoint 同步 |
| `tests/` | 增加三域 contract/loss/routing/resume/data/eval tests | 所有新测试在 CPU tiny models 上通过 |

### 4.3 算法层面必须写成显式不变量
- `DOMAINS = ("math", "code", "logic")`；global step (s=0..61) 的 domain = `DOMAINS[s % 3]`。
- 每域计数器 `u_d) 仅在该域 optimizer update 完成后加 1；checkpoint 保存 `{"math":21,"code":21,"logic":20}`。
- `token_weight=min(1,u_d/10)`；`latent_weight=max(0,1-u_d/10)`。第一轮每域首步都是 latent-only；第 11 次本域 update 开始 token-only。
- Latent 默认 `layers=last_three`、`lambda=1.0)、`positions=all valid response prediction positions`；每层先独立做 feature normalization，再对 hidden feature mean，最后对层和有效 response positions 求均值。不要因三层而把 λ 放大三倍。
- Token loss：student top-16；student/teacher log-prob 必须按完整词表归一化；top-16 内 student probability 重归一化作为权重；reward 在**同一个 domain rollout batch** 内按样本标准差缩放，不中心化；token 候选维度求和，response position 做 masked mean。
- 同一个 optimizer update 只来自一个 domain。跨域梯度不做同一步平均；不需要 PCGrad。
- 教师保持 eval + requires_grad_(False)，teacher forward 全部 no_grad；推理时完全不加载 teacher。
- 所有 prompts 由 student tokenizer/template 渲染一次；同一 input_ids、attention_mask、position semantics 传给 teacher。禁止各教师独立 chat template。

### 4.4 必须先验证的表示层细节
当前 `_layer_rows(..., last_three=True)` 通过 decoder 的 `output_hidden_states=True` 取 `hidden_states[-3:]`。对标准 HF Qwen2Model，hidden_states 包含 embedding 输入、各 block 的输入/输出序列以及最后 RMSNorm 后的 final hidden state；末三项应对应 block 26/27 的输出和 block 28 final-normalized output。仍需在实际固定 transformers 版本上用 forward hook 验证，不凭索引猜测。
测试判据：
1. 用 hook 记录最后三个 decoder block 的输出及 final RMSNorm 输出。
2. 对照 `hidden_states[-3:]` 的 tensor，确认第一、第二项分别与 block 26/27 输出一致，第三项与 final RMSNorm 输出一致。
3. 确认 causal shift：预测 response token (y_t) 的 hidden state 来自其前一个 prefix position，而不是 token 自身之后的位置。
4. 对 student/三个 teacher 分别跑同样的断言。
5. Token objective 应始终使用 final-normalized hidden state（即 `student_layers[-1]`）接 LM head；latent loss 对齐选定的三个层对。
6. 另外核验实际 lm_head 行数与 tokenizer common support。论文描述 full-vocabulary normalization；如果模型 lm_head 有 padding rows 超过 151,665，不能默认它们概率质量为零。应测完整 head 的 out-of-support probability mass；若非零，loss logsumexp 必须覆盖真实有效词表，而不能只在截断 support 上重新归一化。

### 4.5 关键测试清单
1. **三域路由**：每个 batch 的 domain、teacher revision、token target teacher、latent target teacher 必须完全一致；伪造错误 teacher ID 应失败。
2. **Crossfade 边界**：对每域 u=0,1,9,10,11,20,21 检查 α/β；更新 Math 时 Code/Logic counters 不变。
3. **62-step schedule**：domain 序列重复 Math/Code/Logic 20 次后 Math/Code；计数精确 21/21/20；三教师都经历相同的 u=0..20 clock。
4. **Loss 数值**：在 tiny Qwen2 与小词表上，top-16 reward、std scaling、masked reduction 与独立 reference 实现逐元素对照；梯度与 autograd/reference 误差阈值写死。
5. **Representation gradient**：student hidden 有梯度；teacher hidden 和 teacher parameters 无梯度；最后三层分别收到梯度；layer mean 不随层数线性放大。
6. **Token-only**：确认不调用 teacher hidden-state extraction，不保留任何 latent graph；teacher 仍只做 token scoring。
7. **Sampler**：ordered mode 在固定 manifest 下重启、resume 后 presentation IDs 完全一致；改变 manifest/order digest 必须拒绝 resume。
8. **Spool on-policy**：同步学生权重后生成 fingerprint；只要 fingerprint、tokenizer revision、generation config、prefix、domain、seed 任一变化，cache miss；损坏 gzip/hash 不能被复用。
9. **Checkpoint**：保存/恢复 optimizer、scheduler、sampler state、per-domain counters、RNG、generation fingerprint；从 step 30 恢复后下一个 domain 与 crossfade weight 与 uninterrupted run 一致。
10. **Teacher memory**：resident 与 cpu_swap 两种模式输出 logits/hidden state 在容差内一致；切换只能发生在 optimizer boundary。
11. **Padding/shift**：含不同 prompt/response 长度的 batch，右 padding、response mask、prediction positions、teacher/student input tensors 一致；padding 不进入 loss denominator。
12. **Failure behavior**：空 response、nonfinite loss、teacher load failure、OOM、checkpoint manifest mismatch 都必须 fail closed，不允许静默跳过域或伪造完整 62 步。

## 5. 实验矩阵与验收门槛

### 5.1 实验阶段
| 阶段 | 目标 | 数据/预算 | 预计 G4 时间 | 通过标准 / 停止条件 |
|---|---|---:|---:|---|
| A. 资产审计 | Logic teacher/tokenizer/license + v6 data manifest | CPU/Hub；不训练 | 0 GPU h | 四模型架构/词表/模板审计零差异；数据来源与批准完成 |
| B. Tiny contract | 三域路由、loss、crossfade、checkpoint、spool | CPU tiny models | 0 GPU h | 新测试全部通过；旧双教师测试不回归 |
| C. GPU smoke | 每域至少一次真实 forward/backward；再跑 6–9 global updates | 每步 1 prompt × 1 response；短输出 | 0.5–1.5 h | 无 OOM/NaN；三教师 logits 与 hidden shapes 正确；三域都产生有限梯度；checkpoint resume 一致 |
| D. Full token-only | 三教师 MOPD-style baseline | 62 steps；32×4；16K | 约 35–45 h | 正常完成 62 updates；每域计数 21/21/20；checkpoint-62 和 rollout audit 完整 |
| E. Full Latent-MOPD | 三教师 latent + token crossfade | 与 D 完全同预算/数据/seed policy | 约 35–45 h | 同上；三域 crossfade 完整；无非法 spool reuse |
| F. Dev evaluation | Base / Token-only / Latent 三模型配对 dev | 三域 dev 固定题目 | 视生成长度，另计 3–8 h | 完整逐题结果、输出长度、截断、EOS、paired CI；dev 选模型 |
| G. Final paper-style eval | Table 1 九项任务及教师上限 | GYM、BBH、MuSR、LCB-e、MBPP(+)、AIME24/25、Minerva | 视 sample counts，单独预算 | 所有模型使用相同 prompt、generation config、verifier 版本；最终测试集仅一次锁定评测 |

G4 时间是依据现有 R1 实验约 4.6 G4-hours/8 additional updates 的项目记录估算，不是三教师实测值。首次 smoke 后用真实每域 step time 更新。若 Logic 生成更长，或 CPU teacher swap 增加开销，按实际观测上调。

### 5.2 必须完成的对照
| Arm | 初始化 | Teacher panel | Objective | Steps | 作用 |
|---|---|---|---|---:|---|
| Base | 原始 R1 1.5B | 无 | 无 | 0 | 基线 |
| MOPD-style token-only | 同一 Base R1 | Math+Code+Logic | top-16 token channel；α=1 全程 | 62 | 普通三教师 token 蒸馏对照 |
| Latent-MOPD | 同一 Base R1 | 完全相同三教师 | last-3 latent + top-16 token crossfade | 62 | 论文主方法 |
| Constant joint（可选但强烈建议） | 同一 Base R1 | 相同三教师 | token weight=1、latent weight=1 全程 | 62 | 分离 crossfade schedule 的贡献 |
| Rep-only（可选） | 同一 Base R1 | 相同三教师 | last-3 representation only | 62 | 复现论文单通道控制；但 λ/positions 按论文 Appendix A 单独配置 |

核心增量是 `Latent-MOPD - Token-only`；总收益是 `Latent-MOPD - Base`。若只做 Base vs Latent，不能将收益归因于 hidden-state channel。Constant joint 对照能检验 crossfade 的价值，论文 Table 1 报告 crossfade 相对同配置 constant joint 在 9 项中 8 项更好。

### 5.3 dev/test 评测协议
- **模型选择只看 dev**。保留 Math-Verify 主口径与 conservative secondary score；代码分别报告 200 题全量与剔除 8 blocked 后的 192 题。Logic dev 单独报告每个 task 和 macro average。
- Base、Token-only、Latent 三个模型必须使用同一 tokenizer/template、相同 prompt IDs、相同 generation 参数和相同 verifier/Docker image；保存每题 raw output、parsed answer、correctness、tokens、EOS/truncation。
- 主评估建议沿用论文 final benchmark 的 sampling protocol（T=0.7、top-p=0.95；AIME24/AIME25 16 samples，Minerva 4，BBH/MuSR/GYM 1，LCB-e 4，MBPP 2）；若先做便宜的 greedy 8K/16K 诊断，必须另标为诊断口径，不能与论文主表分数混用。
- 代码题必须在固定 Docker 沙箱内运行；抽取失败按错误样本计分，基础设施失败/缺失结果不能默认为错误或跳过，必须单独标记并使整套评测失败，修复后重跑。
- 逐题配对报告：百分点差、paired bootstrap 95% CI、win/tie/loss 数、McNemar discordant pairs、输出 token 中位数/P95、截断率、EOS 率。单 seed 只作为 case study，不称统计显著。
- 256/192 的 dev 样本对 +5pp 的判断能力有限；若要作强结论，预先定义两轮固定 seed/采样复测或扩大独立 dev，不能看完结果再挑题。
- 最终 test 只在 dev 锁定 checkpoint 和配置后跑一次；MATH-500/GSM8K test、HumanEval+/MBPP+、BBH/MuSR/GYM test 不得回流训练。

### 5.4 停止与回退规则
- 任一教师 tokenizer/hidden shape mismatch、数据 overlap、license/approval 未解决：停止在 Stage A，不启动 GPU。
- smoke OOM：先降低 rollout concurrency / microbatch tokens，或启用 teacher cpu_swap；不先量化教师，不改 loss 来掩盖 OOM。
- smoke 有 NaN、teacher 梯度非零、跨域 teacher route 错误、checkpoint resume 不一致：停止，不启动 full。
- 正式训练中某域连续 3 次更新 loss/grad 非有限，或截断率急剧上升：保存可验证 checkpoint 与日志后停机；不继续烧预算。
- Full Latent 训练完成但 dev 某域明显下降：先比较 Token-only 与 Base，再分析截断/长度/teacher gap；不要立即延长训练或启动 math20。
- 不建议先跑 70-step 延长。论文默认主实验是 62 updates；Appendix F 的 120 updates 是 all-layer representation-only 的 interleaved batching collapse control，不是默认 Latent-MOPD 训练步数。任何 62→70 必须是另一个 run identity 和独立实验，不替代 62-step 主对照。

## 6. 项目现有评测结果与论文表格的区别
- 论文 Table 1 是 same-family 1.5B 三教师、三领域、9 个 benchmark；本项目截至 2026-11-14 只有 Math+Code 双教师，8K greedy dev，不能直接比较 Table 1 的 Norm 或准确率。
- 目前 Base R1 dev 66.41% math / 35.00% code，Latent @62 63.28% / 39.00%。这既没有复现论文的逻辑域，也未达到数学+代码双域 +5pp 的目标。
- 论文报告的 same-family Latent-MOPD Table 1：GYM 52.4、BBH 66.3、MuSR 52.4、LCB-e 73.1、MBPP 65.7、MBPP+ 55.7、AIME24 50.8、AIME25 36.9、Minerva 36.3，Norm=1.05。这些仅作为论文参考值，不是本地项目成绩。
- 当前本地模型在 v5 Math/Code 数据上训练；论文数据池为 17,856 prompts、每域 5,952。必须独立建立 v6 数据发布后，才能称“paper-faithful three-domain replication”。

## 7. 按优先级执行清单

### 第 1 步：完成无 GPU 资产与数据审计
1. pin Logic teacher 的 immutable HF revision，并确认 `cc-by-nc-4.0` 许可；在模型卡和最终 README 中注明。
2. 用四模型 tokenizer audit 检查 151,665 common token IDs、special tokens、chat template、hidden_size/layers/final norm。
3. 获取 DAPO-Math-17k 与 nvidia/OpenCodeReasoning 的数据结构，抽取 prompts 而非 teacher solutions；Reasoning Gym 0.1.19 冻结 Logic 实例。
4. 生成 v6 候选 manifest、overlap/near-duplicate 报告、每域 5,952 题清单；由项目负责人批准后再训练。
5. 新建三教师分支，先完成 CPU tiny tests；不得改写现有双教师 run identity 或 v5 manifest。

### 第 2 步：完成三域实现与 GPU smoke
1. 实现 explicit domain registry、ordered sampler、teacher-local counters、3-teacher pool、generation fingerprint spool。
2. 测试 last-three hidden-state 与 final RMSNorm、causal shift、full-vocabulary logsumexp。
3. GPU smoke 至少让三个 teacher 各完成一次 rollout→teacher scoring→latent/token loss→backward→optimizer→checkpoint。
4. 对照 resident 与 cpu_swap 峰值显存，选择有安全余量的模式。
5. 只有 smoke 通过后才允许启动正式 62-step run。

### 第 3 步：正式实验、评测与简历归档
1. 先跑三教师 token-only 62 steps，再跑三教师 Latent-MOPD 62 steps；共享初始化、数据、域顺序、seed 规则和 generation contract。
2. 先做 dev paired evaluation，锁定 checkpoint；再做论文九项 benchmark 的 final evaluation。
3. 保存 config、run identity、teacher revisions、manifest、每步 metrics、checkpoint manifest、rollout audit、逐题预测、Docker image digest、统计分析脚本。
4. README 清楚写出“论文方法复现/项目改编/尚未复现”的边界；公开代码与配置，不公开模型权重、tokenizer 私有缓存、数据受限内容或任何访问凭据。

### 简历表述
**当前可核查版本（不要夸大）**
> 基于 PyTorch/Hugging Face 实现 1.5B LLM 双教师 On-Policy Distillation 与 Latent Representation Alignment，完成 G4 单卡 62-step 训练、checkpoint 哈希校验和逐题 dev 评测；实现 top-16 token distillation、hidden-state matching 与 per-teacher crossfade，并分析到数学/代码能力收益不对称。

**三教师版本完成后（只有达到门槛才使用）**
> 复现并扩展 Latent-MOPD：将 Math/Code/Logic 三个同家族 1.5B 专家蒸馏至单一 1.5B 学生，复现 domain-routed token+latent supervision、last-three-layer alignment、domain-pure updates 与 per-teacher crossfade；在 [benchmark] 相对 Token-only MOPD 提升 [X]pp（95% paired CI [L,U]），单模型部署无需教师。

若三教师完成但未超过基线，仍可写“实现三教师 Latent-MOPD 并完成受控实验”，不能写“性能提升”。

## 8. 资料与证据索引

### 本地仓库
- `AGENTS.md`：当前双教师项目的身份、数据审批、Docker、checkpoint hard identity 红线。
- `README.md`（当前 main HEAD 4dee8bf）：62-step Latent 训练完成及 Base-vs-Latent dev 对照。
- `docs/GPT6Pro_项目二进展审查与改进提示词_20261114.md`：运行恢复疑点、固定 snapshot 和评测边界。
- `src/losses/latent_opd.py`：top-16 loss、hidden alignment、crossfade。
- `src/training/latent_opd.py`：双教师训练入口与 audit。
- `src/training/teachers_v3.py`：TeacherPool resident/cpu_swap。
- `src/training/sampler_v3.py`：当前分层随机 sampler（非论文 no-shuffle 顺序）。
- `src/training/latent_progress.py`：rollout spool；当前 key 缺 student weights fingerprint。
- `src/training/runtime_v3.py`：hard/soft run identity 与 checkpoint closed-set manifest。
- `reports/g4_opd_v3_experiments_20261112/README.md`：旧 Qwen2.5 双教师实验，不应与 R1 Latent 结果混合。
- 本报告创建于 `reports/latent_mopd_three_teacher_reproduction_plan_20261114.md`。

### 外部资源
- Paper PDF: `arXiv:2610.02381v1`，重点 §3、§4.1–4.4、§5、Appendix A/C/E/F/H。
- Official code URL: https://github.com/fangzy96/Latent-MOPD （本次检查 README 内容为 “Code is coming soon.”）
- Open-MOPD: https://github.com/BytedTsinghua-SIA/Open-MOPD
- OPRD: https://github.com/ShenzhiYang2000/OPRD
- Awesome-OPD: https://github.com/kissshhot/Awesome-OPD
- Student: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Math teacher: https://huggingface.co/hbx/JustRL-DeepSeek-1.5B
- Code teacher: https://huggingface.co/Fate-Zero/Archer2.0-Code-1.5B-Preview
- Logic teacher: https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B
- Math data: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
- Code data: https://huggingface.co/datasets/nvidia/OpenCodeReasoning
- Open-MOPD data (仅作多域数据工程参考，不是论文同一训练集): https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data
- Logic data: Reasoning Gym 0.1.19（论文 Appendix A 的固定版本；需冻结生成实例和 task registry）

## 9. 当前审查边界
- 已直接读取本地 WSL 项目文件、当前 main git 状态/提交历史、最新 README、论文 PDF、GitHub 外部代码库 README、HF 模型/数据仓库元数据。
- 已核实本地 HEAD 为 `4dee8bf`，`origin/main` 同步；工作区有未跟踪审查文档和 `.agents/`，未修改任何训练源码、配置、数据 manifest 或 checkpoint。
- 本次只创建本报告；没有启动 Colab/G4 训练，没有下载模型权重，没有改变 Git 分支或提交，也没有更改 GitHub/Drive。
- 未在本轮独立重跑 62-step 训练和所有 benchmark；62-step 训练与 dev 数字来自当前 main README 与已合并 PR #3 的发布记录。它们作为已归档项目证据引用，不等同于本轮重新执行。
- 未能从官方 Latent-MOPD 仓库取得训练代码，因为截至检查时 README 明确写着代码尚未发布。因此“完整复现”必须依论文公式、Appendix A/C 和外部基础设施自行实现，并在项目中透明标记为 independent reproduction。
- 第三教师的 revision SHA、tokenizer 兼容性、Logic 数据批准和 G4 三教师峰值显存仍是正式训练前的阻断项。

**最终裁决**：这条路线技术上可行，而且当前项目已有大部分关键算法组件；但最先要做的是 Logic teacher + Logic data 的准入、三域训练循环和恢复语义，而不是立刻开 62-step 卡。完成这些后，才能将项目从“双教师 Latent-OPD 探索”升级为有可信对照和证据链的“三教师 Latent-MOPD 复现”。
