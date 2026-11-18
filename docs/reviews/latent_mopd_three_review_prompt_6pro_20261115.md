# 给 6 Pro · 三教师 Latent-MOPD 上卡前技术审查

你是这次实验的独立审查人。我要判断这个实现值不值得花 Colab CU，以及怎样用最少的 GPU 验证发现会使正式 62 步白跑的问题。请审查 ZIP 内的实际源代码和候选数据。正文、旧评审报告和 README 只提供线索。结论需要来自你亲自读到的证据。

## 任务边界

- 目标固定为 Latent-MOPD v1 **Table 1 的同家族三教师 last-3 独立复现**。学生是 Base DeepSeek-R1-Distill-Qwen-1.5B，教师是 JustRL、Archer2、Nemotron-ProRL v2，覆盖数学、代码、逻辑。正式比较使用 Base、同预算三教师 Token-only、同预算三教师 Latent。只要原配方能成立，就保持这个目标。
- 原作者的训练文件、顺序和未公开的细节已经确认无法取得。按附件和可核验公开材料独立重建。请指出对分数和复现声明的具体限制。把与原文不同但可解释的决定标成独立复现偏差。
- 这是只读审查。可解包、核对哈希、阅读公开论文、执行隔离的 CPU 小测试和数据检查。代码候选只允许按照项目规则在 Docker 沙箱执行。数据批准、发布、修改仓库、启动 Colab、上传和对外联系由项目负责人另行决定。
- 三教师数据处于候选状态。先分清候选、已批准、实现、CPU 测试、真实模型 CPU 前向、真实 CUDA 验证、能力提升这几种证据。采用包内冻结代码身份，GitHub main 只能用于核对上轮双教师历史。不能把历史双教师结果、Replay 复建结果或 tiny Qwen2 的合成轨迹当作本次运行结果。
- 重点找能改变**是否上卡、是否继续跑、正式结果能否解释**的问题。既查现有实现的缺陷，也反过来查你提出的每道门槛是否真的必要。对性能优化，先确认它会保持相同训练语义和对照身份。

## 第一步 · 核对送审资产

1. 解压 ZIP，核对 `SHA256SUMS.txt` 并给出实际读过的文件列表。如果不能解包或执行命令，清楚写明哪些文件实际可读、哪些结论因此待证。分析可继续覆盖已读部分。
2. 读 `docs/reviews/latent_mopd_three_progress_20261115.md`、`AGENTS.md`、`docs/latent_mopd_three.md`，再追踪 `src/training/latent_mopd_three.py`、`scripts/prepare_latent_mopd_three.py`、`scripts/evaluate_latent_mopd_three_dev.py` 及其导入。论文至少核对原文 §3、§4.1、Table 1、Appendix A/C，以及 Replay 的 Joint MOPD、Appendix B/C、Table 8。
3. 为每个关键主张标记**源码已证实 / 附件实测 / 论文明确 / 可推断 / 缺证 / 错误**，给出准确的文件、函数、字段、样本 ID 或论文编号。只有读到实际行号时才引用行号。旧 6 Pro 报告审查过的 Gemini 附件与本次入口不是同一实现，逐项重查对当前代码的适用性。
4. 允许 benchmark 或代码缺失成为你结论的限制。外部来源标记版本与出处。减少转述摘要，多给反例、实验设计和判定。

## 第二步 · 从训练入口追一次完整链

### A. 模型、rollout 和路由

- 核查四个 revision、权重缓存目录、HF 元数据的大小/哈希、Nemotron 的两个权重分片和索引。验证模型、tokenizer 和真实 prompt 契约是否绑定。指出哪条证据只有逻辑教师短序列 CPU 加载，哪些跨四模型的真实 CUDA 行为尚无证据。
- 从 `main → read_schedule → run → HFRolloutEngine → update → optimizer.step` 跟踪具体一条 math、code、logic 样本。整理运行时学生和对应教师收到的张量、padding 方向、dtype、设备、position IDs、EOS、mask 与教师标识。证明相同 `input_ids/attention_mask` 真正传递到冻结教师。查 hidden target 是否可能来自错误域。检查教师 `eval/no_grad/requires_grad`、CUDA dropout 与运行 RNG。
- 核验 62×32×4、21/21/20、步骤 1–3、4–30、31–62 和无 shuffle 的真实行为。确认每步 128 条回答，且只进行一次 optimizer 更新。区分源码计数、自报的 `optimizer_updates: 1`、optimizer hook、真实 CUDA 状态计数这几种证据。
- 核对实际 R1 模板在 `enable_thinking=False` 下仍加 `<think>\n` 的现象。结合公开论文判断哪些差异已知、哪些无证据，给出最小 tokenizer 与 EOS 验证方案。检查 rollout 当前 policy、打分 policy、反传时 policy 与恢复策略的绑定。尤其检查模型模式变化及 autocast 是否会使概率口径变化。

### B. Token 目标、Latent 目标与梯度

- 逐项对照 Appendix A/C 的权重、reward、std、stop-gradient、masked mean 和归约公式。用一组可以手算的 logits 和 padding 示例，列出预期的候选、reward、loss 与重要梯度符号。指出代码实际算式与原文的任何差异。
- 审查 151,665 个合法 ID、完整 LM head 151,936 行、rollout mask 和 head 分母的组合。要求一个能测出真实尾部概率质量与梯度影响的低成本实验。说明何时属于 padding 实现细节，何时会破坏策略一致性。检验 logit cast、std 缩放、top-k 精度和数值稳定性。
- 用 `P0 P1 R0 R1 EOS PAD` 或同等微型序列，把预测哪一 token 的位置、选哪层 hidden、EOS 监督、左/右 padding 逐列列出。核对 `prediction_rows` 与 `state_rows` 是否在 Latent 路径也该使用预测位置。明确论文是否提供了解决这个索引语义的充分证据。核验 `hidden_states[-3:]` 与末层 RMSNorm、LM head 输入，教师是否真正 stop-gradient，三层损失的除数，以及 λ=1 的实际总权重。
- 用长短回答混合的固定 batch 手推 8 个逻辑 worker 的组内 packing、`n_microbatch/16` 与跨 worker 聚合。确认单卡的 `len(group)/128` 等价需要哪些前提。检查 gather/span、词表额外行、FP32/BF16 和 fused AdamW 在真实 CUDA 上会不会出现首步才触发的问题。
- Figure 5c 的三个初始 raw representation loss 可以给出量级诊断。结合论文对聚合口径和 on-policy 样本的说明，判断能否当作硬门槛。任何验证数值需区分合成模型与真实学生/教师。不要用其他论文中的大 λ、梯度范数平衡、五教师 TIES 或不同序列阶段替换目标。

### C. 数据冻结、隔离与偏差

- 用 manifest、实际 JSONL 和 schedule 独立核对 5,952×3、源版本与源 ID、前三个代码 shard、每行 prompt hash、token 长度、前 62 个块和候选拒载。至少抽查前两轮及第 61/62 步，核对每域所有正式样本来自对应 train。验证清单是否真正绑定 schedule 内容或仅由行顺序重建。
- 检查 `protected_rows / select_prompts / logic_holdout / approve / load_approved`，分开报告文件哈希、源快照真伪、逐行授权、逐域重复和保护集覆盖。检查本地原始缓存是否被固定上游哈希约束。区分精确归一化保护、近重复保护、未覆盖 benchmark 和语义变体。给一份最省算力的实际重叠审计，先覆盖 1,984 条正式训练样本。
- 比较公开的 DAPO `Answer:`、代码原题、Gym `<answer>` 和 2,048 token filter 与论文证据。评估少量 shard、先填满再哈希排序、Gym 任务分布、筛除计数及 source IDs 是否引入选择偏差。辨别训练提示词和 dev/test 各自的模板职责。
- 检查 Reasoning Gym 0.1.19 的冻结问题、答案、评分元数据与生成器不确定性。实际 `isomorphic_strings` 跨进程曾重建出不同题面，当前评分用封存行。找出独立逻辑 test 中参考候选自检得 0.05 的题，对可解性、评分是否需要例外及是否影响比较给证据。不要用人工凑齐原文 925 条的做法偷偷改变本地集合。
- 审查数据批准和审计证据可否抵御错误文件、无效阶段清单、顺序变化和归档恢复。修复建议须遵守仓库已有的数据准入规则。

### D. Colab、显存、时间和恢复

- 对照 `docs/latent_mopd_three.md`、`colab_bootstrap_v3.sh`、`colab_job_v3.py` 与实际 CLI 入口。梳理从审查代码快照、项目负责人批准发布、上传到 Colab、本地磁盘加载和 GPU 跑，直到 Drive 验哈希与 results 发布的完整链。哪些步骤已有命令，哪些只存在文档，哪些依赖路径、Drive 配额或缺少落盘证据？提供可直接核对参数的顺序。
- 分别检查加载、HF 生成、打分、反向/optimizer、保存/上传的峰值。重点分析 FP32 student 在 HF decode 内 `autocast(cache_enabled=False)` 时重复 cast 的带宽代价。说明如何用同一固定策略及 seed 验证更快路径的等效性，再决定是否做这类修复。核查 4 条并发、16K response、三位教师常驻、KV cache、logit 分块、全参数 Adam、BF16 激活、主机内存、G4 显存和运行时环境。不要拿上轮 LoRA G4 耗时或其他论文多卡吞吐直接报本轮 ETA。
- 检查 `save_checkpoint / restore_checkpoint / prepare_output / run_identity`、原子完成标记、编号选择、RNG、失败回退、未提交 rollout 重算。找出最新 checkpoint 损坏、训练日志比 checkpoint 更新、上传同时清理、Drive 与本地保留不一致、两组同跑空间不足时会怎样。尤其检查 `.checkpoint-incomplete-*` 被周期同步读取的可能性，以及 `--sync-every` 会不会重复扫描大量权重或在 GPU 空闲时拖住会话。
- 设计低 CU 的真实链验收。区分短序列六步 smoke、一个 128 response 的生产规模步、三域+混合路径补测、冷启动续训和完整 62 步。能合并的测试应合并。解释怎样防止小 smoke 看不出 16K 的 OOM、吞吐或损失规模问题。对每一阶段给明确停机、通过和推进条件，写出要看哪条日志、哪份 JSON 和哪种 checkpoint 证据。
- 给出耗时、CU、磁盘与 Drive 的计算式，明确哪些变量要在实测后填入。把上传/缓存准备中 GPU 空等的部分单列出来。若建议性能配置变化，说明是否改变 hard identity 或对照条件，是否因此应新开小试而不是续跑。

### E. 评测、因果比较与上线判据

- 逐项核对 `evaluate_latent_mopd_three_dev.py` 与 scorer 的输入、模型 checkpoint 绑定、冻结提示词、解码、生成完成条件、`</think>` 切分、Math-Verify 和保守判分、Gym verifier、代码 Docker 沙箱。问清无 `</think>`、超长、阻断题、partial credit、超时和异常各怎么进入分母。给出最便宜的同样本 Base 评测链验证。
- 将 AIME24、AIME25、Minerva、LCB-e、MBPP、MBPP+、GYM、BBH、MuSR 列成逐套对照。每套明确原文样本、解码温度、候选数量、token budget、评测版本、过滤和答案提取。检验当前可用资产，区分去污染输入和可执行评测。为正式报告提一个优先级清楚且资源合理的闭环。
- 检查 Base/Token-only/Latent 的实验对照只差 declared factor。需要同初始权重、固定相同题序、预算、seed 配对和评测协议；两组 on-policy 轨迹可自然不同。说明如何报告逐题配对差值和不确定性。别把 Total vs Base 当 Latent 增量，别用上轮 8K dev 或论文的分数给当前 G4 作通过线。若收益落在测量噪声内或不同域互有升降，要说明可以写怎样的研究表述。
- 明确 smoke 对模型能力能证明什么，什么时候才值得花完整两组 CU。对每个你提出的硬阈值说明证据或校准方法。允许结论是“仍缺真实测量”。无需承诺跑完一定增长。

## 最终报告格式

1. **先给放行决定表**。分别对 CPU 离线验证、付费短跑、生产规模探针、正式两组训练标明 `GO / 条件 GO / NO-GO / 缺证`。每项写出阻断原因、下一步、完成后应看到的证据。把“候选发布等了负责人批准”的程序条件和工程缺陷分开。
2. **问题按优先级组织**。P0 会导致结果不可解释、训练错误、数据违规、无法恢复或明显浪费 CU。P1 会显著影响质量、性能、可核验性。P2 属于展示或维护。问题条目用下表字段，确认问题与风险假设分开。可复现的缺陷提供 CPU 反例或最小探针。
3. **论文差异表**。至少包含采样与词表、最后三层与位置、微批归一化、数据/模板、精度与资源、九项评测。每条写明论文证据、当前源码位置、偏差影响与处理决定。
4. **按顺序的最小行动单**。先零 CU 修复/验证，再最低成本真实链，最后一组/两组完整运行和评分。给出合并工作、具体命令或必要参数、退出条件、产物证据、证明范围及是否需要另开运行身份。材料不足时列出缺失文件和待验证项，避免用假设补齐证据。
5. **反方复核与项目叙述**。指出最可能推翻你放行决定的三条证据，给出当前可写到项目经历里的事实与要等真实分数才能写的结论。

| ID / 优先级 | 确认状态及信心 | 文件/函数/数据行与证据 | 对数学、运行或结论的后果 | 最小修复 | 验证命令或可复现实验 | 通过/失败条件 |
| --- | --- | --- | --- | --- | --- | --- |

在报告末单列你真正运行的命令、使用的环境、产物和退出状态。只有实际运行过的结果才写“已通过”。给出明确的工程判断，避免把未知项包装成确定性。
