# 项目二 · 三教师 Latent-MOPD 独立复现进展与上卡前证据

日期 2026-11-15，新加坡时间。审查对象是**当前收录的本地工作区快照**。仓库基线为 `XDE10/qwen-dual-teacher-opd` 的 `main@4dee8bfbc554a22461d4d347f4973fc7fb7a016e`，该提交记录的是上一轮双教师结果。三教师训练、候选构建和新 dev 评测入口来自工作区，尚未进 GitHub main。包内 `SHA256SUMS.txt` 标识本次实际送审版本。

## 1. 当前状态与本次审查目标

三教师训练已实现，CPU 上验证了部分训练和恢复路径，真实 Nemotron 教师通过分片校验与短序列前向。三域候选和 62 步批次已冻结，等待项目负责人批准。**三教师方案还没有真实 GPU 训练、显存峰值、吞吐、CU 预算、checkpoint 恢复或模型涨幅结论。** 本次送审先判断实现和运行链是否具备短跑条件，再确定怎样用实际信号决定是否投入完整两组训练。

原作者的训练样本顺序、部分评测提示词和完整代码无法取得。本项目按公开配方**独立复现**，这是一项已经确定的项目约束。目标固定于 Latent-MOPD Table 1 的同家族三教师 last-3 指标，使用 Base R1、三教师 Token-only 和三教师 Latent 约束因果比较。Replay 论文只给交叉证据以及公开的提示词细节。历史双教师网络和本次源码都可以在同一仓库发现，请按明确入口区分运行身份。

## 2. 上一轮证据与本轮要回答的问题

上一轮正式 run 是 **数学和代码双教师、单卡 LoRA rank 16、v5 数据、数学代码各 31 次更新**。训练 62 步已完成。内部门槛用同一批 v5 dev、HF greedy、8,192 token 上限，并在 `</think>` 之后抽取答案。代码候选由 Docker 沙箱执行。

| 指标 | Base R1 | 双教师 Latent @62 | 变化 |
| --- | ---: | ---: | ---: |
| math_dev · Math-Verify | 170/256 · 66.41% | 162/256 · 63.28% | −3.13 pp |
| math_dev · 保守判分 | 136/256 · 53.13% | 129/256 · 50.39% | −2.73 pp |
| code_dev · 全量 | 70/200 · 35.00% | 78/200 · 39.00% | +4.00 pp |
| code_dev · 192 道可执行题 | 70/192 · 36.46% | 78/192 · 40.63% | +4.17 pp |
| 数学截断数 | 80 | 85 | +5 |
| 代码截断数 | 98 | 97 | −1 |

上轮没有匹配预算的正式 Token-only 对照。恢复时，第 22 步有一部分 rollout 复用缺少学生权重指纹，严格 on-policy 身份无法证明。以上结果能说明这份双教师配置在该 dev 协议下的变化，本轮需要另测三教师方法的总收益及 Latent 的增量。具体路径和版本可核对包内 `README.md`。 [双教师训练归档](https://github.com/XDE10/qwen-dual-teacher-opd/tree/3b6ea99b76cc84898f11988fe828457a302de9b3/latent_mopd_main/20261113T220228Z) · [双教师逐题评分归档](https://github.com/XDE10/qwen-dual-teacher-opd/tree/36312e2212cc66d7cf76be1887aaa686340a6f9a/code_scores/r1_dev62-20261114T104404Z)

原论文 Table 1 的 last-3 组相对 Base 的九项增益为 6.0 至 27.8 pp，相对 Token-only 为 0.6 至 4.8 pp。原文的 Norm 0／0.90／1.05 来自最佳教师的提升空间，不能当成百分点。Replay Table 8 的三教师联合参考中，九项平均是 **37.57／52.01／52.31**，分别对应 Base／Token-only／Latent。Latent 组高出 Base 14.74 pp，高出 Token-only 0.30 pp。后续论文重建了若干评测模板，联合组还采用多次评测。两篇论文的分数是参考，不是本地结果和硬性放行阈值。 [原论文 Table 1 与附录](https://arxiv.org/html/2610.02381v1) · [Replay Table 8 与附录 C](https://arxiv.org/html/2611.02746v2)

## 3. 三教师配置与论文对照

| 角色 | 固定 Hugging Face 模型 | revision |
| --- | --- | --- |
| Student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Math | `hbx/JustRL-DeepSeek-1.5B` | `0637e4096c789c67f9eecbe8355e0bdeddede1c2` |
| Code | `Fate-Zero/Archer2.0-Code-1.5B-Preview` | `db1e1ff9f3c36f7177f3896358ce26edeb62ce56` |
| Logic | `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`（ProRL v2） | `a8e647c7a46f04e84ce732d6fa3fa627ba343b8d` |

四个模型的普通词与特殊词 ID 映射共同覆盖连续的 0 至 151664，审计差异为 0。四个模型配置为 Qwen2、28 层、hidden size 1536。代码按固定 revision 获取模型配置与 tokenizer，权重复用 Drive 缓存或从 HF 下载后验哈希。逻辑权重包含两份 safetensors 及其索引。训练用的完整输入为学生模板的 prompt token 加学生生成的 response token，教师评分拿到相同的 `input_ids/attention_mask`。实际固定 R1 模板收到 `enable_thinking=False` 后仍以 `<think>\n` 结束 prompt，需要结合论文的模板说明审查。

| 因素 | 原文 Table 1 同家族 last-3 | 当前实现与审计边界 |
| --- | --- | --- |
| 优化预算 | 62 次更新，32 prompts × 4 responses，Math→Code→Logic，21/21/20 | 正式默认相同。合计 1,984 次 prompt 输入与 7,936 条回答 |
| 采样 | 温度 1.0，repetition penalty 1.0，响应最多 16,384 token | HF 自回归，top-p=1、top-k=0，最大 prompt 2,048 token 取自 Replay 公开重建细节。生成时屏蔽审计词表之外的 ID |
| 优化器 | AdamW，固定 1e-5，betas 0.9/0.999，decay 0.01，epsilon 1e-8 | 全参数 FP32 AdamW，CUDA fused，BF16 autocast，梯度非有限时失败。核对原文或参考实现对梯度裁剪、数值精度的证据 |
| Token 项 | 学生 top-16、候选内学生概率归一化、完整词表 log-prob、领域内标准差缩放 | 候选只从审计有效 ID 中选，log-prob 分母含整个 LM head，`reward_scale` 用全步候选奖励的样本 std，未减均值。审计有效词与额外 head 行的含义及梯度 |
| Latent 项 | 同宽 identity 对齐最后三层，所有有效回答位置，固定系数 1 | `hidden_states[-3:]` 使用最后两层中间状态和最终 norm 输出；位置由 shifted response mask 决定。需要核对原文 t 的语义及实际 Qwen2 张量 |
| Crossfade | 各教师的前 10 次更新上升/淡出 | 前 3 个全局步纯 Latent，第 4–30 步混合，第 31–62 步纯 token。Token-only 从第一步开始只用 token |
| 微批与归约 | 8 个 worker，每 worker 16 responses、每 worker 24,576 token，按响应数积累 | 单卡按顺序切 8 组，各组用 padded token 预算进行 packing，微批平均后乘 `len(group)/128`。本地复建的 packing/顺序须单独验证 |
| 分布式与训练后端 | 原文 8 worker，论文记录 vLLM、torch、transformers 环境 | 单卡 CUDA BF16、HF rollout、三位教师常驻，Torch 2.8.0/cu128 与 transformers 4.57.6 是项目的 Colab 入口版本。实际 CUDA 路径还未运行 |
| 测试协议 | 九项 Table 1 benchmark，各自固定提示词、解码预算和评分 | 目前只实现内部三域 dev。原文可直接核对的配置与独立重建细节分开审查 |

主训练链从 `src/training/latent_mopd_three.py:main` 进入。它审计 tokenizer 后读入发布的清单，重构轮转和 prompt IDs。每步用当前学生生成所有回答，再按该域选冻结教师。同一批输入完成无梯度候选打分、领域标准差估计和有梯度损失计算，随后逐微批反传并调用一次 optimizer。新实现不读取历史 rollout spool，恢复时会根据恢复的模型和 RNG 重新生成未提交的步骤。

关键交叉文件是 `src/losses/latent_opd.py` 的 `loss_weights`、`response_batch`、`hidden_alignment`、`reward_scale`，`src/losses/mopd_v3.py` 的 `prediction_rows`，`src/teachers/opd_contract.py` 的 `teacher_batch_from_student`。请以主入口实际调用链为准，旧双教师文件中的同名功能可能具有另一种数值行为。

## 4. 候选数据清单与隔离边界

候选所在目录为 `runs/latent_mopd_three/candidate/`。三个 `*_train.jsonl` 各 5,952 条，所有候选训练行的 `training_approved` 为 false，两份文件级批准列表都为空。每行存储固定的 prompt、源 ID、渲染长度与 prefix SHA-256。正式默认按文件内冻结顺序消费，训练只读取 prompt 字段。逻辑题的答案与元数据用于核验和判分。

| 域 | 原始来源与模板 | 池/顺序与构建观测 |
| --- | --- | --- |
| Math | 官方 DAPO-Math-17k parquet 固定源版本，保留源数据的 `Answer:` 指令 | 5,952 条，正式前 672 条。构建中遇到 250 条重复、35 条命中保护题 |
| Code | NVIDIA OpenCodeReasoning 固定源版本，选定 `split_0` 的前三个 parquet 文件，取 train 题面 | 5,952 条，正式前 672 条。构建中遇到 220 条重复、56 条命中保护题、6 条超长 |
| Logic | Reasoning Gym 0.1.19，独立选取 38 个任务，使用包内默认的 `<answer>` 用户指令 | 5,952 条，正式前 640 条。构建中遇到 199 条重复、14 条命中逻辑保护题。train/dev/test seed 分别为 42440021/42430021/4242 |

过滤计数指达到各域 5,952 条之前已扫描的部分。题池按 seed 42 和样本 ID 排序后冻结，训练期间不 shuffle。三域冻结 prompt 最大长度分别为 1,545、1,959、790 token。数学和代码保护集包括 v5 dev 456 题、MATH-500、AIME24/25、Minerva、HumanEval+/MBPP+、LiveCodeBench 源题面，实际文件与数量在 manifest。算法为规范化匹配、3-gram Jaccard、13-gram/LCS 和短串包含规则，逻辑题用规范化文本去重。保护集清单、扫描范围、同源改写和跨域近重复仍需审查。没有语义级去污染证明。旧 v5 数据仅提供 math/code dev 和保护输入。

逻辑 dev 为 200 条，逻辑 test 为 924 条。三组逻辑集合经规范化文本检查互不重叠。曾复现 `isomorphic_strings` 在新 Python 进程里相同 seed/索引生成不同实例，评分器已改用冻结题面、冻结答案与 verifier 元数据。用参考答案或生成器记录的有效候选做自检，dev 200/200 计满分，独立 test 923/924 计满分，1 条参考候选得到 0.05。需查明该条是否影响正式测试展示，独立 test 目前仅存档，未运行模型。924 条的冻结资源属于本项目重建，原文 GYM 为 925 条。

| 证据 | SHA-256 |
| --- | --- |
| `candidate_manifest.json` | `16c6927688988ede73ba2333ceab24b5ccfb7f00901caac9228ca120244c69a2` |
| `schedule.jsonl` | `cdec06eff12178f7a7bad046799b380f39544894371d417ac8883571a202ffd3` |
| `src/training/latent_mopd_three.py` | `70514ec0bf02f840cb7679d6c5e8b6a97ea96166db00af6a0d9f69431d9552de` |
| `scripts/prepare_latent_mopd_three.py` | `d0d309a1335b380497a5b7f5931c871f9791aabd85fc17f6275c11e872906bc5` |
| `scripts/evaluate_latent_mopd_three_dev.py` | `b5facd28be3913e2a1d4e110167dd5bf4f52368f34df8771c79828d7af906b30` |
| `tests/test_opd_v3.py` | `85469fa9183e3fada1c2040c0d4a078d32627cf1307cea255f38798eabc77a99` |

## 5. 已执行的验证及证明范围

| 检查 | 实际结果 | 可下的结论和边界 |
| --- | --- | --- |
| WSL2 CPU 测试 | `OMP_NUM_THREADS=2 ~/.venvs/opd/bin/python -m pytest -q`，201 passed、1 skipped | 覆盖现有回归及新增三域 schedule、worker 边界、Qwen2 最后 norm、真实 optimizer hook、冻结教师梯度。模型为 tiny Qwen2 |
| CPU 补充实验 | 手工调用当前 update，按数学→代码→逻辑走 6 次，每次构造 128 条短回答。optimizer hook 每轮增 1，权重变化、损失/梯度有限、教师无梯度 | 证明这套 unit/integration 链能执行更新。回答是构造的，三域路由由补充实验驱动，未穿过真实 GPU `run` |
| Tokenizer 审计 | 真实四个固定版本的映射相同，15 万余 ID 全集逐项比较 | 同 token 支持有实证。未测真实 CUDA rollout 时额外 head 行概率质量 |
| 真实逻辑教师 | 两份权重分片与索引通过 HF 元数据校验，加载 1,777,088,000 参数。CPU 用学生模板的 22 prompt tokens 和 2 response tokens 成功取三层 `[2,1536]`，LM head 为 `[2,151936]`，最后状态等于最终 norm 后输出，logits 有限 | 验证逻辑教师加载、同源前向与末层接口。真实 4 模型同时上卡、真实学生反传仍未实测 |
| 数据和审批门 | 三域 5952×3 行、全部 prefix hash、重复项、文件哈希、1,984 prompt 的 62 步索引与候选拒载已核验 | 证明包内候选身份与审批阻断。本轮没创建正式 release，未跑批准数据的真实训练 |
| 小模型保存和恢复 | 本地额外验证了全参数保存、校验、保留最近两个、恢复 student/optimizer | 真 GPU 的中断续跑、Drive 冷恢复与固定种子一致性暂无证据 |
| 仓库旧发布 | `tools/verify_project2_release_v5.py` 报告 `errors: []` | 旧数据 v5 的校验，独立于新候选 |
| 工程环境 | `tools/verify_repository.py` 的布局检查因三个既有未提交审查材料报错。WSL Docker 集成不可用，Windows Docker Desktop 的 Linux engine pipe 也未响应 | 全仓布局校验不能报通过。Docker 恢复前不能执行本轮代码评分。三教师 G4 实际消耗 0 CU |

短跑默认命令在 `docs/latent_mopd_three.md`。示例短跑预算是总 6 步、每步 1 题 × 4 答、1,024 新 token。它覆盖三位教师各一次纯 Latent 和一次混合更新。第二种命令使用正式身份与预算加 `--stop-at-step`，属于单独的性能验证。审查应判断是否先用第 1 步核对正式形状，更早发现开销问题，再决定预算内何时覆盖第 4–6 步的混合路径。改 `--prompts` 或 token cap 后是不同运行身份。短跑的模型分数没有验证最终涨幅的统计能力。

## 6. 当前可执行的评测与运行链

`scripts/evaluate_latent_mopd_three_dev.py` 接受固定 Base R1 或 full checkpoint 的 `student/`。输出口径为 HF greedy、默认 16,384 生成 token、统一学生 tokenizer，抽取 `</think>` 后的回答。三个学生使用同一 math_dev、code_dev、逻辑 dev。数学复用 Math-Verify 和保守判分；逻辑报满分率和 verifier 平均分；代码生成 `code_dev.candidates.jsonl` 后走 `scripts/score_code_local_v3.py` 的 Docker。逻辑和数学的可运行评分代码、本地冻结输入、GPU 未执行的生成路径与论文最终评测须分别判断。

原文九项是 AIME24、AIME25、Minerva、LiveCodeBench-easy、MBPP、MBPP+、GYM、BBH、MuSR。当前仅把若干公开测试集用于去污染，**整套论文评测器未接通**。6 Pro 应为短跑放行与复现成果验收分别设定最小评测和取证条件。配置选择使用 dev，正式 test 用于最终报告。基线分数必须在本轮协议重跑或用完全同协议可核查的产物，旧 8K 分数只作历史比较。

## 7. 上卡前需要审清的风险

以下是待核实项，不预设它们是已确认的损失错误。

1. **算法语义**。论文 normalized log-prob 与当前候选/生成支持的关系，额外 head 行进入分母的影响，Latent 对齐是预测回答的前一位置还是回答 token 自身，可用真实短序列对照。两个公开论文的微批、精度与 loss scaling 证据不能混写。
2. **GPU 性能**。`run` 的 FP32 学生用 `autocast(cache_enabled=False)` 进行 HF 自回归。要验证连续 decode 会不会反复转换线性层权重、引入巨大带宽开销。三位教师常驻、本步 Adam 状态、最长 batch KV 和打分分块的实际峰值都待测。建议在审查中区分分配量、保留量与整卡显存，拆分预填充、生成、打分、反向、optimizer 和 checkpoint 时间。此前双教师曾在生成阶段触及约 93.9 GiB，那个配置和本次配置各自独立。
3. **磁盘与恢复**。以 1.777B 等尺寸近似，FP32 学生参数与两份 Adam FP32 moments 一份完整 checkpoint 约 21.3 GB，未计配置和额外文件。生效的保留策略是 2 份完成态，落新文件期间还会有第 3 份。近期 checkpoint 采用临时目录后原子发布；旧的 `colab_job_v3.py` 按目录 `rclone copy`，未见专门排除 `.checkpoint-incomplete-*` 的选项，也无法从本地保留规则直接推断 Drive 远端上限。当前默认选择编号最高的 checkpoint 并验证，损坏时的降级策略需核实。
4. **数据证据**。候选使用公开源版本与冻结文件 hash，审查缓存源文件是否与 HF immutable blob 绑定、逻辑生成在跨环境如何重建、保护集覆盖和正式清单构建是否足够。沿用批准范围和 sandbox 规则。
5. **部署与对照**。三教师命令可在文档找到，但新的未提交源码、release、tokenizer 与模型缓存如何准确进入 Colab、同步到 Drive 和发布到 results 尚无真实证据。确认两组都从同一 Base 权重独立开始，唯一方法差异是损失及其调度。使用同一 dev，正式样本、步数、seed、解析、失败分母都要固定。

## 8. 提交给 6 Pro 的阅读顺序

1. `docs/reviews/latent_mopd_three_review_prompt_6pro_20261115.md` 给定审查任务。本文用于定位证据。
2. `AGENTS.md`、`docs/latent_mopd_three.md` 与三个新入口，先找真实执行与未接通的路径。
3. `runs/latent_mopd_three/candidate/` 的 manifest、schedule、train/dev/test 文件和 `runs/latent_mopd_three_assets/` 的模型配置与 tokenizer 审计。
4. 两篇主论文、LastOPD、上一份 6 Pro 审计，均在 `reports/argon_latent_mopd_audit_20261115/`。上一份审查是针对 Gemini 附件的历史证据，不是本次新训练器验收。
5. `tests/test_opd_v3.py` 中 `latent_mopd_three` 的测试，随后核对调用到的 runtime、rollout、loss、评测和 Colab/Docker 文件。

随附 ZIP 是本地审查快照，含原路径的源码、候选训练和逻辑 dev/test 文件、旧 v5 dev 依赖、模型配置/tokenizer、Nemotron 分片索引元数据、论文、两份 Markdown 和本次重跑的 CPU 验证输出。完整 P2 历史、原始大型 parquet、模型权重及凭据位于包外。整仓测试数字来自完整工作区。审查包可校验快照和运行指定的小测试，涉及省略数据/历史的测试仍需在完整工作区执行。从解压根目录可用以下命令核对。

```bash
sha256sum -c SHA256SUMS.txt
python -m pytest -q tests/test_opd_v3.py -k latent_mopd_three
```
