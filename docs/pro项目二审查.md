# GPT-6-Pro 全面技术审计与方案改进评审提示词 (Review & Audit Prompt)

> **文档定位**：本文件为提交给 GPT-6-Pro 的完整项目审计与评审提示词（Review Prompt）。它与代码仓库（包含全部源码、配置、数据发布清单与实验报告）一同交付给外部顶级后训练与在线策略蒸馏（OPD）专家 GPT-6-Pro，旨在对《数学与代码双教师在线策略蒸馏》项目进行地毯式的技术审计、错误排查、统计学重估与方案改进论证。

## GitHub 审查入口

- [项目仓库](https://github.com/XDE10/qwen-dual-teacher-opd) · [审查分支](https://github.com/XDE10/qwen-dual-teacher-opd/tree/codex/project2-review-20261112) · [固定代码提交 `cf22c1a`](https://github.com/XDE10/qwen-dual-teacher-opd/tree/cf22c1ab47369329b1aeb24f13817cb66352d46f)
- [OPD v3 训练与评测报告及机器可读矩阵](https://github.com/XDE10/qwen-dual-teacher-opd/tree/cf22c1ab47369329b1aeb24f13817cb66352d46f/reports/g4_opd_v3_experiments_20261112) · [固定结果快照 `dc3eee1`](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc3eee16afce6047bf1b9bd62893c1d82e1dc1f8)

请用有权限的 GitHub 连接读取私有仓库，先看固定提交中的 `AGENTS.md`、`README.md` 和实验报告，再核对下文提到的源码与结果。答复开头列出实际读到的提交、文件和日志。若连接不可用，请列出最少需要补充的文件，无需索要令牌。以下背景与假设均供核对。GPU 训练已暂停，本次请先审查并给出建议或补丁草案。

---

## 0. 评审角色与审计目标 (Role & Audit Objectives)

### 0.1 你的专家角色设定 (Your Persona)
你是全球顶尖的 LLM 后训练（Post-Training）、强化学习（RL/RLHF）与大模型在线知识蒸馏（Online Policy Distillation, OPD）首席科学家与系统架构专家。你具备极高的理论洞察力、严苛的代码审查素养与批判性科研思维，对伪科学、统计噪声、未对齐的基线、虚假提升以及工程隐患具有零容忍的审视标准。

### 0.2 审计核心目标 (Core Audit Objectives)
本项目旨在让 **1.5B 参数的学生模型** 在数学与代码两个专业领域，利用自身的在线生成轨迹（On-Policy Rollouts）接受冻结的高能力领域教师模型（7B / 14B）给出的 **Token 级反向 KL 散度（Reverse KL Divergence）** 监督更新，探究专职双教师的协同效益、跨域干扰/遗忘机制与对齐瓶颈。

你需要结合本提示词给出的详尽背景、实验数据与仓库源码，完成以下六大维度的深度审计：
1. **代码与工程实现缺陷审计**：审查损失函数、自动微分梯度推导、显存优化机制、Token 级对齐逻辑与微批次加权中的代码漏洞（Ranked Correctness Bugs）。
2. **理论机理与反常现象解释**：对实验中暴露的“专职数学教师在高学习率下引发长度爆炸与精度退化”、“近端教师混合（$\\alpha=0.7$）导致后期熵失控暴增”、“离线 SeqKD SFT 陷入停滞”等反常现象给出坚实的理论解释。
3. **统计有效性与评测可信度重估**：评估当前样本量（256/200/164）下的统计功效（Statistical Power）、最小可检测效应（MDE）与置信区间，识别过度声称（Overclaiming）风险。
4. **验证边界与未决实验甄别**：厘清已在 GPU 验证与仅完成 CPU 契约测试的严格界限，指出学术结论中的证据缺口。
5. **R1 / Latent-MOPD 方案审计**：根据源码与日志定位 16K 配置的首步瓶颈，区分高显存采样、训练变慢和会话中断。另行核对隐藏层损失尺度及其对实际更新的影响。
6. **算力预算约束下的实验优化路线**：在严格限制的单张 G4 96GB GPU 预算下，制定优先级最高、科学收益最大化的后续实验与消融方案。

---

## 1. 项目背景与 AGENTS.md 绝对硬约束 (Background & Inviolable Constraints)

在审查代码和提出修改建议时，必须严格遵守仓库的核心契约与硬约束（违背即视为错误结论）：

### 1.1 锁定的模型版本与身份 (Pinned Model Identities)
所有主线实验（`opd-v3` 与 `sft-v3`）的模型必须精确锁定至以下 Hugging Face Commit SHA，严禁随意替换模型或混用权重：
* **Student（学生模型）**：`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`（词表大小 `config_vocab_size = 151936`，分词器长度 `len_tokenizer = 151665`，基础词表 `151643`，`eos_token_id = 151645` 即 `<|im_end|>`，`pad_token_id = 151643` 即 `<|endoftext|>`）。
* **Math Specialist Teacher（数学专职教师）**：`Qwen/Qwen2.5-Math-7B-Instruct` @ `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`（注意：其 `chat_template` 默认内置数学系统提示，其哈希值不同于通用模型，因此必须强制采用单次渲染直传机制）。
* **Code Specialist Teacher（代码专职教师）**：`Qwen/Qwen2.5-Coder-7B-Instruct` @ `c03e6d358207e414f1eca0bb1891e29f1db0e242`。
* **General Teacher（通用教师，D3 对照组）**：`Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`。
* **Prior-Aligned Math Teacher（先验对齐 14B 数学教师，D4 Math14B）**：`Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`（详见 `reports/tokenizer_alignment_report_14b.json`）。
* **R1 / Latent-MOPD 探索实验模型（2026-11-12，`src/training/latent_opd.py`）**：
  * Student：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
  * Math Teacher：`hbx/JustRL-DeepSeek-1.5B` @ `0637e4096c789c67f9eecbe8355e0bdeddede1c2`
  * Code Teacher：`Fate-Zero/Archer2.0-Code-1.5B-Preview` @ `db1e1ff9f3c36f7177f3896358ce26edeb62ce56`
  * 三者架构严格同构（Qwen2 架构，28 层，隐藏维度 $d=1536$），由 `latent_opd.audit_inputs()` 动态核验。

### 1.2 严格核验的 151,665 共同词表支持 (Audited Common Vocabulary Support)
* 严禁用 `min(vocab_size)`（在 Qwen2.5 中为 151,936，包含 271 个未绑定的填充行）。
* 必须通过 `src/teachers/tokenizer_audit.py` 读取上界并验证连续性：共同有效支持集严格固定为 **151,665 个 ID（0～151664）**。
* 在生成阶段，通过 `CommonSupportMask` 将 151,665 之后的未绑定位置屏蔽（设置为 $-\infty$）；在训练计算 logits 时，矩阵乘法前直接切片投影矩阵权重 `head.weight[:151665]`，在其上做重归一化 `log_softmax`。
* 尾部被剔除质量（Removed Mass）严禁用 FP32 计算 $1.0 - \exp(\text{lse}_{\text{support}} - \text{lse}_{\text{full}})$（在小于 $10^{-8}$ 时会产生严重的浮点下溢抵消错误），必须在对数空间通过 `logaddexp` 计算：
  $$\text{extra} = \text{logsumexp}_{v \ge 151665}(s(v)), \quad \text{total} = \text{logaddexp}(\text{lse}_{\text{support}}, \text{extra}), \quad \text{mass} = \exp(\text{extra} - \text{total})$$

### 1.3 学生提示单次渲染与原样 Token 直传契约 (Single-Render Token Pass-Through)
* **硬性铁律**：所有 Prompt 的 Token IDs **只能由学生模板渲染一次**，随后原样（完整 `input_ids` 与 `attention_mask`）直接输入给教师模型！
* 严禁教师端重新套用自身的 Chat Template，严禁先 Decode 学生前缀再二次 Tokenize，严禁教师端引入任何专有系统提示或后缀！
* OPD v3 提示词与生成预算契约（`configs/project2_prompt_contract.json`）。R1 探索复用提示词与数据入口，其实际 rollout 上限由 R1 启动参数单独指定：
  * 上下文硬上限：4,096 tokens；
  * `math_system`：`"Please reason step by step, and put your final answer within \\boxed{}."`；
  * `code_system`：`null`（保持学生端默认模板）；
  * 生成长度上限：训练中数学 1,024 tokens、代码 1,536 tokens，采样温度 $T = 1.0$。

### 1.4 模型选择必须仅依据开发集 (Dev-Only Model Selection)
* 所有的超参数选择（如学习率筛选、Checkpoint 优选、消融判断）**只能依据开发集**（`math_dev` 256 题、`code_dev` 200 题）。
* 最终测试集（`MATH-500`、`GSM8K test`、官方 EvalPlus `HumanEval+`、`MBPP+`）仅用于最终产出报告，严禁利用测试集反向指导超参调优。
* 在线训练准确率（Online Rollout Accuracy）仅为训练题集上 $T=1.0$ 的带噪遥测指标，严禁当作模型评测成绩。

### 1.5 代码评测强制两级 Docker 沙箱 (Docker-Only Sandbox Execution)
* 代码评测严禁在宿主机直接以子进程执行。宿主机无可用 Docker 时必须立刻报错停机，严禁静默回退。
* 内部 `code_dev` 必须使用专用沙箱镜像（`sha256:3b0c23d0...`），并在容器内施加严格的 CPU（3s）、内存（256MB）、文件大小与无网络（`--network none`）限制。
* 官方代码测试集必须使用官方锁定的 `EvalPlus v0.3.1` 镜像（`sha256:79a6508d...`），保持与学术榜单环境一致。

### 1.6 数据逐行批准铁律 (Data Approval Protocol)
* 数据范围仅以 Release Manifest 中的 `approved_training_files` / `approved_evaluation_files` 以及逐行 `training_approved: true` 为准。
* 严禁通过直接手工篡改 JSONL 字段或批量重新计算 SHA-256 的方式绕过数据审计。

### 1.7 严格的已验证 vs. 未运行边界声明 (GPU-Verified vs. CPU-Only Boundary)
* 报告与文档中必须清晰割裂「代码已实现/CPU 契约测试通过」与「GPU 实际完成训练/评测」的边界，未运行的配置必须显式声明为未运行，严禁虚构实验成果。

---

## 2. 核心数学公式与算法工程实现 (Mathematical Formulations & Algorithmic Mechanics)

### 2.1 序列与响应槽位对齐机制 (`src/losses/mopd_v3.py`)
设输入序列由提示词 $x_{0:\ell_p-1}$ 与生成响应 $y_{0:\ell_r-1}$ 组成。在因果自回归模型中，索引 $pos + 1 = \ell_p + t$ 处的 Token $y_t$ 是由位置 $pos = \ell_p + t - 1$ 的隐藏状态预测产生的：
* `prediction_rows(response_mask)`：提取 `response_mask[:, 1:].nonzero()`，得到所有回答槽位对应的预测位置 $pos \in \{\ell_p - 1, \dots, \ell_p + \ell_r - 2\}$，对应的目标 Token 为 $y = \text{ids}[\text{batch\_idx}, pos + 1]$。
* `hidden_rows`：Base Transformer Decoder 前向传播后，**仅**在预测位置上切片收集隐藏状态（形状为 $N_{\text{resp}} \times d_{\text{model}}$），随后仅送入投影头前 151,665 维计算 Logits，提示词位置和 Padding 位置的 Logits 绝不参与计算，显著降低计算与显存开销。

### 2.2 目标分布与近端教师阻尼 (Proximal Target Formulation)
对于每个有效回答预测位置 $i$ 及词表 $V = \{0, \dots, 151664\}$，学生与教师的 Logits 分别转换为 FP32 概率分布：
$$\log p_i = \text{log\_softmax}(s_i), \quad \log q_i = \text{log\_softmax}(t_i)$$
为防止强教师对弱学生施加过于剧烈的惩罚，引入基于置信域（Trust-Region）思想的近端教师阻尼（Proximal Teacher Alpha，$\alpha \in (0, 1]$，参考 TROPD 与 NeMo RL）：
$$\log \tilde{q}_i(v) = \text{logaddexp}\big(\log q_i(v) + \log \alpha, \; \text{sg}[\log p_i(v)] + \log(1 - \alpha)\big)$$
当 $\alpha = 1.0$ 时，退化为原始教师目标 $\tilde{q}_i = q_i$；当 $\alpha < 1.0$（如 `d4_prox07` 中的 $\alpha=0.7$）时，目标分布在概率空间混合了停止梯度（Stop-Gradient）的学生自身分布。

### 2.3 三种核心损失函数解析推导 (`_forward_chunk` & `_grad_chunk`)
1. **全词表反向 KL 散度 (`exact_rkl`，项目主线默认)**：
   * **前向损失**：
     $$\mathcal{L}_i^{\text{exact}} = \sum_{v \in V} p_i(v) \big(\log p_i(v) - \log \tilde{q}_i(v)\big)$$
   * **解析梯度（关于学生 Logits $s_i(v)$）**：
     令 $r_i(v) = \log p_i(v) - \log \tilde{q}_i(v)$，$\bar{r}_i = \sum_{u \in V} p_i(u) r_i(u) = \mathcal{L}_i^{\text{exact}}$，则：
     $$\frac{\partial \mathcal{L}_i^{\text{exact}}}{\partial s_i(v)} = p_i(v) \big(r_i(v) - \bar{r}_i\big)$$
     *(性质：当 $p_i = \tilde{q}_i$ 时，$r_i(v) = 0 \implies \nabla_{s_i}\mathcal{L}_i^{\text{exact}} = 0$。因使用了显式解析梯度，反向传播无需保留完整的全词表计算图)*。
2. **Top-$K$ 反向 KL 散度 (`topk_rkl`，MOPD Eq. 5，带无偏最优修正)**：
   * 选取目标分布 $\tilde{q}_i$ 的 Top-$K$ 集合 $\mathcal{K}_i = \text{arg top-}K_{v \in V} \log \tilde{q}_i(v)$（$K=64$）：
     $$\mathcal{L}_i^{\text{topk}} = \sum_{v \in \mathcal{K}_i} \Big[ p_i(v)\big(\log p_i(v) - \log \tilde{q}_i(v)\big) - p_i(v) + \tilde{q}_i(v) \Big]$$
     末尾项 $-p_i(v) + \tilde{q}_i(v)$ 为 Bregman 散度修正项，确保在局部子集求和时，当 $p_i(v) = \tilde{q}_i(v)$ 时梯度依然精准为 0。
   * **解析梯度**：令 $w_i(u) = p_i(u)(\log p_i(u) - \log \tilde{q}_i(u))$，则：
     $$\frac{\partial \mathcal{L}_i^{\text{topk}}}{\partial s_i(v)} = \mathbb{I}[v \in \mathcal{K}_i] w_i(v) - p_i(v) \sum_{u \in \mathcal{K}_i} w_i(u)$$
3. **Token 级策略梯度 (`pg`，MOPD Eq. 4，带截断优势函数)**：
   * 对学生在线采样的真实 Token $y_i \in V$，计算截断优势值（Advantage Clip $c = 10.0$）：
     $$A_i = \text{sg}\Big[ \text{clamp}\big(\log \tilde{q}_i(y_i) - \log p_i(y_i), \; -c, \; +c\big) \Big], \quad \mathcal{L}_i^{\text{pg}} = -A_i \log p_i(y_i)$$
   * **解析梯度**：$\frac{\partial \mathcal{L}_i^{\text{pg}}}{\partial s_i(v)} = A_i \big(p_i(v) - \mathbb{I}[v = y_i]\big)$。

### 2.4 序列与领域归一化加权 (`sequence_weights` in `src/training/opd_v3.py`)
每个批次包含多个领域 $\mathcal{D}$，每个领域包含 $S_d$ 条生成序列，序列 $j$ 的响应长度为 $\ell_{d,j}$，总生成序列数 $S = \sum S_d$：
* **`sequence_mean`（MOPD 默认，即 $1/|y|$ 归一化）**：
  $$w_{d,j} = \frac{1}{S \cdot \ell_{d,j}} \implies \mathcal{L}_{\text{step}} = \frac{1}{S} \sum_{d \in \mathcal{D}} \sum_{j=1}^{S_d} \frac{1}{\ell_{d,j}} \sum_{t=1}^{\ell_{d,j}} \mathcal{L}_{d,j,t}$$
* **`token_mean`**：每 Token 权重均等 $w_{d,j} = 1 / \sum_{d,j} \ell_{d,j}$。
* **`domain_token_mean`**：领域内 Token 均等加权后按领域权重平衡。

### 2.5 关键系统与工程机制
1. **`GenerationCopy` 显存外 FP32 动态合并**：在 BF16 精度下，若反复调用 PEFT 的原生 `merge_adapter()` 和 `unmerge_adapter()`，会导致 $\text{bf16}(\text{bf16}(W + \Delta W) - \Delta W) \neq W$ 的渐进式数值漂移。`GenerationCopy` 维护一份不可导的 Base 模型深拷贝，每个 Rollout 阶段前在 FP32 空间完成权重合并并转为 BF16，彻底消除底模漂移。
2. **`sampling_v2` 动态审计链**：通过 Monkey-patch 劫持 `model._get_logits_processor`，强制剥离 Qwen2.5 官方自带的 `repetition_penalty = 1.05~1.1`，确保采样仅受 `CommonSupportMask` 约束。
3. **`RowDivergence` 分块 Autograd**：对微批次内包含数万 Token 的张量，按 `row_chunk = 64` 在前向与反向中分块重算 FP32 Log-Softmax，避免一次性在显存中保留数十 GiB 的 FP32 Logits 矩阵。
4. **梯度投影手术 (PCGrad, `src/training/opd_v3.py`)**：在单步内按顺序计算数学与代码反向传播，当且仅当数学梯度与代码梯度夹角为钝角（$g_{\text{math}} \cdot g_{\text{code}} < 0$）时，将各自梯度正交投影到对方的法平面上：
   $$g'_{\text{math}} = g_{\text{math}} - \frac{g_{\text{math}} \cdot g_{\text{code}}}{\|g_{\text{code}}\|^2} g_{\text{code}}, \quad g'_{\text{code}} = g_{\text{code}} - \frac{g_{\text{math}} \cdot g_{\text{code}}}{\|g_{\text{math}}\|^2} g_{\text{math}}, \quad g_{\text{final}} = g'_{\text{math}} + g'_{\text{code}}$$
5. **软硬运行身份与密封 Checkpoint (`src/training/runtime_v3.py`)**：
   * **硬身份**：配置参数、代码源文件哈希、分词器哈希、提示词契约哈希与关键依赖版本（torch/transformers/peft）。任何一项不匹配则严禁 Resume，必须完全重新开始。
   * **完整性校验**：加载 Checkpoint 前验证 `checkpoint_manifest.json` 声明的文件及其 SHA-256。请同时核查来源信任边界与反序列化安全，文件哈希只证明文件与清单一致。

### 2.6 Latent-MOPD 实验机制 (`src/losses/latent_opd.py`)

[论文附录 A、C](https://arxiv.org/html/2610.02381v1) 是对照依据。当前实现为单卡 LoRA、数学与代码双域、使用项目 v5 数据。论文主实验有逻辑教师、不同数据和 8 个训练 worker。
2026-11-12 研发的 R1-Distill-1.5B 探索实现：
1. **单步单域严格交替**：从日志第 1 步的数学开始，随后代码，计划共 62 步（各域 31 步）。每步 32 题 × 4 个回答，生成上限 16,384 token，温度 1.0，AdamW 固定学习率 `1e-5`。
2. **权重衰减调度 (`loss_weights(variant, domain, u)`)**：
   * `variant == 'token'`: $(w_{\text{tok}}, w_{\text{lat}}) = (1.0, 0.0)$ 恒定。
   * `variant == 'latent'`: $w_{\text{tok}}(u) = \min(1.0, u / 10)$, $w_{\text{lat}}(u) = \max(0.0, 1.0 - u / 10)$ 对双域均线性过渡 10 步。
   * `variant == 'math20'`: 只将数学隐藏层的淡出延长至 20 步；数学 token 权重仍用原来的 10 步升权，代码调度沿用 `latent`。
3. **Top-16 候选打分与样本标准差缩放**：学生选自身概率最高的 16 个 token，候选权重 $\tilde{p}_{i,k}$ 在这 16 项内重归一化。计算奖励 $R_{i,k} = \tilde{p}_{i,k}(\log q_i(c_k) - \log p_i(c_k))$ 时，师生的 log 概率均在完整有效词表上归一化。将整步该域的有效奖励合并计算样本标准差，除以 $\max(\sigma_R,10^{-6})$，保留奖励均值，不做减均值操作；优势停止梯度。
4. **最后 3 层隐藏层表征对齐 (`hidden_alignment`)**：提取 Transformer 最后 3 层的隐状态 $H_S, H_T \in \mathbb{R}^{3 \times N_m \times d}$（$d=1536$），做 $L_2$ 单位化后计算均方误差对齐损失：
   $$\mathcal{L}_{\text{lat}} = \frac{1}{N_m} \sum_{i=1}^{N_m} \frac{1}{3 d} \sum_{\ell=1}^3 \left\| \frac{h_{\ell,i}^S}{\|h_{\ell,i}^S\|_2} - \text{sg}\left[\frac{h_{\ell,i}^T}{\|h_{\ell,i}^T\|_2}\right] \right\|_2^2$$

---

## 3. 数据集发布 (v5) 与评测协议 (v3) 详解 (Data Release v5 & Evaluation Protocol v3)

### 3.1 数据集发布版本演变与 v5 构成
从 `portable_release` (2026-10-24) 到 `v4` (2026-10-25)，再到当前主线唯一绑定的 **`v5` (2026-11-08 批准)**：
* **`project2_math_train.jsonl` (2,500 题)**：1,500 题来自 MATH-lighteval，1,000 题来自 GSM8K。相比 v4，剔除了 133 道与评测集重合的污染题目，并从原始题库按相同 Level 难度严格分层等量补齐（Level 1: 2, Level 2: 25, Level 3: 27, Level 4: 35, Level 5: 43, GSM8K: 1）。
* **`project2_code_train.jsonl` (2,488 题)**：包含 1,247 道 AceCode 函数题与 1,241 道 open-r1 标准输入输出（stdin_stdout）竞赛题。相比 v4 剔除了 12 道重合题，因原始代码池在仓库重排中清理，未做补齐。
* **开发集 (Dev Sets，严格锁定，禁止参与训练)**：
  * `project2_math_dev.jsonl`：256 题。
  * `project2_code_dev.jsonl`：200 题（100 函数题 + 100 stdin 题）。
* **训练答案旁路文件 (`project2_math_train_answers.jsonl`)**：仅包含 2,500 道数学训练题的参考答案，仅用于在线生成轨迹的实时解题正确率监控，绝不参与训练 Prompt 拼接。

### 3.2 四重去污染规则 (Decontamination v5 Rules)
对所有训练题执行 NFKC 归一化、大小写折叠与词符提取后，与保护集（包括 MATH-500、GSM8K test、MATH 5000 测试题、Dev 集、HumanEval+、MBPP+）进行四级比对：
1. `normalized_exact`：归一化词序列完全一致；
2. `jaccard_3gram`：3-gram Jaccard 相似度 $\ge 0.85$；
3. `ngram13_lcs`：存在共享 13-gram 且最长公共子序列覆盖率 $\ge 60\%$；
4. `short_containment`：短文本（$<13$ 词）包含且占训练样本长度 $\ge 50\%$。
* 代码题预先剥离模板样板句，并对 AST 做抽象语法树 SHA-256 精确比对。

### 3.3 评测协议 v3 与双判分器体系
* **解码配置**：全部评估采用 Greedy 解码（`temperature=0.0`，`do_sample=False`），强制设置 `use_model_defaults=False` 并剔除 `repetition_penalty`，使用 `CommonSupportMask(151665)`。
* **生成上限**：数学评测固定 `max_new_tokens = 2048`；内部代码评测固定 `1536`；EvalPlus 评测固定 `1024/1536`。
* **数学双判分器**：
  * **主口径 (`Math-Verify`)**：基于官方 `math_verify` 库解析 LaTeX Boxed 答案并进行符号等价性验证。
  * **副口径 (`conservative_check`)**：仓库自研的有界 AST + SymPy 确定性比较器（零 `eval`，零 `sympify`）。**注意**：保守判分器对 `MATH-500` 官方金标参考答案的自判仅能识别 **389/500 (77.80%)**，对 `math_dev` 参考答案仅能识别 81.64%（无法处理隐式乘法、多项式元组、矩阵等复杂格式）。
* **内部 `code_dev` 200 题裁判机制与 8 题人工阻断**：
  * 经全面审查，200 题中 **192 题处于 `reviewed` 状态**，具体比较器类别如下：
    * `83` 项 `exact`（递归类型严格相等，bool 不等于 int/float，字典键无序）；
    * `90` 项 `stdout_tokens_exact`（空格切分后逐 Token 匹配）；
    * `5` 项 `stdout_tokens_casefold`（忽略大小写的标准输出匹配）；
    * `4` 项 `unordered`（语义多重集匹配）；
    * `4` 项 `constructive`（具有题目特定约束检查的构造题：`code/vfc_134964`, `code/vfc_19131`, `code/vfc_18674`, `code/vfc_17062`）；
    * `2` 项 `float`（相对/绝对误差容忍）；
    * `2` 项 `tree`；
    * `1` 项 `maximum_geometry`；
    * `1` 项 `stdout_lines_exact`。
  * **8 题被标记为 `blocked_after_review`**（题目存在歧义或测试集缺失：`ace/oss_11659`, `ace/oss_618`, `ace/oss_11985`, `ace/oss_16281`, `ace/oss_13256`, `ace/oss_17515`, `ace/oss_18226`, `ace/oss_10109`）。
  * **分母完整性**：汇报 `code_dev` 时同时汇报全量分母 `passed / 200`（理论上限 96.0%）与剔除阻断后的分母 `passed / 192`，绝不静默丢弃未通过样本。
* **两级 Docker 沙箱机制**：
  * 内部沙箱镜像（`code_dev`）：`sha256:3b0c23d0...`，参数 `--network none --read-only --cap-drop ALL --memory 512m --cpus 1`，限制进程 CPU 3s、内存 256MB。
  * 官方 EvalPlus 沙箱：`sha256:79a6508d...`，参数 `--network none --memory 6g --cpus 2`，运行官方 `evalplus==0.3.1`。

---

## 4. 历次实验编年史与完整指标矩阵 (Chronological Experiments & Quantitative Metrics)

以下 OPD v3 主线实跑使用单张 Colab G4，评测使用 GPU 生成与本地 WSL2 Docker 打分。R1 探索的 L4 短跑和 G4 中断记录见 §6。

### 4.1 实验进展编年史 (Sessions 1–6 + R1 Latent-MOPD)
1. **Session 1 (2026-11-09, `56dd839`)**：
   * 完成 2 步混合 Smoke 验证，显存占用 35.45 GiB，单步耗时 ~2.1s。
   * 运行教师适配度探针（`probe_teacher_fit_v3.py`）：测定未训练学生（熵 0.317/0.588）与各候选教师的初始逆向 KL 散度。发现 `Qwen2.5-Math-7B-Instruct` 的初始数学 KL 高达 **`0.369`**（教师自身熵高达 1.263），远高于通用教师 `7B-Instruct`（KL 0.154）与 14B 教师（KL 0.170），预示其推理先验存在剧烈偏离。
2. **Session 2 (2026-11-09, `bc02083`)**：
   * 执行 30 步学习率扫描（`1e-5`, `3e-5`, `1e-4`）。
   * 测定第 21–30 步窗口的配对 McNemar 检验（$n=640$）：`1e-4` 在线数学准确率达到 68.8%，显著优于 `1e-5` (64.1%, $p=0.009$)。
   * 发现数学 KL 极高值主要集中在换行符、空格及 `<|im_end|>` 等格式 Token 上；推动将绝对学生熵约束放宽为师生相对熵约束（Gate v1 $\to$ Gate v2）。
3. **Session 3 (2026-11-10, `bc02083`)**：
   * 完成 D0 基线评测；产出 4,800 条教师有效输出轨迹。
   * 训练离线序列蒸馏基线 `sft_v3_seqkd_mix`（75 步），发现离线 SFT 在开发集上陷入停滞（`math_dev` 58.59% vs 58.20%，`code_dev` 38.50% vs 39.00%）。
   * 将 `d4_dual` (`3e-5`) 从第 30 步断点无损续训至 80 步。
   * **Dev-30 学习率重大决策**：在第 30 步开发集上，`d4_dual_lr1e-4@30` 的 `math_dev` 达到 **61.33%**，大幅超越 `3e-5` 的 **56.25%**（增益 +5.08 pp，多解出 13 题），项目负责人正式决定将主学习率确立为 **`1e-4`**。
4. **Session 4 (2026-11-10, `bc02083`)**：
   * 将 `d4_dual_lr1e-4` 续训至 80 步：代码能力持续提升（`code_dev` 42.50%，`MBPP+` 57.41%），**但数学在 30 步后发生明显退化**（`math_dev` 从 61.33% 跌至 59.77%，`MATH-500` 从 56.60% 跌至 54.80%，低于 D0 基线；截断数暴增至 41 题；Gate v2 报警）。
   * 训练通用单教师对照组 `d3_general_lr1e-4`（80 步）：数学表现优异（`MATH-500` **58.80%**，`GSM8K` 77.26%），但代码显著受抑（`HumanEval+` 仅 50.61%，`code_dev` 40.00%）。
5. **Session 5 (2026-11-11, `06e52c3`)**：
   * 引入近端教师阻尼配置 `d4_prox07_lr1e-4`（$\alpha=0.7$），旨在平抑 `Math-7B` 的剧烈冲击。
   * 观测到 `code_dev` **43.50%** @50，`HumanEval+` **57.32%** @80，后者较 D0 高 10.37 个百分点。两个分数来自不同 checkpoint。
   * **严重副作用**：在 55 步之后，学生分布发生严重的无序熵膨胀（数学熵从 0.317 暴增至 3.19，增加 10 倍；代码熵从 0.588 增至 2.41；EOS 发生率骤降至 75.5%；数学题大量超长截断）。
6. **Session 6 (2026-11-11, `ae43794`)**：
   * 引入同构先验的大容量数学教师 `Qwen2.5-14B-Instruct` 搭配专职代码教师 `Coder-7B`（`d4_math14b_lr1e-4`）。
   * **全项目唯一以零失败、零警告通过 Gate v2 稳定气质检的 `1e-4` 模型**（显存占用 70.39 GiB，中位耗时 96.6s/步）。
   * `@50` 的六套评测点估计均高于 D0，其中 `GSM8K` 为 **78.09%**。`@80` 的双域开发集均值为 **51.22%**，`MATH-500` 为 56.80%，`HumanEval+` 为 55.49%，`MBPP+` 为 55.03%（D0 为 55.29%）。请核对逐题差异、截断和选择口径。
7. **R1 / Latent-MOPD 探索 (2026-11-12, `src/training/latent_opd.py`)**：
   * 采用 `DeepSeek-R1-Distill-Qwen-1.5B` 学生，基于 Top-16 策略梯度与最后 3 层隐藏层 MSE 蒸馏。
   * L4 的 128-token smoke 使用旧全词表 reverse KL，覆盖基础训练和续跑；当前 top-16 实现通过 CPU 损失与梯度对照。
   * G4 上用 16,384 token 上限尝试纯 token 组。32 并发时采样到 **96,139 MiB（93.9 GiB）**。后续 16 并发也没有留下可续训 checkpoint，停止原因与证据边界见 §6。

---

### 4.2 训练遥测与稳定性门禁矩阵 (Training Telemetry & Gate v2 Audit Table)

> **审计核对提示 (Audit Discrepancy Note)**：在 `reports/g4_opd_v3_experiments_20261112/README.md` 的 Table 2 中，`d3_general_lr1e-4` 与 `d4_prox07_lr1e-4` 记录的初始 KL 来自 `gate_v1_first_window`（前 10 步均值，分别为 0.1604 与 0.2185），而其余项来自 `gate` v2（Step 1 基准点）。下表完整展示 Step 1 基线与末 10 步均值：

| 模型运行 ID | 教师配置 (Math + Code) | 学习率 | 步数 (续训) | 中位步耗 | 显存峰值 | 数学 KL (Step 1 $\to$ 末 10 步) | 代码 KL (Step 1 $\to$ 末 10 步) | 数学相对熵比 ($H_S/H_T$) 末窗 vs Step 1 | 代码相对熵比 ($H_S/H_T$) 末窗 vs Step 1 | 数学学生绝对熵变化 ($H_{S,\text{last}} / H_{S,1}$) | 代码学生绝对熵变化 ($H_{S,\text{last}} / H_{S,1}$) | 数学 EOS 下降量 (Step 1 - 末窗) | 数学在线准确率 (首 10 步 $\to$ 末 10 步) | Gate v2 稳定性裁定 (`pass`, `failed`, `warn`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`sft_v3_seqkd`** | `Math-7B` + `Coder-7B` (离线 4800 轨迹) | `1e-4` | 75 (—) | `5.4s` | 10.9 GiB | Loss: 0.205 $\to$ 0.170 | — | — | — | — | — | — | — | N/A (离线 SFT 基线) |
| **`d4_dual`** | `Math-7B` + `Coder-7B` | `3e-5` | 80 (从 30) | `95.3s` | 55.5 GiB | 0.369 (0.338) $\to$ 0.207 | 0.223 (0.228) $\to$ 0.180 | 0.747 | 0.821 | 0.314 (0.317 $\to$ 0.100) | 0.549 (0.588 $\to$ 0.322) | 0.0406 (0.969 $\to$ 0.928) | 57.81% $\to$ 67.03% | **`pass=True`** (0 失败; warn: 数学学生绝对熵低) |
| **`d3_general_lr1e-4`** | `7B-Instruct` (单教师覆盖双域) | `1e-4` | 80 (—) | `91.7s` | 41.2 GiB | 0.154 (0.160) $\to$ 0.088 | 0.297 (0.249) $\to$ 0.191 | 0.713 | **`0.694` (FAIL)** | 0.395 (0.317 $\to$ 0.125) | 0.465 (0.588 $\to$ 0.273) | 0.0500 (0.969 $\to$ 0.919) | 65.62% $\to$ 67.97% | **`pass=False`** (失败: `code.entropy_vs_teacher`; 2 警告) |
| **`d4_dual_lr1e-4`** | `Math-7B` + `Coder-7B` | `1e-4` | 80 (从 30) | `95.5s` | 55.8 GiB | 0.369 (0.289) $\to$ 0.217 | 0.223 (0.221) $\to$ 0.170 | **`0.670` (FAIL)** | 0.828 | 0.328 (0.317 $\to$ 0.104) | 0.588 (0.588 $\to$ 0.346) | 0.0484 (0.969 $\to$ 0.920) | 62.50% $\to$ 65.62% | **`pass=False`** (失败: `math.entropy_vs_teacher`; 1 警告) |
| **`d4_prox07_lr1e-4`** | `Math-7B` + `Coder-7B` ($\alpha=0.7$) | `1e-4` | 80 (连续跑完) | `100.9s` | 59.2 GiB | 0.369 (0.219) $\to$ 0.266 | 0.223 (0.212) $\to$ **`0.332` (FAIL)** | **`3.820` (FAIL)** | 1.051 | **`10.067`** (0.317 $\to$ 3.193) | **`4.102`** (0.588 $\to$ 2.410) | **`0.2141` (FAIL)** (0.969 $\to$ 0.755) | 63.75% $\to$ 59.38% | **`pass=False`** (失败: 代码 KL、数学熵比、EOS 下降; 严重熵爆炸) |
| **`d4_math14b_lr1e-4`** | **`14B-Inst` + `Coder-7B`** | `1e-4` | 80 (—) | `96.6s` | 70.4 GiB | **`0.170` (0.131) $\to$ `0.088`** | **`0.223` (0.221) $\to$ `0.166`** | **`0.865`** | **`0.818`** | **`0.525`** (0.317 $\to$ 0.167) | **`0.575`** (0.588 $\to$ 0.338) | **`0.0422`** (0.969 $\to$ 0.927) | **`64.38%` $\to$ `67.50%`** | **`pass=True` (0 失败，0 警告，极度稳定)** |

---

### 4.3 开发集与测试集完整评测矩阵 (Complete Benchmark Evaluation Matrix)

* **评测标准**：全部模型均为 Greedy 解码，使用 `CommonSupportMask(151665)`，数学评测上限 2,048 tokens，内部代码评测上限 1,536 tokens，EvalPlus 上限 1,024/1,536 tokens。
* **分项说明**：`math_dev` (256)、`code_dev` (200，含 8 题人工阻断)、`MATH-500` (500)、`GSM8K test` (1319)、`HumanEval` (164)、`MBPP` (378)。

| 模型 / 评估检查点 | `math_dev` MV / 保守 (256) | `math_dev` 截断数 | `code_dev` 全量 / 剔除阻断 (200 / 192) | `code_dev` 函数 / Stdin (100 / 100) | 双域开发集均值 (MV + Code) | `MATH-500` MV / 保守 (500) | `MATH-500` 截断数 | `GSM8K test` MV / 保守 (1319) | `GSM8K` 截断数 | `HumanEval` Base / Plus (164) | `MBPP` Base / Plus (378) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D0 (1.5B 初始基线)** | 58.20% (149) / 46.88% (120) | 17 | 39.00% (78) / 40.62% | 65 / 13 | 48.60% | 55.00% (275) / 44.40% (222) | 25 | 73.92% (975) / 70.13% (925) | 5 | 51.22% (84) / 46.95% (77) | 64.55% (244) / 55.29% (209) |
| **SFT (`seqkd_mix` @ 75)** | 58.59% (150) / 49.22% (126) | 17 | 38.50% (77) / 40.10% | 67 / 10 | 48.55% | *(仅测开发集)* | — | *(仅测开发集)* | — | *(仅测开发集)* | *(仅测开发集)* |
| **`d4_dual_lr1e-5` @ 30** | 60.16% (154) / 49.61% (127) | 16 | 40.00% (80) / 41.67% | 69 / 11 | 50.08% | *(仅测开发集)* | — | *(仅测开发集)* | — | *(仅测开发集)* | *(仅测开发集)* |
| **`d4_dual` (`3e-5`) @ 30** | 56.25% (144) / 46.88% (120) | 18 | 42.00% (84) / 43.75% | 74 / 10 | 49.12% | *(仅测开发集)* | — | *(仅测开发集)* | — | *(仅测开发集)* | *(仅测开发集)* |
| **`d4_dual` (`3e-5`) @ 80** | 59.77% (153) / 50.39% (129) | 17 | 41.00% (82) / 42.71% | 74 / 8 | 50.38% | 58.00% (290) / 45.40% (227) | 34 | 74.60% (984) / 74.53% (983) | 6 | 59.76% (98) / 54.88% (90) | 65.34% (247) / 56.61% (214) |
| **`d3_general_lr1e-4` @ 80** | **61.33% (157)** / 50.00% (128) | 20 | 40.00% (80) / 41.67% | 71 / 9 | 50.66% | **58.80% (294)** / **46.80% (234)** | 28 | 77.26% (1019) / 77.03% (1016) | **2** | 54.27% (89) / 50.61% (83) | **67.72% (256)** / **59.52% (225)** |
| **`d4_dual_lr1e-4` @ 30** | **61.33% (157)** / **51.95% (133)** | 21 | 41.00% (82) / 42.71% | 75 / 7 | 51.16% | 56.60% (283) / 44.20% (221) | 34 | 75.13% (991) / 75.06% (990) | 7 | 60.98% (100) / 56.10% (92) | 65.34% (247) / 55.82% (211) |
| **`d4_dual_lr1e-4` @ 80** | 59.77% (153) / 49.61% (127) | 25 | 42.50% (85) / 44.27% | **76** / 9 | 51.13% | 54.80% (274) / 44.00% (220) | 41 | 74.83% (987) / 74.75% (986) | 6 | 60.37% (99) / 55.49% (91) | 65.61% (248) / 57.41% (217) |
| **`d4_prox07_lr1e-4` @ 50** | 59.38% (152) / 48.83% (125) | 34 | **43.50% (87)** / **45.31%** | 74 / **13** | **51.44%** | 55.00% (275) / 43.00% (215) | **51** | 74.22% (979) / 74.07% (977) | 7 | 62.20% (102) / 56.71% (93) | 63.49% (240) / 54.76% (207) |
| **`d4_prox07_lr1e-4` @ 80** | 55.47% (142) / 46.48% (119) | 34 | 43.00% (86) / 44.79% | 74 / 12 | 49.23% | 55.20% (276) / 43.40% (217) | 45 | 73.77% (973) / 73.69% (972) | 10 | **62.80% (103)** / **57.32% (94)** | 63.76% (241) / 54.23% (205) |
| **🔥 `d4_math14b_lr1e-4` @ 50** | 58.59% (150) / 48.44% (124) | 27 | 43.00% (86) / 44.79% | 75 / 11 | 50.80% | 56.80% (284) / 45.00% (225) | 28 | **78.09% (1030)** / **77.94% (1028)** | 6 | 61.59% (101) / 54.88% (90) | 65.08% (246) / 55.82% (211) |
| **🔥 `d4_math14b_lr1e-4` @ 80** | 60.94% (156) / 49.61% (127) | 20 | 41.50% (83) / 43.23% | 72 / 11 | **51.22%** | 56.80% (284) / 46.20% (231) | 37 | 77.26% (1019) / 77.10% (1017) | 5 | 60.37% (99) / 55.49% (91) | 64.55% (244) / 55.03% (208) |

---

## 5. 严格验证边界：GPU 已实测 vs. 纯 CPU 契约测试 (GPU-Verified vs. CPU-Only Boundaries)

为保证科研诚信与结论严谨，仓库严格区分以下两类配置：

### 5.1 已在 G4 GPU 与本地 Docker 完成验收的配置 (共 9 项)
1. **D0 基线**：未训练学生 `Qwen2.5-1.5B-Instruct`（全套 6 项评测完成）；
2. **SFT 离线蒸馏**：`sft_v3_seqkd_mix`（75 步训练 + 开发集双评测完成）；
3. **Smoke & Probe**：`smoke`（2 步测试）+ 教师适配度探针；
4. **D4 专职双教师 (`Math-7B` + `Coder-7B`)**：
   * `d4_dual_lr1e-5`（30 步）；
   * `d4_dual` (`3e-5`，30 步 + 续训至 80 步 + 全套评测）；
   * `d4_dual_lr1e-4` (`1e-4`，30 步 + 续训至 80 步 + 全套评测）；
   * `d4_prox07_lr1e-4` (`1e-4`, $\alpha=0.7$，80 步连续训练 + `@50` 与 `@80` 全套评测）；
5. **D3 通用单教师**：`d3_general_lr1e-4`（80 步训练 + 全套评测）；
6. **D4 先验对齐双教师**：`d4_math14b_lr1e-4`（80 步训练 + `@50` 与 `@80` 全套评测）。

### 5.2 OPD v3 配置矩阵与执行状态

仓库中共包含 21 个 `project2_opd_v3_*.json` 配置文件。下表列出全部配置及其声明的消融因子与真实执行状态：

| 配置文件名 (`configs/project2_opd_v3_*.json`) | 相对 `d4_dual` 的声明差异项 | 具体差异细节 | 真实执行状态 (Execution Status) |
| :--- | :--- | :--- | :--- |
| `smoke.json` | 步数/批次/生成截断/预热 | 2步, 4+4题, max_tokens 64, 无预热 | **GPU 已实测** (2 步 Smoke) |
| `d4_dual.json` | *(基准对照组)* | lr=3e-5, Math-7B + Coder-7B, 混合 64+64, exact_rkl | **GPU 已实测** (80 步 + 6 套评测) |
| `d4_dual_lr1e-5.json` | `learning_rate` | lr = 1e-5 | **GPU 已实测** (30 步扫描) |
| `d4_dual_lr1e-4.json` | `learning_rate` | lr = 1e-4 | **GPU 已实测** (80 步 + 6 套评测) |
| `d3_general_lr1e-4.json` | `teachers`, `learning_rate` | 双域教师均为 Qwen2.5-7B-Instruct, lr = 1e-4 | **GPU 已实测** (80 步 + 6 套评测) |
| `d4_prox07_lr1e-4.json` | `loss`, `learning_rate` | proximal_teacher_alpha = 0.7, lr = 1e-4 | **GPU 已实测** (80 步 + 6 套评测) |
| `d4_math14b_lr1e-4.json` | `teachers`, `tokenizer_audit`, `learning_rate` | 数学教师改为 Qwen2.5-14B-Instruct, lr = 1e-4 | **GPU 已实测** (80 步 + 6 套评测) |
| `d1_math.json` | `data_files`, `teachers`, `prompts_per_step`, `max_new_tokens` | 仅数学域单教师, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d1_math_lr1e-4.json` | 同上 + `learning_rate` | 仅数学域单教师, lr = 1e-4 | **纯 CPU 契约测试 (从未上卡)** |
| `d2_code.json` | `data_files`, `teachers`, `prompts_per_step`, `max_new_tokens` | 仅代码域单教师, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d2_code_lr1e-4.json` | 同上 + `learning_rate` | 仅代码域单教师, lr = 1e-4 | **纯 CPU 契约测试 (从未上卡)** |
| `d3_general.json` | `teachers` | 双域教师均为 Qwen2.5-7B-Instruct, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_blocked.json` | `schedule` | schedule = blocked (10步分块交替), lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_blocked_lr1e-4.json` | `schedule`, `learning_rate` | schedule = blocked (10步分块交替), lr = 1e-4 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_pcgrad.json` | `gradient_surgery` | gradient_surgery = pcgrad, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_pg.json` | `loss` | loss.mode = pg (策略梯度), lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_topk.json` | `loss` | loss.mode = topk_rkl (Top-64 RKL), lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_prox07.json` | `loss` | proximal_teacher_alpha = 0.7, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_warmstart.json` | `init_adapter` | 加载 SFT 权重初始化 LoRA, lr = 3e-5 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_warmstart_lr1e-4.json` | `init_adapter`, `learning_rate` | 加载 SFT 权重初始化 LoRA, lr = 1e-4 | **纯 CPU 契约测试 (从未上卡)** |
| `d4_aligned_lr1e-4.json` | `teachers`, `learning_rate` | 7B-Instruct 数学 + Coder-7B 代码, lr = 1e-4 | **纯 CPU 契约测试 (已被 14B 取代)** |

> ⚠️ **重大科研结论警示**：
> 1. 本项目至今**从未在 GPU 上实际测量过单域教师训练带来的跨域遗忘（Catastrophic Forgetting）或跨域负迁移**（`d1_math*` 与 `d2_code*` 从未运行）！
> 2. 分块调度（`blocked`）vs 混合调度（`mixed`）、梯度手术（`pcgrad`）vs 无手术、策略梯度（`pg`）与 Top-$K$ 损失相比全词表 RKL 的优劣，**在真实 GPU 上均无任何实验数据支撑**！

---

## 6. 最新实验状态与下一步：R1 / Latent-MOPD

* **L4 工程短跑**：旧全词表 reverse KL、每步 2 题、128 token 上限，验证了 `token`、`latent`、`math20` 的基础更新及标准 `latent` 的续跑。该记录不验证当前 top-16 或正式长度。[L4 日志](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc3eee16afce6047bf1b9bd62893c1d82e1dc1f8/latent_smoke/20261112T065855Z)
* **当前实现的 CPU 验证**：WSL2 下 `~/.venvs/opd/bin/python -m pytest -q` 为 196 项通过、1 项跳过，包含 top-16 数值及梯度对照。正式训练与评测仍未完成。
* **G4 32 并发**：16K 生成上限，纯 token 组，首步期间采样到 96,139 MiB（93.9 GiB）。会话结束时没有可用 checkpoint，原因未确认。[首次 G4 记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc3eee16afce6047bf1b9bd62893c1d82e1dc1f8/latent_top16_token/20261112T075031Z)
* **G4 16 并发**：仍是 32 题 × 4 条回答、16K 上限，改为第一步结束即保存 checkpoint、每 30 秒采样 GPU。20:57 的尝试由负责人关闭；21:12 重跑，21:33 最后同步时 `metrics.jsonl` 为空，Drive 无 checkpoint，复查时会话已结束。最近一轮没有退出码或错误栈，停止原因未确认。时间为新加坡时间。[手动关闭记录](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc3eee16afce6047bf1b9bd62893c1d82e1dc1f8/latent_top16_token_retry/20261112T125718Z) · [最近一次 G4 日志](https://github.com/XDE10/qwen-dual-teacher-opd/tree/dc3eee16afce6047bf1b9bd62893c1d82e1dc1f8/latent_top16_token_retry/20261112T131202Z)
* **请先设计低成本定位实验**：沿 `colab_latent_top16.sh`、`latent_opd.train_step`、`HFRolloutEngine`、候选打分、反向传播和 `colab_job_v3.py` 逐段定位。用批次进度、有效生成 token/s、长度分位数、截断率、各阶段耗时、allocated/reserved 峰值和 checkpoint 可恢复性区分假设。评估长尾回答、批内空转、动态 KV cache、投影及重复前向，并明确哪一项能由现有日志定位。保留 16K 正式配置；缩短上限或减少 rollout 的测试应说明它只能回答什么。
* **正式对照的顺序**：在相同 R1 基线、数据、预算和评测口径下，先验证纯 top-16 和标准 `latent`。比较两域 dev 和最终测试，并优先报告同一 checkpoint 的收益。若标准 `latent` 的结果表明数学淡出时长值得研究，再考虑 `math20`；给出相对于继续 OPD v3 的时间成本与取舍。

---

## 7. 疑似缺陷、工程漏洞与开放问题清单 (Suspected Issues, Bugs & Open Questions)

请逐项核对下面 13 个审查线索，给出证据，并判断属于已证实缺陷、设计取舍、记录冲突还是待验证假设。

### 1. `latent_opd` 微批次加权与打包依赖
* **源码位置**：`src/training/latent_opd.py` 第 100–101 行：
  ```python
  loss = routed_loss(...) * (count / len(rollouts))
  ```
* **问题机理**：在 `routed_loss` 内部，隐藏层对齐和 Token 策略梯度均是对当前微批次 $m$ 内的 **Token 总数 $N_m$ 取平均**（`.mean()`）；随后外层又乘以该微批次包含的 **序列条数比例 `count / len(rollouts)`**。
* **待核对**：该归约与打包方式有关，一般不等价于严格的整步 `sequence_mean` 或 `token_mean`。论文附录 A 也描述了微批次 token 均值乘回答数比例。请先比较论文每 worker 的规则与本项目单卡实现，再判断是设计取舍、复现差异还是实际缺陷。

### 2. 隐藏层对齐损失尺度与梯度贡献
* **源码位置**：`src/losses/latent_opd.py::hidden_alignment`：
  ```python
  student = F.normalize(student.float(), dim=-1)
  teacher = F.normalize(teacher.detach().float(), dim=-1)
  return (student - teacher).square().mean(dim=(0, 2))
  ```
* **问题机理**：`student` 与 `teacher` 在特征维度已是 $L_2$ 单位向量，二者欧氏距离方差 $\|u-v\|_2^2 = 2(1-\cos(u,v)) \in [0, 4]$。代码调用 `.mean(dim=(0, 2))`，在特征维度取平均相当于除以了 $d = 1536$！
* **证据与待测项**：论文式 (8) 明确除以 $d$。旧 L4 短跑的纯 reverse KL 步与 latent 起始步梯度量级不同，不能据此推断当前 top-16 混合步的两项贡献。请在同批轨迹上比较权重前后各项梯度范数、方向与实际更新，再判断是否需要尺度消融。

### 3. `eval_v3.py::grade_math` 未向判分器透传 `truncated` 状态 (Truncation Masking in Evaluation)
* **源码位置**：`src/evaluation/eval_v3.py::grade_math`：
  ```python
  conservative = conservative_check(text, reference)
  standard = math_verify_check(text, reference)
  ```
* **待核对**：`grade_math` 将 `truncated` 写入输出，两套判分调用均未传这个参数。请区分提前给出有效答案与误取中间答案，核查项目原定评测口径，并比较统一判负、现行判分及分层汇报的敏感性。

### 4. OPD v3 训练 1,024 token 与评测 2,048 token 的长度口径
* **问题机理**：训练时数学生成上限被强制限制为 1,024 tokens（`configs/project2_prompt_contract.json`），而离线评测却给予 2,048 tokens。
* **潜在漏洞**：当学生向啰嗦的 `Math-7B` 学习时，训练中超过 1,024 步的序列被直接腰斩，无法得到终止符与最终答案的监督；而离线评测放宽至 2,048 步，人为掩盖了一部分在 1,024 步时已经截断的推导缺陷。

### 5. 近端教师 ($\alpha=0.7$) 的概率混合与后期熵变化
* **问题机理**：`d4_prox07_lr1e-4` 在 50 步前代码大幅提升，但在 55~80 步学生熵暴增 10 倍，EOS 暴跌。
* **待验证机制**：对数比 $\log p(v) - \log \tilde{q}(v) \le -\log(1-\alpha)$。请推导在学生分布停止梯度时，这个上界对单步梯度的影响，再检查能否用 EOS、长度和分布遥测支持熵变化的解释，并列出替代机制。

### 6. `Math-7B` 先验偏离与局部近视 RKL 的冲突
* **现象**：`Math-7B` 单独表现极强（训练题自判 85.12%），但作为教师在 `1e-4` 训练至 80 步时，学生 `MATH-500` 却退化至 54.80%（低于基线 55.00%），截断数翻倍；而 `7B-Instruct` 与 `14B-Instruct` 却没有此退化。
* **待验证机制**：局部 RKL 可能通过逐 token 学习放大教师的长推导和 EOS 偏好。请用同题截断、长度分位数、EOS 遥测和可比学习率对照检验这一解释。

### 7. 小样本评测集统计检验功效极其受限 (Low Statistical Power)
* **数据现状**：`math_dev` 为 256 题（每题权重 0.39%），`code_dev` 为 200 题（每题 0.5%），`HumanEval+` 仅 164 题（每题 0.61%）。
* **统计漏洞**：在二项分布下，256 题在 60% 准确率附近的二项标准误为 $\approx 3.06\%$（95% 置信区间宽达 $\pm 6.0\%$）。在 `@30`、`@50`、`@80` 之间观察到的 1~3 道题（0.4%~1.5%）的微弱波动，在统计学上是否纯属采样噪声？

### 8. 代码评测分布偏移 (Prompt Format Distribution Shift)
* **问题机理**：训练题与 `code_dev` 均使用纯用户 Prompt，且包含 50% 的 `stdin_stdout` 竞赛输入输出题；而官方 `HumanEval+` 与 `MBPP+` 均为 100% 函数级补全，且使用了带预填充助手前缀的 Chat 格式（`<<<__EVALPLUS_MAGIC_SPLITTER__>>>`）。半数的 stdin 训练题是否对最终标准评测产生过负迁移？

### 9. v5 补齐缺乏语义级审核与历史剔除项追踪缺口
* **待核对**：依据 v5 发布清单、审核报告和迁移台账，核对补齐样本的确定性检查、AI 或人工语义审核实际覆盖的范围，以及历史否决样本能否逐项追溯。

### 10. `evaluation_matrix.json` 中初始 KL 口径统计不一致
* **问题机理**：矩阵中 `d3_general_lr1e-4` 与 `d4_prox07_lr1e-4` 采用前 10 步滑动均值作为初始 KL，而其余项采用 Step 1。这种统计口径的不对齐需要被修正并说明。

### 11. `d4_prox07_lr1e-4` 实验记录中 `resumed_from_step: 50` 的矛盾
* **记录冲突**：`evaluation_matrix.json` 标注该模型从第 50 步断点续训，但训练控制台日志明确显示其在 `session5_fix_math7b` 中单次连续跑完了 1~80 步（8,448s），`session5_eval_prox07` 仅是恢复目录评测了 checkpoint-50 与 80。

### 12. `AGENTS.md` 标注配置数 (13个) 与仓库实际配置数 (21个) 的滞后
* 仓库中实际包含 21 个严格受控的 `project2_opd_v3_*.json` 配置文件，文档陈述滞后。

### 13. `README.md` 验证说明中历史遗留的自相矛盾描述
* 核查 `README.md` 的训练入口及验证边界章节，区分针对某个入口的历史限制与已有上卡、续跑、评测记录，指出仍有歧义的表述。

---

## 8. 要求 GPT-6-Pro 输出的核心交付物与报告规范 (Requested Deliverables)

请 GPT-6-Pro 按照以下 7 项严格的交付物结构输出你的评审报告，严禁任何形式的套话、空泛结论或未经证实的揣测：

### 交付物 1：代码与实现缺陷分级清单 (Ranked Code Correctness Bugs & Vulnerabilities)
* 按严重程度严格划分为 **P0 (Critical / 阻断性或数值完全错误)**、**P1 (Major / 严重偏离理论或引入偏差)**、**P2 (Minor / 规范性与非关键隐患)**。
* 每一项缺陷必须明确指出：
  1. 涉及的具体代码文件路径与行号区间；
  2. 缺陷的数学/工程机理与复现条件；
  3. 导致的实验后果（如显存爆炸、梯度归零、样本加权失真）；
  4. 最小修改量的标准修复代码块（Drop-in Replacement）。

### 交付物 2：统计有效性与评测可靠性重估 (Statistical Re-Analysis)
* 对 Table 2 中的核心对比（D0 vs D4 Dual @30/80、D3 General vs D4 Math14B @80、@50 vs @80 波动）进行严谨的统计重估：
  1. 单个比例可报 Wilson 或精确区间；模型差值优先基于同题通过／失败的配对记录估计；
  2. 在注明配对不一致率、功效与多重比较等假设的前提下，估计最小可检测效应（MDE）；
  3. 说明证据支持的效应方向和不确定性。缺少逐题配对时，列出无法计算的检验和所需记录；
  4. 判定是否存在过拟合开发集（Dev Leakage）或多重假设检验偏差。

### 交付物 3：核心实验现象的深度理论诠释 (Theoretical Explanations for Phenomena)
* 运用强化学习与分布对齐理论，深入解答以下三大机理问题：
  1. **专职数学教师与后期截断**：解释局部 RKL、EOS 和自回归累积如何可能影响长题，给出区分教师偏好与学习率效应的证据。
  2. **近端目标与熵变化**：推导概率混合对局部梯度的影响，标出从单步数学性质外推到训练动力学还需要哪些证据。
  3. **离线 SeqKD 与在线 OPD**：核查是否有可比的训练和评测预算，再讨论两者的轨迹覆盖差异及可能机制。

### 交付物 4：有限 GPU 预算下的高回报后续实验路线 (Prioritized Next-Experiment Plan)
* 在单卡 G4 96GB 的预算下，先提出一项能定位 R1 首步瓶颈的诊断，再给出最多两项正式对照，并说明继续 OPD v3 是否更划算：
  1. 明确列出推荐执行的配置组合（学习率、步数、教师路由策略、生成截断）；
  2. 给出显存、有效生成速度和总时长的估算方法、假设及待实测项；
  3. 写清每场实验的主指标、判停阈值、checkpoint 恢复方式，以及可支持的结论。

### 交付物 5：消融配置的增删决策建议 (Ablation Pruning & Addition Strategy)
* 对 §5.2 中尚未在 GPU 验证的 OPD v3 配置，以及 `math20` 等探索组进行审查：
  1. 哪些配置应当**彻底废弃并从计划中移除**（说明冗余或无科学价值的理由）；
  2. 哪些配置必须**保留并优先上卡验证**（重点评估 `d1_math` / `d2_code` 单域遗忘对照的必要性）；
  3. 是否需要新增急需的关键消融（例如：长度归一化 RKL、余弦隐藏层对齐修正）。

### 交付物 6：科研诚信、论文撰写与过度声称风险预警 (Overclaiming Risks & Academic Warnings)
* 列出如果将当前数据与结论直接写入学术论文，会被顶级审稿人（NeurIPS/ICLR/ACL）精准击穿的软肋与漏洞；
* 给出如何合规、客观表述成果的措辞修改指南。

### 交付物 7：报告格式规范 (Required Output Structure)
* 全文采用专业学术技术规范，以严谨精炼的简体中文输出，专业术语保持标准英文；
* 给出对比总结表格。有足够证据的代码问题附最小补丁或伪代码；待验证问题给可执行的验证步骤。最后标明推荐下一步，以及哪些结论已有足够证据用于实习项目介绍。
