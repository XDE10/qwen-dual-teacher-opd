# opd-v3：单卡 MOPD 式双教师在线蒸馏（设计与运行说明）

状态（2026-11-08）：代码已实现，CPU 契约测试通过（tiny Qwen2，`tests/test_opd_v3.py`）。真实 Qwen 前向、显存峰值、吞吐和能力提升都**未验证**，需要第一次 smoke 实测后再写进报告。opd-v2 保持不变，用于溯源。

## 1. 与小米 MOPD 的对应关系

参考：MOPD（arXiv 2606.30406，Xiaomi，用于 MiMo-V2-Flash 后训练）。

| 项 | MOPD | opd-v3 |
|---|---|---|
| 学生采样 | 学生在线 rollout | 相同；HF 批量采样，走审计过的 sampling-v2 采样链（T=1，无 top-k/top-p，屏蔽共同支持集外 token） |
| 教师路由 | 按 prompt 所属领域派发给对应教师 | 相同（`teachers.{math,code}`）；同一 (model, revision) 只加载一次 |
| 一步内的数据 | 多领域混合 batch | 默认 `schedule.type=mixed`（如 32 math + 32 code）；`blocked` 只作消融 |
| 损失 | 逐 token reverse KL；PG 形式（式 4，优势双边截断）与 top-k 形式（式 5，带 `-p+q` 校正） | `loss.mode = exact_rkl / pg / topk_rkl`；`exact_rkl` 是全支持集精确版（K=V），本项目默认 |
| 序列归一化 | 1/|y| 后对序列求平均 | `normalization=sequence_mean`（默认）；另有 `token_mean`、`domain_token_mean` |
| 教师来源 | 由同一 SFT 起点做领域 RL 得到（同源） | **异源**：现成的 Qwen2.5-Math-7B / Coder-7B，比学生大约 5 倍。处理办法见第 7 节：先用探针测距离，再按关卡决定是否启用 SFT 预热（`d4_warmstart`）、`proximal_teacher_alpha` 或通用教师 |
| 基础设施 | 教师作为独立 prefill 服务，与采样异步重叠 | 单卡同进程：教师常驻显存，rollout 与打分串行 |
| 参数更新 | 全参 | LoRA r16（单卡显存约束；`disable_adapter()` 可零成本拿到原始学生做参照） |

## 2. 一个 optimizer step

1. 每个领域的分层采样器（`sampler_v3.py`）各取若干 prompt。v2 按文件顺序读取，数学前 1419 行全是 MATH，代码前 908 行全是 function，导致短程实验看不到 GSM8K 和 stdin 题。
2. 在 FP32 里把 LoRA 增量 `W + s·BA` out-of-place 写进生成副本（`rollout_v3.GenerationCopy`）。训练模型从不 merge/unmerge，因为 bf16 下 `W + ΔW − ΔW` 不能还原。
3. 每个领域左 padding 批量 rollout，EOS 之后截断（EOS 保留）。
4. 按领域路由教师。micro-batch 右 padding（位置与单条序列一致），只计算"预测回答 token"的行，lm_head 切到前 151665 行，`log_softmax` 即是共同支持集上的重归一化分布。
5. 分块重算的自定义 autograd（`mopd_v3.RowDivergence`）给出损失和梯度，不保留 [N, V] 的 FP32 张量。
6. 可选：分别累积 math/code 梯度，记录余弦；开 `gradient_surgery=pcgrad` 时做投影。
7. 记录遥测：每域 KL、熵、采样 token 的 logp/logq、KL 随回答相对位置的十分位、支持集外概率质量、各阶段耗时与显存峰值。

## 3. 显存规划（bf16，解析估算，未实测）

| 组件 | 1.5B 学生 | 3B 学生（可选） |
|---|---|---|
| 两位 7B 教师常驻 | 28.4 GiB | 28.4 GiB |
| 训练学生 + LoRA + AdamW | ≈3.2 GiB | ≈6.5 GiB |
| 生成副本 | ≈2.9 GiB | ≈5.8 GiB |
| rollout KV（64 条 × 4096） | ≈7 GiB（28 KiB/token） | ≈9 GiB（36 KiB/token） |
| 训练 micro-batch（16k token，只算回答行） | ≈10–15 GiB | ≈12–17 GiB |
| 合计 | ≈52–57 GiB | ≈62–67 GiB |

* **A100 80GB**：用默认配置（`rollout_batch_size=64`，`train_microbatch_tokens=16384`）。
* **G4 96GB（RTX PRO 6000 Blackwell）**：smoke 测出峰值后，可以把 `train_microbatch_tokens` 提到 24576–32768、每步 prompt 提到 64+64。
* **40GB 卡**：`teacher_placement=cpu_swap`，两位教师放主机内存，每步按领域换入两次。

G4 的软件栈：GPU 是 sm_120，必须用 cu128 的 torch 2.8.0 wheel（`scripts/colab_bootstrap_v3.sh` 会检查 `torch.cuda.get_arch_list()`）。v2 bootstrap 用的 cu126 wheel 没有 sm_120 kernel。注意力用 PyTorch SDPA，不依赖 flash-attn。vLLM 0.10.2 只用于评测，可选安装。

**1.5B 还是 3B**：主线保持 1.5B。官方报告的师生差距：1.5B 学生 MATH 55.2、HumanEval 61.6、MBPP 63.2，3B 学生 65.9 / 74.4 / 72.7，Math-7B 的 MATH 是 83.6，Coder-7B 的 HumanEval / MBPP 是 88.4 / 83.5。3B 的可提升空间大约只有 1.5B 的一半（MATH 17.7 对 28.4，HumanEval 14.0 对 26.8，MBPP 10.8 对 20.3），而 MATH-500 单模型的 95% 置信区间约 ±4.3、HumanEval+ 约 ±7.5，3B 的提升更难测出显著性。3B 每步计算量约为 1.5B 的两倍，许可证是 Qwen Research License，还要把它加入 tokenizer 审计。建议预算有余时，在第三周用 D4 配置跑一个 3B 点作为规模对照。

## 4. 实验配置（`configs/project2_opd_v3_*.json`）

| 配置 | 作用 | 与 `d4_dual` 的差异 |
|---|---|---|
| `smoke` | 上卡 2 步冒烟（v4 数据，生成 64 token） | — |
| `d4_dual` | 主方法：混合批次、双专家教师、exact RKL、序列归一化 | 基准 |
| `d1_math` / `d2_code` | 单域诊断，每域曝光与 D4 相同（各 80×32） | 数据、教师、每步 prompt |
| `d3_general` | 通用单教师 Qwen2.5-7B-Instruct（与学生同一后训练谱系） | 教师 |
| `d4_blocked` | 块状交替（每 10 步换域），每域总曝光相同 | schedule |
| `d4_pcgrad` | 梯度手术 | gradient_surgery |
| `d4_pg` / `d4_topk` | MOPD 式 4 / 式 5 两种损失实例化 | loss.mode |
| `d4_warmstart` | 先 SFT 预热、再 OPD（从 `sft_v3_seqkd_mix` 的 adapter 起步） | init_adapter |
| `project2_sft_v3_seqkd_mix.json` | 教师输出上的 LoRA SFT：既是 SeqKD-Mix 基线，也是预热 | 另一个训练器（sft-v3） |

`tests/test_opd_v3.py::test_experiment_configs_validate_and_differ_only_in_declared_factors` 保证上表的"差异"列成立。所有配置都指向 v5 数据（2026-11-08 已批准）。

## 5. 运行顺序

```bash
PROJECT_ROOT=/content/project2 INSTALL_VLLM=1 bash scripts/colab_bootstrap_v3.sh
python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_d4_dual.json configs/project2_opd_v3_d3_general.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_smoke.json --output-root "$OPD_OUTPUT_ROOT"
# smoke 通过后：d4_dual → d1_math / d2_code → d3_general / d4_blocked …
python -m scripts.generate_eval_v3 --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --adapter "$OPD_OUTPUT_ROOT/opd_v3_d4_dual/checkpoint-80/student" --suites math_dev math500 gsm8k_test code_dev humanevalplus mbppplus \
  --release-manifest data/project2_release_v5_20261108/release/manifest.json --output-dir "$OPD_OUTPUT_ROOT/eval/d4_dual"
```

smoke 必须实测并记录：每阶段耗时与峰值显存（`metrics.jsonl` 的 `phases`）、支持集外概率质量、生成副本与 PEFT 前向的一致性、checkpoint 续训（`resume_from_checkpoint`）。`python -m scripts.summarize_run_v3 --run-dir <目录>` 把这些汇总成一个 JSON。

预热路线（需要时，见第 7 节）：

```bash
python -m scripts.generate_eval_v3 --model Qwen/Qwen2.5-Math-7B-Instruct --revision ef9926d75ab1d54532f6a30dd5e760355eb9aa4d \
  --suites train_math --max-new-math 1024 --save-token-ids --output-dir "$OPD_OUTPUT_ROOT/teacher_outputs/math7b"
python -m scripts.generate_eval_v3 --model Qwen/Qwen2.5-Coder-7B-Instruct --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --suites train_code --max-new-code 1536 --save-token-ids --output-dir "$OPD_OUTPUT_ROOT/teacher_outputs/coder7b"
python -m scripts.train_project2_sft_v3 --config configs/project2_sft_v3_seqkd_mix.json --output-root "$OPD_OUTPUT_ROOT"
python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_d4_warmstart.json --output-root "$OPD_OUTPUT_ROOT"
```

## 6. 数据与评测协议

* 数据：v5（`data/project2_release_v5_20261108/release/`，2026-11-08 按负责人指示批准）在 v4 基础上去掉与评测/dev 重合的 133 条数学题和 12 条代码题，数学按同 level 补齐。说明见 [reports/project2_release_v5_20261108/README.md](../reports/project2_release_v5_20261108/README.md)。
* 评测协议 v3（`src/evaluation/eval_v3.py`）：所有模型使用同一套 greedy 解码和固定预算。数学同时报 Math-Verify（主口径）和仓库保守判分器（严格口径）。参考答案自判时，保守判分器在 MATH-500 只认 389/500、math dev 只认 209/256，Math-Verify 在三个数学集上都认 100%。HumanEval+/MBPP+ 按 EvalPlus 官方 chat 提示构造。代码只生成候选文件，执行仍在本地 Docker 沙箱完成。

## 7. 异源教师：风险、测量与退路

**MOPD 自己的对照（§4.4.2）。** 他们把同源的数学 RL 教师换成更强但分布不同的 Qwen3-235B-A22B，其他设置不变。初始逐 token reverse KL 从约 0.04 升到约 0.19。PG 形式下数学准确率逐步下降，熵从 0.30 收缩到 0.21；top-k 形式在约第 18 步发散。他们的结论是同源教师对稳定优化很关键。我们的情况属于同一类：同家族、更大、不同源的教师。

**为什么仍然能做。**
* 强教师 → 弱学生的在线蒸馏是常规做法：Qwen3 用大模型给小模型做先 off-policy、后 on-policy 的蒸馏；Thinking Machines 用 Qwen3-32B 蒸馏 SFT 过的 Qwen3-8B-Base。它们的教师都不是从学生做 RL 得来的，常见做法是先做一段 SFT，把初始 KL 降下来。
* 跨模型蒸馏最难的一关是词表与模板不一致。这里四个模型的词表逐 ID 一致（已审计），提示由学生模板渲染一次，两位教师的默认 system 与我们的提示一致。
* MOPD 失稳的是 PG 和 top-k 两种形式。我们默认用全支持集精确 KL：每个位置对整个分布求梯度，没有"单个采样 token 被教师判为极不可能、得到巨大负优势"的问题。它不是免疫，但比那两种形式稳。
* 我们的可提升空间大（MATH 55 → 84），MOPD 的学生离教师只差几分。

**测量（训练前约 15 分钟）。** `scripts/probe_teacher_fit_v3.py` 在未训练学生的 rollout 上，测数学教师、代码教师、通用教师（Qwen2.5-7B-Instruct，与学生同一后训练谱系）对两个领域的 3×2 KL 矩阵，并给出熵、"教师认为极不可能的学生 token"占比、KL 随位置的分布。

**关卡（D4 前 20–30 步）。** 同时满足以下四条才继续：各域 KL 呈下降趋势；学生熵相对第 1 步下降不超过 30%（MOPD 的失稳信号是 0.30 → 0.21）；EOS 率与平均回答长度变化在 ±30% 以内；梯度范数没有反复出现超过中位数 10 倍的尖峰。

**退路（按成本排序）。**
1. `proximal_teacher_alpha` 设为 0.5–0.8（只需重启训练）。
2. SFT 预热：生成两位教师在训练题上的回答，做 1 个 epoch 的 LoRA SFT，再从该 adapter 做 OPD（`d4_warmstart`）。这一步同时产出 SeqKD-Mix 基线，没有白做的工作。
3. 改用通用教师（`d3_general`）。

**为什么不换成真正同源的教师。** 那需要先对 1.5B 学生分别做数学 RL 和代码 RL 得到教师，即 MOPD 的 Stage 2。代码 RL 需要大规模沙箱执行，Colab 没有 Docker；三周、单卡内做不完，得到的 1.5B 教师也很弱。
