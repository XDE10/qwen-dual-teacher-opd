# OPD v2 本地修订与 Colab 操作说明

日期：2026-10-24。v2 不代表正式实验完成。真实 Qwen 前向、A100 40/80GB 显存、Linux/CUDA 安装、Colab 恢复和最终能力分数均为 **未验证**。

## 本轮交付入口

- `CHANGELOG.md`：改动、测试结果和保留边界。
- `project2_opd_v2_smoke.json`：3 步 math/code/math 工程配置，未运行；单教师短配置另存 math_short/code_short。
- `all_cpu_tests.log`：原四文件加新增测试的合并结果；`v2_final_tests.log` 包含追加的真实 tiny OPD checkpoint 恢复用例。
- `pip_freeze.txt`、`pip_check.log`、安装与重试日志：genai-test 的实际环境。
- `memory_estimate.json`：显存解析估算，未验证。
- `code_evaluation/comparator_registry_unreviewed.json`：200 个内部 dev 的逐题 comparator 草案，语义适配未验证。
- `code_evaluation/reference-selfcheck.json`：参考解作为候选，经 Docker 的本轮运行结果。它是源用例相容性检查，不是学生分数或语义完备证明。
- `decontamination_20261024.json`：只读去重候选；根 `reports/decontamination_20261024.json` 保存同一结果。
- `preregistration_draft.md`：D3/D4 更新预算匹配与曝光匹配草稿，**未冻结**。

## 1. 本地环境与保留项

本轮使用 `D:/anaconda3/envs/genai-test/python.exe`，Python 3.12.15。按原 requirements-cpu-snapshot 安装；首次 SymPy 遇 Windows 文件占用，失败日志保留，重试成功。pip check 无损坏依赖。原 README 四个测试文件本轮 **20 passed**。

命令均在项目根执行，以下为 bash 形式：

```bash
conda activate genai-test
python -m pip install -r requirements-cpu-snapshot.txt
python -m pip freeze > reports/handover_20261024_v2/pip_freeze.txt
python -m pip check
python -m pytest tests/test_approved_training_inputs.py tests/test_training_entries_cpu.py tests/test_opd_contract.py tests/test_losses_and_masks.py -q
python -m pytest tests/test_opd_v2.py tests/test_code_evaluation_v2.py -q
python tools/verify_handover_v2.py
```

`reference.py`、所有发布数据、所有已存在的 manifest 与批准 SHA 原样保留。原实现快照在 `before/`。新实现位于 `src/training/*_v2.py`、`src/losses/chunked_opd_v2.py` 和 `src/evaluation/*_v2.py`；旧 `src/training/opd.py` 未改。旧训练入口新增显式 `scripts/train_project2_opd_v1.py` 用于追溯；默认脚本转到 v2，并要求 `implementation_version=opd-v2`，防止旧配置静默改变实验。

原迁移清单记录旧 README、入口和沙箱文件，合法修订后原 `verify_migration.py` 会报告这三处差异。本轮不刷新清单或 SHA；`verify_handover_v2.py` 验证发布身份不变，并单独披露这三处源码/文档差异。新源码身份由 v2 run_identity 和交付哈希记录。

## 2. 采样契约与实际 processor 链

`src/training/sampling_v2.py` 不继承学生 model.generation_config 的分布设置。显式固定 do_sample=True、T=1、top_p=1、top_k=0、repetition_penalty=1，并清除 ngram、suppression、forced token、sequence bias、min-length、typical/min-p/epsilon/eta、watermark、guidance 等分布改写。仍保留学生 EOS 停止语义与 generation budget。

**实测发现**：Transformers 4.57.6 会在 `use_model_defaults=None` 时把“等于全局默认”的新配置字段回填为模型默认值。第一次新增测试确实捕获了 SuppressTokensLogitsProcessor，不是仅凭猜测。最终同时设置 `use_model_defaults=False`，用 pinned `_prepare_generation_config` 和 `_get_logits_processor` 解析契约，并在每次真实 generate 调用处核验链只能含 `CommonSupportMask`。PEFT 的实际 base model 链也被核验，而不是只检查 PEFT wrapper。

启动落盘 `resolved_generation.json` 与 run_identity.generation，包含完整解析配置、use_model_defaults=False 和 processor 类名/支持边界。tiny 测试将模型默认 top_k/top_p/penalty/suppression 设为有害值，实际生成分数中共同支持内所有 token 保持正概率、支持外概率恰为零。真实 Qwen 的链仍为未验证。

Colab 上读取真实固定模型的 generation_config 的检查命令（本轮没有运行，不加载权重）：

```bash
python -m scripts.check_generation_config_colab_v2 --config reports/handover_20261024_v2/project2_opd_v2_smoke.json --allow-config-fetch
```

不传 `--allow-config-fetch` 时只读缓存；加参数只允许读取 config 元数据，不下载权重。输出 raw 模型配置和 v2 请求配置。实际模型链要在后续训练入口加载模型后验证，不能用元数据打印代替。

## 3. 分块 KL、归一化与资源

`src/losses/chunked_opd_v2.py` 先按 sequence_chunk 扫描目标 mask，再以 response_row_chunk 挑预测行，最后取完整共同词表前 N 列。每个有效行的完整 log_softmax 分母始终覆盖全部 N=151665 个 token，**没有词表 Top-K、没有尾部近似**。

为避免“前向分块、反向却保存所有 FP32 中间量”，使用自定义一阶 autograd：保存原 logits/mask，backward 按同样分块重算概率，使用精确导数 `p * (logp - logq - KL)`。FP64 数值、学生梯度与原 reference.aligned_kl 对照；教师返回 None 梯度，词表外/忽略位置/最后无目标位置均为零梯度。该实现针对一阶优化，不承诺高阶导数。reference.py 未改。

模型的完整 `[B,L,V]` logits 和学生全 logits 梯度仍存在；本实现只压低 KL 工作区，**没有消除完整 LM head 的分配**。学生/教师评分均显式 use_cache=False；仅 rollout 使用 KV cache。

默认 `loss_normalization=response_tokens_v2`：每个 micro-batch 的均值 KL 乘其 response token 数反传，所有 micro-batch 完成后将累计梯度除以全更新有效 token 总数，再裁剪/step。这样无需保留整更新所有前向图。旧等 micro-batch 权重只能显式选择 `microbatch_mean_v1`。测试比较长短回答合并与拆分的参数梯度，数值一致。

各阶段（学生加载、教师卸载/加载、rollout、学生评分、教师评分、KL/mass、backward、optimizer、checkpoint）独立同步/reset峰值，记录 allocated、reserved、peak allocated/reserved 和 CPU RSS 端点。CPU RSS 不是连续采样峰值；PyTorch峰值不等于设备所有分配。学生初始加载单独写 startup 时间记录。step total_wall 含教师切换和保存，pure_compute 只含列出的计算阶段；完整作业成本还应加 startup/外围数据和人工排队时间。

removed mass 在原始完整 logits 上、按 response 行分块计算，全 micro-batch 的 token 向量汇总后报 mean/max/min/p50/p95/p99。均值是 token 加权，不再只取最后一批。极小负值保留为 FP32 数值诊断，不暗中修饰为 0。

L 上限需按完整序列计算：数学 P<=3072 加 1024，代码 P<=2560 加 1536，均 L<=4096。解析估算 L=4096 时：学生和一个教师 BF16 权重约17.06 GiB，完整双 logits约2.32 GiB，学生 logits 梯度约1.16 GiB；计入LoRA/优化器和保守工作区后已知项约21.02 GiB。再加假设的6～14 GiB未知激活/内核/allocator等余量，规划包络约27～35 GiB。这是**假设包络，不是测量或上界保证**；加载峰值、SDPA实现、CPU RAM与实际显存占用均未验证。

A100 40GB/80GB 均从 response_row_chunk_size=16、sequence_chunk_size=128 开始，KL额外工作区规划约0.10 GiB；这是低工作区起点，不是整机“保证安全”的参数。40GB 特别要求最长长度真实前后向后至少保留4 GiB实测余量，80GB也需实测后才放大chunk。不得通过缩短问题、Top-K或降低契约来使验收通过。

## 4. 身份、封闭 checkpoint 与 Drive 恢复

v2 run_identity 记录：规范化配置、manifest/audit SHA、提示契约 SHA、本地 tokenizer 文件 SHA、训练/采样/runtime/loss/参考/提示适配器/数据准入/入口源码 SHA、解析后的 generation_config/processor 和已安装依赖版本。输出目录是身份的一部分，切换 Drive 路径需要新实验，不能伪装同一次恢复。

checkpoint 必需 `engine.pt`、`run_identity.json`、`student/adapter_config.json`、`student/adapter_model.safetensors`。清单带 schema，非空且包含必需集合；必须是规范根内相对路径、没有越界/符号链接，实际文件全集与所列集合相等、逐文件 SHA 一致。**在 torch.load 之前全部验证**。身份与 checkpoint 清单提供一致性检测，不是针对恶意篡改者的数字签名。

保存 optimizer、scheduler、每域位置/曝光、step、Python/NumPy/Torch/CUDA RNG，恢复后从完整更新边界继续；中途未保存的 micro-batch重新生成。CPU tiny 的 OPD adapter/optimizer/scheduler/RNG恢复用例与拒绝用例见新增测试。真实 Qwen/CUDA逐步等价仍为未验证。

已有后续 checkpoint 或 metrics step 默认拒绝，显式 `resume_policy=archive_future_v2` 才将后续 checkpoint 与旧 metrics 移入时间戳归档，并保留 <=恢复step的日志；归档不删除结果。resume字段及resume_policy不改变实验身份。

Colab 在 notebook 先挂载 Drive，再使用 bash：

```bash
export PROJECT_ROOT=/content/project2
export OPD_OUTPUT_ROOT=/content/drive/MyDrive/project2/runs
cd "$PROJECT_ROOT"
bash reports/handover_20261024_v2/colab_bootstrap.sh
# 以下是未来人工启动命令，本轮未执行
python -m scripts.train_project2_opd_v2 --config reports/handover_20261024_v2/project2_opd_v2_math_short.json --output-root "$OPD_OUTPUT_ROOT"
python -m scripts.train_project2_opd_v2 --config reports/handover_20261024_v2/project2_opd_v2_code_short.json --output-root "$OPD_OUTPUT_ROOT"
python -m scripts.train_project2_opd_v2 --config reports/handover_20261024_v2/project2_opd_v2_smoke.json --output-root "$OPD_OUTPUT_ROOT"
```

Linux/CUDA 候选清单选择 torch2.8.0/cu126，独立于本轮CPU torch2.14.1；这是未验证候选环境，不能称为已验收锁文件。官方安装组合来源：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。bootstrap 只装依赖并跑CPU tiny测试，不下载模型、不启动GPU训练。Drive应只放输出，项目/tokenizer缓存放本地磁盘；首次真实权重下载另行授权，本轮未做。

中断后流程：恢复相同项目版本和环境 → 挂载同一Drive输出根 → 找最后完整checkpoint（不完整目录保留并改用上一完整点，不能刷新SHA）→ 新配置副本仅添加resume_from_checkpoint与显式resume_policy → 同一max_steps/seed/数据/源码/依赖运行：

```bash
export RESUME_CHECKPOINT="$OPD_OUTPUT_ROOT/opd_v2_route_smoke/checkpoint-1"
python - <<'PY'
import json,os
from pathlib import Path
p=Path('reports/handover_20261024_v2')
c=json.loads((p/'project2_opd_v2_smoke.json').read_text())
c['resume_from_checkpoint']=os.environ['RESUME_CHECKPOINT']
c['resume_policy']='archive_future_v2'
(p/'resume_runtime.json').write_text(json.dumps(c,indent=2))
PY
python -m scripts.train_project2_opd_v2 --config reports/handover_20261024_v2/resume_runtime.json --output-root "$OPD_OUTPUT_ROOT"
```

同一个输出根、相同依赖和源码才允许恢复；不把max_steps从1改3。长序列、切换和恢复的验收仍按v1接手记录中的逐token/状态/峰值对照，不能只检查文件存在。

## 5. 代码候选评分与 Colab 隔离边界

链路：学生生成文本 → 单一代码块或完整Python解析 → 模块级entry/signature验证 → 从私有source tests保留实际参数表达式（避免JSON把tuple变list）→ Docker候选执行 → 显式comparator判分 → 原始结果/身份记录。source tests只进入评测worker，不进入学生提示。

支持 comparator：exact（Python精确相等，5与5.0相等但不使用epsilon，bool与数值区分）、float（明确atol/rtol、拒绝NaN/Inf）、stdout_float_tokens、unordered多重集合，以及单独命名的历史stdout_tokens_exact。构造型多解题必须专用checker；当前没有注册的checker则拒绝，不回退固定stdout。

200题草案为function/exact、stdio/stdout_tokens_exact，全部标记unreviewed，**不等于完成逐题语义终审**；正式学生评分必须提供reviewed sidecar。参考解自检使用reference_selfcheck_only，验证源比较协议，不据此把registry改成reviewed。

```bash
python -m scripts.evaluate_code_v2 --mode registry
python -m scripts.evaluate_code_v2 --mode reference-selfcheck
python -m scripts.evaluate_code_v2 --mode student --candidates /path/to/student_candidates.jsonl --registry /path/to/reviewed_comparators.json --output-dir /path/to/internal_dev_results
```

本机实际通过genai-test的PATH找到 `D:/Docker/DockerDesktop/resources/bin/docker.exe`，使用已有固定镜像 `sha256:3b0c23d092b8ff94262fd3518cf8250c4b1d65c014d6090eea3b23383145201e`。禁止自动pull；命令由当前ROOT/worker.resolve重建，保留历史环境JSON只读。容器禁网、只读、非root、去capability、限制CPU/AS/pid/time；只挂只读worker，无项目/密钥挂载。

Colab通常无法使用该Docker daemon。主方案是Colab仅生成候选，将task_id/代码/身份文件交给本地Docker评分。受限子进程+rlimit只能限制部分资源，不能隔离宿主文件/网络/凭据，**不与Docker等价，本轮未实现为默认fallback**。没有Docker时评测停止并报告基础设施缺口。

第一次参考候选自检199通过、1条出现5.0与expected_repr=5的类型不相等；其源断言是 `==5`，并未规定必须int。新版将exact定义修正为无容差的Python数值相等，保留bool区分，新增回归测试。数据和源测试均未更改。前一次结果保留 `reference-selfcheck_first_199plus1.json`，最终本轮结果单列。源测试通过、覆盖充分和学生语义正确仍是三件事。

官方评测独立使用EvalPlus v0.3.1自带oracle，不用内部comparator重写官方协议。已提供task集合强校验、单候选pass@1 JSONL转换和受限Docker命令。完整Python程序放 `solution` 字段，数据文件用官方环境变量override固定本地版本；来源：[v0.3.1 HumanEval loader](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/data/humaneval.py)、[MBPP loader](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/data/mbpp.py)、[official evaluation](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/evaluate.py)。

```bash
# 在预先构建且锁定evalplus==0.3.1的Linux镜像下执行；本轮官方评分未验证
python -m scripts.evaluate_evalplus_v2 --dataset humaneval --predictions /path/to/humaneval_predictions.jsonl --output-dir /path/to/official_humaneval --image-id sha256:YOUR_PREBUILT_IMAGE
# 仅在已验收容器边界后加 --execute-docker；内部dev与官方分数不能合并
```

本轮没有构建/下载官方评测镜像，也没有官方分数；适配器单元测试通过不等于工具容器端到端通过。首次调用前必须锁定真实镜像ID、配置可写结果目录权限、验收版本/数据SHA和失败分类。

## 6. 只读去重如何解释

使用固定当前2500数学/2500代码训练题，对MATH-500、GSM8K test、数学dev256、HumanEval+、MBPP+和代码dev200执行精确规范化、token 5-gram Jaccard、数字模板5-gram，以及有最小长度/长度比约束的5-gram containment（用于题面加短包装）。只输出候选ID/分数，不删除。

为独立校准108个已知重叠，本轮只下载公开MBPP **数据** full/train parquet（374行），保存在本目录external_sources；没有下载模型。repo、固定revision和SHA见mbpp_train_source.json；与封存MBPP+复现108个task ID重叠。先用这些正例与108个异ID负例选阈值，再扫描训练题。

没有候选阈值达到95%正例召回，最终预扫描筛查阈值0.5，实际召回见calibration；这是**校准未达标的候选筛查**，不是“通过去重验收”。MBPP+存在改写，短题还可能没有共享5-gram。已知ID匹配召回100%不等于文本去重召回100%。另列0.8敏感性参数，不能据训练集命中后反向调阈值。重度改写漏检测试明确记录能力边界。

Embedding近邻未执行/未验证：没有固定且可用的本地embedding模型，本轮禁止下载模型权重；5-gram/数字模板不是embedding替代结果。所有命中均需人工/规格复核，数字替换题可能只是同模板练习。报告包含若全部移除，对function/stdio各1250配平需要多少替补；本轮实际不删、不补、不改批准。

```bash
python -m scripts.decontaminate_project2_v2
```

精确/5-gram未命中不能证明没有语义污染；阈值校准不足和embedding缺失必须随结果报告。

## 7. D3/D4 资源与公平比较

teacher_key=(model,revision)。同一教师映射两个域时，v2不卸载/重载，已新增身份复用测试。不同教师先释放再加载，保留Python/NumPy/Torch/CUDA RNG。实际Qwen切换内存回落仍未验证。

每step记录teacher_reloaded、含切换total_wall和pure_compute，另有startup和checkpoint阶段。D3/D4按相同输入计划比较，报告实际每域prompt/token/接口曝光。更新预算匹配和曝光匹配分开登记，不能把固定步数解释成固定token成本；具体草稿见 preregistration_draft.md，未冻结。
