# CHANGELOG — handover 2026-10-24 v2

## 范围与版本

本轮不下载模型权重、不调用付费 API、不启动 GPU 训练。只新增 v2 实现、测试、脚本与报告，并将默认训练入口改为显式要求 v2 配置。原 `src/training/opd.py` 和 `src/losses/reference.py` 保留；旧代码/README快照保存在 `before/`。发布数据、manifest、批准字段和既有SHA全部不变。

## 已修正的实现

1. **采样**：新建GenerationConfig，显式完整支持采样/无惩罚；额外关闭 Transformers 的 `use_model_defaults` 回填。记录解析配置及processor，实际generate链逐次fail-closed核验。首轮新增测试捕获真实suppression注入，修复后通过。
2. **KL显存路径**：按序列/response行分块，先行后列；完整共同词表重归一化和reverse KL不变。自定义一阶backward重算块，避免保存全response概率；模型完整logits仍存在。教师评分use_cache=False。
3. **累积归一化**：默认response_tokens_v2，整更新按有效token归一化；microbatch_mean_v1仅显式保留。长短回答合并/拆分梯度一致性测试通过。
4. **身份与恢复**：提示/tokenizer/训练与loss源码SHA、完整resolved generation配置、依赖/Python/CUDA平台身份；封闭checkpoint文件集合，在torch.load前校验；缺项/空清单/越界/改契约/改tokenizer/改源码拒绝测试通过。已有后续结果默认拒绝，显式archive_future_v2归档后续checkpoint与日志，防止重复step。
5. **资源**：逐阶段同步/reset峰值，记录allocated/reserved及CPU RSS端点；整步removed mass token加权均值/max/分位数；含切换总成本与纯计算成本分列。解析内存包络和chunk16/sequence128候选起点已提供，A100测量未验证。
6. **代码评测**：候选提取、签名检查、保留源调用参数类型、Docker执行、显式exact/float/unordered/stdout协议；构造题没有专用checker则拒绝。200题comparator草案全部unreviewed，未据源测试通过宣称语义终审。
7. **官方评测**：EvalPlus0.3.1候选solution转换、task集合/单候选检查、官方本地数据override与固定镜像Docker命令，内部dev单独协议。真实官方工具镜像和评分未验证。
8. **沙箱**：运行时按当前ROOT重新定位只读worker挂载，历史JSON只读；不隐式pull、不回退宿主机执行。
9. **Colab**：Linux/CUDA候选requirements、只安装/CPU检查的bootstrap、配置化Drive输出及显式恢复流程；D3以模型身份复用同一教师。D3/D4两类预算草稿标注未冻结。

## 实际验证结果

- genai-test Python3.12.15；首次pip安装因WinError32文件占用失败，重试成功；pip check通过，freeze已保存。
- 原README四文件：**20 passed**（readme_cpu_tests.log）。新增v2：**30 passed**（v2_final_tests.log），含真实tiny OPD adapter/optimizer/scheduler/RNG恢复与FP64数值/梯度对照。
- 最终合并：**50 passed**（final_50_cpu_tests.log）；3条tiny PEFT配置提示警告，无失败。这里是CPU测试，不是正式模型训练或CUDA验收。
- Docker真实内部dev参考候选链路：**200/200 passed**。已有固定镜像，未下载新镜像；容器Python3.8.20。首次199+1结果保留；失败源于exact协议额外强制int/float类型一致，与源断言5==5.0不符，修复为无epsilon的Python数值相等并保留bool区分。没有改数据或源测试。
- Docker比较器夹具：浮点容差接受、相同输出精确比较拒绝、无序多重集合接受，三项符合预期。结果在code_evaluation/docker_comparator_fixtures.json。
- 发布保护检查：60个保护文件零变化；迁移清单全文件SHA复核只出现预期的README、默认入口、沙箱shim三处差异，未刷新任何旧SHA。reference.py不变。见immutable_verification.json。

## 只读去重结果

本轮下载公开MBPP train数据parquet（374行）仅用于校准，固定revision/SHA保存在mbpp_train_source.json；没有下载模型，未修改data/目录。

- MBPP train与当前MBPP+复现 **108个task ID重叠**。
- 在扫描训练题前校准：候选阈值均未达到95%召回目标，筛查阈值0.5下正例召回 **76/108=70.37%**，108个异ID负例误报0。阈值校准未达标，不能据此宣布无污染。
- 精确规范化命中0；5-gram/数字模板/受约束containment共 **192对候选**，涉及138个训练ID：数学117个、代码21个。按目标集合：MATH-500 111对、math_dev52对、GSM8K1对、code_dev28对，HumanEval+/MBPP+为0对。相似模板不自动等于污染。
- 若把0.5层所有代码候选都移除，function仍1250，stdio从1250到1229，需要21条新的独立批准stdio替补才能恢复配平；本轮**没有删除或替补**。
- 0.8敏感性层34对候选，其中code_dev1对，仅涉及1条stdio训练题；该层同样未经过充分召回验收。
- Embedding近邻**未执行/未验证**：缺固定本地embedding模型且本轮禁止下载模型。重度改写漏检边界已测试，词面方法不是embedding。

## 未验证 / 后续门槛

真实学生与教师generation_config打印脚本未执行；真实Qwen processor链、模型前向/adapter梯度、A10040/80GB峰值、ColabLinux依赖与Drive恢复一致性均未验证。不得据CPU测试承诺A10040GB安全。Colab中代码执行没有Docker时不受支持；受限子进程与Docker不等价。

200题参考候选通过只验证已有有限测试；逐题float/无序/多解语义checker终审尚未完成。官方EvalPlus适配器单测通过，官方评分未验证。D0～D4正式配置、教师基线与预算冻结仍未完成。

## 源码定位

- `src/training/sampling_v2.py`：generation_config、resolved_contract、audit_actual_chain、generate_exact。
- `src/losses/chunked_opd_v2.py`：row_blocks、_ExactKL.forward/backward、removed_mass_values、summarize_mass。
- `src/training/opd_v2.py`：update_loss、save_checkpoint、run。
- `src/training/runtime_v2.py`：build_identity、validate_checkpoint、prepare_output、PhaseRecorder。
- `src/evaluation/code_v2.py`、`scripts/code_eval_worker_v2.py`：候选协议和真实容器评分。
- `src/evaluation/evalplus_v2.py`：官方协议适配。
- `src/data/decontamination_v2.py`、`scripts/decontaminate_project2_v2.py`：只读候选与校准。
- `tests/test_opd_v2.py`、`tests/test_code_evaluation_v2.py`：新增回归；精确行号保存于source_locations.txt。

外部协议依据仅为官方源码：[EvalPlus v0.3.1](https://github.com/evalplus/evalplus/tree/v0.3.1)、[PyTorch安装组合](https://pytorch.org/get-started/previous-versions/)。来源不替代本项目实测。
