# Gemini 建议审查与实施记录 v3

开始于2026-10-24，完成核查日期2026-10-25（Asia/Shanghai）。本轮没有下载模型、调用付费API或启动模型训练。原发布数据、manifest、批准字段、既有SHA和reference.py保持不变。

## 1. 对建议的结论

两条建议的方向可以实施，但不能把“静态规则跑完”解释为所有200题都可以签发，也不能把高相似度候选直接解释为确认污染。

已完成200题的逐项静态决策，产出 [reviewed_comparators.json](reviewed_comparators.json)：**192题status=reviewed，8题status=blocked_after_review**。后者也保留在文件中，有明确问题、原题证据和行身份，不通过修改status冒充已解决。`evaluate_code_v2 --mode student`已接通这份sidecar，会处理全部200个ID、执行192题、保留8题阻断；不会偷偷删掉8题再报200题正确率。

去重方面，原0.5阈值校准仅70.37%已知文本正例召回，未达到95%目标。0.8敏感性层中只有一对代码训练/dev命中：`code/vfc_121257` / `code/vfc_119319`。逐段核查确认它们是同一道道路/铁路共同连通问题，已通过**新版本采样sidecar**排除训练侧一题。原发布文件不变；其他候选全部标记为未确认，不按分数批量删除。

## 2. 200题审核的具体内容

每个registry条目绑定当前dev文件SHA和完整行SHA，记录原题引用、AI审核身份、比较器类型、逐题理由及局限。它是本地工程审核记录，**不是人工金标或数字证书**；不能证明有限用例完备。

审核结果按类型：83 exact、4 unordered、2 float、2 tree、1最大值坐标checker、90 stdout token精确、5允许大小写的token比较、1保留行布局、4专用构造checker，以及8题阻断。

主要修正：

- 平均值与elapsed time使用明确的有限数容差（atol=1e-12、rtol=1e-9），拒绝NaN/Inf和把字符串当数字；该参数是本版自定义数值等价协议，不声称来自题面未给出的官方容差。混合输出只对平均值字段放宽，其余整数计数精确。
- 固定电压常量、要求round到指定位数/两位小数的结果不因“float类型”自动放宽；仍比较量化后的唯一数值。
- 未指定遍历/集合顺序的部分列表按多重集合比较；保留重复计数。嵌套dict不以repr插入次序判错。
- 最大rate并列时接受任一对应坐标，不强制参考程序选择第一个。
- 只有题面明确允许大小写的五题使用casefold；其他字符串不全局放宽。
- 晶体图案保留行内字符和行布局，不以split把图案压成任意token序列。
- 递归exact区分bool和数字，避免嵌套字典内True与1被误当同义。

函数题目前评分仍主要覆盖返回值。要求打印内容、原地别名修改或多接口行为的覆盖局限逐题列出；不将返回值测试通过说成全规格正确。

## 3. 发现并修正的多解题

当前dev确实仍含以下四类多解任务，历史“排除构造题”的概述不能代替当前源码/数据事实：

- `code/vfc_134964`：机器人绕网格环，题面允许任意方向。checker检查有效格、无重复访问、完整覆盖、最终回到S。
- `code/vfc_19131`：任意非空零和子集。checker检查每个testcase的索引范围、数量、互异和元素和为0，拒绝多余token。
- `code/vfc_18674`：将1到n²糖果袋分给n人。checker检查完整唯一使用所有袋、每人n袋和组和相同。
- `code/vfc_17062`：把2n桌腿部件配成n组。checker核对输入多重集合与每对长度和一致，不要求参考的配对顺序。

这些checker直接检查候选对输入的约束，不比较参考输出字符串。回归测试同时覆盖不同于参考的合法构造、重复元素/索引、越界、错误和与多余输出。

## 4. 不能签发为可评分的8题

- `ace/oss_11659`：只说更新projects，未说明必须返回列表；返回None的原地实现不能据题面直接判错。
- `ace/oss_618`：配置字符串的字段、格式、空白和布尔拼写未定义。
- `ace/oss_11985`：允许扩展名表只以示例形式出现，未作为正式输入/常量完整规定。
- `ace/oss_16281`：请求嵌套形状和成功/失败响应schema缺失。
- `ace/oss_13256`：未规定启动和错误消息的确切文本。
- `ace/oss_17515`：键值分隔符、空白及输出键次序不完整。
- `ace/oss_18226`：文字倒计时到0，例子和参考排除0，存在冲突。
- `ace/oss_10109`：题面要求两个函数，当前接口/源测试只覆盖第一个。

它们需要独立的规格澄清、侧车checker或新的评测协议版本；本轮不改写封存题面、源测试或批准记录来“解决”问题。现有registry明确阻断，不签发一个误导性的200/200可评分文件。

## 5. 直接调用学生评测

从项目根运行（bash形式）：

```bash
python -m scripts.evaluate_code_v2 \
  --mode student \
  --candidates /path/to/student_candidates.jsonl \
  --registry reports/handover_20261024_v3/reviewed_comparators.json \
  --output-dir /path/to/student_review_v3_results
```

候选文件仍须包含全部200个sample_id，每行`sample_id`和`text`；8个阻断题不会进入执行器。输出`coverage.total_tasks=200`、`blocked_tasks=8`和passed_tasks，逐题保留阻断理由。若另报可评分子集分数，应明确分母192及覆盖率192/200，不能把它称为原200题全量语义正确率。

容器worker自动根据review-v3选择`code_eval_worker_v3.py`。原v2协议仍可追溯；新版验证registry ID全集及文件/逐行身份，换数据会拒绝。Docker禁网、只读、非root、资源限制不变，没有宿主代码执行回退。

参考程序作为候选的实际回归：`code_evaluation/reference-selfcheck.json`，192通过、8明确阻断。另用同一参考候选调用真实`--mode student`入口验证接线；结果保存在`student_entry_reference_probe/student.json`。这不是学生模型能力测试。

## 6. 去重决策与采样排除

原报告`reports/decontamination_20261024.json`不修改。新版覆盖记录在 [decontamination_review_20261025.json](decontamination_review_20261025.json)，保留192对候选和138个训练ID的标记。除确认同题的一对外，其余191对为`marked_candidate_unconfirmed`，没有据相似度自动判为污染。

确认的一对具有相同的N/K/L城市道路铁路条件、两种连通定义、全部约束、逐城市输出和三组样例；差异为换行/标题包装和解释段。原Jaccard仅约0.526，但containment约0.805；真正的确认来自题面逐段核对，不能反推0.8阈值已获得普遍语义保证。官方HumanEval+/MBPP+在此报告没有命中，不能据此证明没有漏检；embedding仍未验证。

[training_exclusions.json](training_exclusions.json)只列训练侧`code/vfc_121257`，绑定manifest与审核覆盖报告SHA。训练先经原批准加载器，再执行减法过滤，保留其余样本相对顺序。绝不新增批准、回填候选或删除发布行。

有效池为数学2500、代码2499，其中function1250、stdio1249。要恢复1250/1250必须另行提供一个独立批准的stdio替补；本轮没有这样做。D3/D4须共用同一sidecar、输入次序和有效曝光登记，预算未重新冻结。

新配置（未训练）：

```bash
python -m scripts.train_project2_opd_v2 \
  --config reports/handover_20261024_v3/project2_opd_v3_smoke.json \
  --output-root /content/drive/MyDrive/project2/runs
```

另有math_short/code_short。旧v2配置保留追溯，**不自动带排除策略**，不得继续当本次确认后的正式默认配置。root README已改为推荐v3配置。排除文件和过滤源码SHA进入run_identity；有效采样池变化会形成新实验身份，不能从旧采样协议checkpoint继续冒充同一实验。

## 7. 测试与未验证边界

- 新增`tests/test_reviewed_comparators_v3.py`：13 passed，含排除策略字节和源码进入运行身份的验证。
- 最终合并：63 passed；日志`pytest_all_cpu_final.log`，仅CPU tiny与协议测试。较早的62项日志也原样保留。
- Docker参考候选：192 passed、8 blocked_after_review；没有把阻断题算作算法失败或通过。
- 真实`--mode student`入口以参考解作候选：192 passed、8 blocked_after_review；只验证接口接线，不是学生能力结果。
- 四个与参考字符串不同的合法构造均在Docker通过，重复索引的错误构造被拒绝；见`checker_alternatives_docker.json`。
- 发布文件/manifest/reference.py的前后SHA核对、源码改动摘要及CLI探针结果见`verification_summary.json`。

真实学生输出、训练、GPU/Colab、跨域能力和官方EvalPlus评分均未验证。比较器签发范围仅限已明确的评测协议与有限测试，不保证覆盖充分；8题规格问题仍明确未解决。

源码定位：`src/evaluation/reviewed_registry_v3.py:10`验证身份，`:25`构造审核后job；`src/evaluation/comparators_v3.py:20`为构造检查，`:82`为比较分派；`src/evaluation/code_v2.py:88`接通worker并保留阻断题；`src/data/training_exclusions_v3.py:7`执行减法过滤；`src/training/opd_v2.py:71`接入配置；`src/training/runtime_v2.py:48`绑定排除身份。
