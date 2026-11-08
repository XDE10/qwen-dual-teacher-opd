# 代码训练补题发布 v4（2026-10-25）

已将有效代码训练池从2499恢复为 **2500题：function1250、stdin/stdout1250**。没有重新放回已确认的泄漏题，也没有修改旧发布文件的批准或SHA。

## 当前文件与入口

- 新发布：`data/code_training_repair_20261025_v4/release/manifest.json`。
- 当前代码训练文件：`data/code_training_repair_20261025_v4/release/project2_code_train.jsonl`。
- 根配置：`configs/project2_opd_v4_smoke.json`，另有v4 math_short/code_short。
- 新增：`code/vfc_96076`；移除：与dev同题的`code/vfc_121257`。
- 原始代码JSONL第2452行（零基slot2451）替换为新题，其余2499条记录逐字节保留。
- 数学train2500/dev256、代码dev200、dev保护ID文件均与portable_release逐字节相同。

从项目根执行只读验收：

```bash
conda activate genai-test
python tools/verify_code_repair_v4.py
```

今后训练使用v4配置，不再使用v3的2499池配置：

```bash
# 未来命令，本轮未执行训练
python -m scripts.train_project2_opd_v2 \
  --config configs/project2_opd_v4_smoke.json \
  --output-root /content/drive/MyDrive/project2/runs
```

新配置移除了旧manifest绑定的training_exclusions字段，因为新代码文件已物理替换该槽位、没有那条重复题。原v3排除sidecar仍保留追溯。新manifest产生新运行身份/输出目录，不能从旧数据池checkpoint伪装为同一实验续跑。模型revision、模板、完整词表KL和优化目标未改；D0～D4正式预算仍未冻结。

## 补题来源与准入

历史封存的train-only reserve共179条，124条有逐题AI pass，其中17条既不在当前训练集也不在dev保护ID内。这17条仅有审核证据，不因为是备用题就自动批准训练。

本次选择`code/vfc_96076`：给定四条水平/竖直线段长度，求能围成矩形的最大面积。输入、输出和正整数约束完整，输出是唯一整数，示意图不是求解所必需。

来源为`open-r1/verifiable-coding-problems-python`，固定revision `b761a24a95fa03289a231d2d31c183636ffb9833`。只搜索源parquet的problem_id列，再读取命中row group：`train-00006-of-00011.parquet`、row group2、组内offset984。源文件SHA核对通过，剥除原生Markdown fence后的代码和全部测试与封存候选一致；没有修改候选程序或用例。证据见`source_verification.json`。

历史AI审核request ID：`c8eb9c6c-37b8-487e-ab12-c3743b0ff298`，完整pass记录和原请求已复制到本目录。旧export只因`train_interface_quota_surplus`未采用该题，不是语义isolate/uncertain重试。本轮没有重发请求或花费API预算。

新的准入决策见`admission_decision.json`，明确本轮用户授权只补一条、AI非人工金标、每项证据文件哈希和导出行身份。原reserve中的training_approved=false不改；导出器只有在所有门槛通过后，才创建独立新发布中的批准行，其他备用题不随之放行。

## 本轮实际检查

- 本地固定学生tokenizer文件SHA匹配既有审计，真实学生模板提示 **608 tokens**，预留1536，总2144<=4096；提示ID均在共同支持内。
- 与1325个当前/历史dev与dev候选题面版本及HumanEval+/MBPP+执行精确/5-gram/数字模板/containment筛查，阈值0.5零命中；既有Protected/0.85三元gram检查通过。
- 与当前训练池的family_id、problem_hash、reference AST无碰撞，0.85训练题面检查无命中。新题不在1441个保护ID中。
- Docker **202组源用例全部通过**（201种不同输入；重复用例保留，没有删掉）。
- **48组独立oracle全部通过，包含576个四元组小例子**：穷举24种对边分组，以每组最短边限制对应矩形尺寸，独立于参考的排序公式。
- 三个错误程序均被拒绝：固定首例输出、空输出、把最小两条边乘积误当答案。
- 新发布通过approved loader、提示白名单、接口配额、唯一ID、保护集隔离和逐字节保留检查；60个旧发布保护文件零变化。
- 新增13项回归，最终合并CPU测试 **76 passed**。见`pytest_admission.log`（首批12项）及`pytest_all_cpu.log`（包含发布后检查的最终76项）。

这些是有限源测试、独立小规模oracle和静态审核证据，不是穷尽正确性证明。Embedding近邻未执行/未验证；不声称排除了所有语义改写或预训练污染。保留v3里其他未确认的相似度候选，不借补题把它们自动改成通过。

## 拒绝记录与工具修正

- `code/vfc_90438`：历史AI pass且独立小例子通过，但第21组源用例出现输入协议错误，因此本轮不准入。未修改源程序、用例或旧AI结论。证据在`rejected/code_vfc_90438/`。
- `code/vfc_121103`：与旧保护dev `code/vfc_26925`存在0.5106 containment命中，保守不选，不将这一分数单独宣称为确认重复。证据在`rejected/code_vfc_121103/`。
- 预检读取历史JSONL时发现literal U+2028不能用str.splitlines当记录分隔；新脚本改为按真实文件行读取，保留首次解析错误日志并新增回归。数据未修改。
- 原始gold code带Markdown fence，新源核对脚本复用已有python_code抽取规则；保留raw/normalized SHA，不把格式抽取伪称成代码内容修改。

## 实现定位与复现

- `scripts/prepare_code_training_repair_v4.py:43`：冻结候选、批准证据、提示长度、保护/去重预检。
- `scripts/verify_code_repair_source_v4.py:15`：定点回读固定源revision和SHA。
- `scripts/validate_code_training_repair_v4.py:59`：独立几何oracle；`:72`运行Docker源程序与错误实现。
- `src/data/code_training_repair_v4.py:16`：证据门槛；`:67`只替换一条并核对配额。缺审核、源测试/独立oracle失败、mutant存活、overlap或缺门槛字段均拒绝。
- `scripts/export_code_training_repair_v4.py:15`：只写新目录，禁止覆盖已发布v4。生成新manifest和根配置，不批量刷新旧SHA。
- `tools/verify_code_repair_v4.py:19`：无需模型/网络的发布复核。

预检/源核对/执行脚本均可单独复核；export仅供首次发布，重复执行会拒绝覆盖。后续发现问题应再建版本，不回写本版。候选私有程序、测试与审核意见只在reports中，未导出到学生训练问题提示。

## 未执行与保留边界

没有下载模型、调用付费API、运行学生/教师真实前向或启动GPU训练。代码dev的192可评分/8阻断状态没有改变，官方EvalPlus和Colab/A100验收仍未验证。本轮只完成一条训练补题及数据版本接线。
