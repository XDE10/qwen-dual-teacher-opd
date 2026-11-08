> 历史文档（2026-10 冻结）：本文是重排前冻结的接口契约，保留原文；机器可读的当前版本见 [configs/project2_prompt_contract.json](../../configs/project2_prompt_contract.json)。内部链接可能失效，旧→新映射见 [docs/migration/path_mapping.json](../migration/path_mapping.json)。

# 项目二 OPD 提示词与张量接口契约

版本：`project2-prompt-v2-20261023`。机器可读配置：`configs/project2_prompt_contract.json`。

数学消息固定为显式官方数学 system 加原始 user 题面：`Please reason step by step, and put your final answer within \boxed{}.`。以冻结的学生 tokenizer `apply_chat_template(..., add_generation_prompt=True)` 渲染一次；模板和 tokenizer 文件身份来自 `reports/tokenizer_alignment_report.json`。代码域保持代码 v2 已审计的学生默认 system。生产教师路径没有 chat template 调用。

数据流：原始题面 → `render_student_prompt` → 学生 generation → 原样保留完整 token IDs 与 attention mask → `teacher_batch_from_student` 核验 → 教师在 no_grad/eval 下 forward → 共同支持集上的 response-only reverse KL → 学生一次更新。

- `input_ids`：`torch.long[B,L]`，包含 prompt 与实际生成回答。生成前屏蔽共同支持集以外的输出行。任何越界有效 token 都使准入失败。
- `attention_mask`：同形状，仅 0/1。首版每行只能有一个连续有效区间，允许左右 padding，不允许 packing。
- `expected_prefixes`：学生渲染时保存的未 padding IDs；逐行与完整序列的有效前缀精确匹配，不能只验证前缀长度。
- `prompt_lengths`：上述前缀长度，不含 padding。必须至少有一个回答 token。
- 返回教师输入与学生完整输入共享原张量；教师设备迁移保持数值、形状和 attention 不变。模型需要的 position IDs 必须由同一 attention 规则计算，GPU smoke 要记录两者有效 position IDs 一致。
- `response_mask[B,L]` 标记目标 token 位置，包括学生实际生成的 EOS，不额外添加 EOS。计算 KL 时用 `logits[:, :-1]` 与 `response_mask[:, 1:]`，避免偏移一位。
- 固定四模型共同有效支持集是 ID `[0,151665)`，实际执行从审计读取上界并验证连续性；学生/教师头部填充行不进入采样或 KL。双方裁切后分别以 FP32 log_softmax 重归一化，温度首版为 1。
- padding 值本身可以是学生 pad ID；padding 位置不进入损失，不作为有效生成 token 检查。禁止把 prompt 或参考答案纳入 response mask。

数学生成上限保持 1024、代码 1536、总上下文 4096。超预算题完整排除并报告，不截断题面。项目一 RL 长度讨论不自动修改此处。

CPU 验收覆盖错误前缀、越界 ID、左右 padding、非连续 attention 及 64 个固定 tokenizer 探针；报告位于 `reports/project2_prompt_contract_audit.json`。这些检查没有真实教师模型，也没有验证冻结参数、尾部概率、显存、训练梯度或数学准确率。接入训练入口时必须调用该接口，并在 A100 上验收上述项目，不能因独立模块通过就声称 OPD 已跑通。
