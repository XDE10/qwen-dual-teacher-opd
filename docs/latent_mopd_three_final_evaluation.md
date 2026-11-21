# 开发集与最终评测

开发集用来选 checkpoint。对 60、70、80 步先做固定小样本筛选，再扩充 60 与 70 的配对样本。70 步用于最终九项评测，依据见[收尾报告](../reports/latent_mopd_three_closeout_20261121/README.md)。

## 论文参考和本地 70 步

论文的 Base 和同家族 last-3 62 步取[表 1](https://arxiv.org/html/2610.02381v1#S4.T1)。本地 70 步使用同一份已校验学生权重，按[附录 A](https://arxiv.org/html/2610.02381v1#A)披露的生成与判分口径组织评测。原始分、补回明确答案后的分和改提示重测的分分别记录。

- 数学使用 AIME24、AIME25 和 Minerva。AIME 每题 16 个生成，Minerva 每题 4 个，按回答计分，生成上限为 31,744 tokens。本地 AIME25 的温度为 0.6，AIME24 与 Minerva 为 0.7。Math-Verify 比较数学表达式，受控复判记录逐条证据。
- 代码使用 MBPP、MBPP+ 和 LiveCodeBench-e。MBPP 和 MBPP+ 用同一批 378 题各两个程序，EvalPlus 执行基础和扩展测试。LCB-e 原版生成 165 题，按日期范围计分 67 题，每题四个程序。代码始终在 Docker 中判题。
- 推理使用 BBH、MuSR 和 Reasoning Gym。BBH 27 个配置各 40 题。MuSR 三个子任务各 250 题。GYM 用本地重建的 37 类各 25 题，并按各任务验证器计分。作者逐题清单未公开，GYM 只列参考对照。

## 读分方式

评测分数记录题目数量、实际输出和失败情况。更改提示词或答案提取后保留一条独立重测记录。MBPP 调整回答格式和函数接口提示，LCB-e 明确输入输出与完整代码交付，MuSR 物品追踪使用逐事件的角色视角并复判明确的最终选项。各组原分与重测证据见[结果表和判分说明](../reports/latent_mopd_three_closeout_20261121/README.md)。
