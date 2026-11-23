# 读懂项目结构

先用 [首页的 CPU 命令](../README.md#下载后先跑一次) 验证环境。接下来按一题如何变成一次训练更新、一次评测结果的顺序读。

```text
源码根目录
├── scripts/      数据、评测、续训和归档命令
├── src/
│   ├── data/     读取批准数据、过滤重复和重合题、Docker 沙箱入口
│   ├── teachers/ 输入契约与共同词表审计
│   ├── training/ 生成、训练循环、运行身份、断点
│   ├── losses/   预测行、词表投影、Top-16 与 last-3 的数学
│   └── evaluation/ 生成适配、答案提取、数学与代码判分
├── data/         主线需要的冻结开发题、MATH-500 与代码判题注册表
├── configs/      数学与代码开发评测共用的提示词契约
├── requirements/ CPU 和 CUDA 的 Python 依赖
├── tests/        训练、数据门禁、对齐、判分和续训回归
├── tools/        快照交付检查
├── docs/         运行说明和架构导读
└── reports/      70 步成绩、分母和原始与重测协议
```

## 1. 一条题怎么进入训练

[`prepare_latent_mopd_three`](../scripts/prepare_latent_mopd_three.py) 下载固定版本的 DAPO、OpenCodeReasoning 和评测数据，并用固定版本 Reasoning Gym 生成逻辑题。先与受保护题比对、去重，再用学生模板渲染并检查 2,048 token 上限。构建产物是候选清单。负责人按候选哈希批准后，三份训练 JSONL 才能被训练入口读取。

[`read_schedule`](../src/training/latent_mopd_three.py) 读批准清单和三份 JSONL，校验文件哈希、逐行状态、受保护 ID、渲染后长度与前缀哈希。按数学、代码、逻辑顺序轮转，每次从当域的批准池顺序取 32 题。旁边的 `schedule.jsonl` 是排程的核对记录。

## 2. 一步训练里四个模型各做什么

1. [训练器 `run`](../src/training/latent_mopd_three.py) 查当前域，把它的题交给 [学生 rollout](../src/training/rollout_v3.py)。每题生成四条回答，正式配置每步合计 128 条，单条最多 16,384 个新 token。
2. 把每条前缀与回答拼接、右填充成 `[B,L]`。[`response_batch`](../src/losses/latent_opd.py) 从回答目标位置向前一格找到预测位置。比如位置 3 开始是回答，位置 2 的 logit 就负责预测第一个回答 token。
3. 教师用学生生成的同一份 token 和 mask。在共同的 151,665 个 token 上做全支持归一化，再取学生 Top-16 候选算优势。使用当步教师有各自的更新计数，前三个全局步只做 last-3 隐状态对齐，到第 31 步切为纯 token 目标。
4. 学生带梯度重跑。隐状态目标把每个预测位置的三个层向量做 L2 归一化后求 MSE。训练器按微批回答条数占比累积梯度，整步只调用一次 AdamW。三位教师均冻结并常驻同一 GPU，每步只调用当前域的一位。

要精读损失，从同一文件的 `score → objective → update` 开始。看代码中的 `right_pad` 和打包边界时，带 `v2` 或 `v3` 的若干模块提供共享函数。部分旧文件路径也由现有 checkpoint 的源码指纹引用。

## 3. 更新之后留下什么

`runs/` 在运行时生成。先从 `metrics.jsonl` 看每步耗时、截断数和梯度。卡住时看 `phase_metrics.jsonl` 最后一个变化的阶段和进度。到保存点，`checkpoint-N/student/` 存模型，`engine.pt` 存训练状态，运行身份与文件哈希记录在同目录。恢复用完整包，评测用封存学生权重。[断点与派生续训](latent_mopd_three_continuation.md)。

## 4. 分数怎么出来

[`evaluate_latent_mopd_three_dev`](../scripts/evaluate_latent_mopd_three_dev.py) 生成三域的开发集回答。Math-Verify 做数学等价判分，Reasoning Gym 用对应任务验证器，代码先产出候选，再由 Docker 跑测试。开发集用于选步数。选定 70 步后的九项基准、对论文的比较和重测证据在 [收尾结果](../reports/latent_mopd_three_closeout_20261121/README.md)。[`scores.json`](../reports/latent_mopd_three_closeout_20261121/scores.json) 是同一批结果的机器可读记录。

## 5. 以后新增文件放哪

可以从源码启动的入口放 `scripts/`，核心逻辑放 `src/`，影响结果的协议写在 `docs/` 或对应的分数说明。保留覆盖现有训练和判分契约的测试。模型、优化器、生成原文与 GPU 日志进运行归档，当前结果只提交摘要和分母。旧双教师试验、迁移台账、过程报告及对应测试可按固定提交查阅。[精简前快照](https://github.com/XDE10/qwen-dual-teacher-opd/tree/61f021c1aad6d31415b2de4f905153363f11374b)。
