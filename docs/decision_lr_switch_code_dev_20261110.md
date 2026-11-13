# 决定记录：学习率切换规则中「code_dev 不低」的定义（2026-11-10）

负责人确认，由 AI 代为记录。

本记录写于 2026-11-10 13:51 UTC（北京时间 21:51），早于三个 checkpoint-30 的 code_dev 打分。写入时，这三个 checkpoint 的 code_dev 都还没有打过分：本地 `~/opd_scores/` 和 `results` 分支的 `code_scores/` 里都没有它们的分数。之后补充了 1e-4 分数的来源说明和"code_dev 条件"一节里的文件情况，判定标准没有改动。

## 负责人原文（2026-11-10）

> 确认判定标准：「code_dev 不低」= checkpoint-30 下，1e-4 的 code_dev 通过数 ≥ 3e-5 的通过数，不设容差。 这是我作为负责人在 2026-11-10 定的。

## 这条定义补充的规则

规则来自第二次上卡报告的"决定"一节（[reports/g4_session2_lr_sweep_20261109/README.md](../reports/g4_session2_lr_sweep_20261109/README.md)）和交接文档第 4.4 节第 2 条（[docs/handover/kiro_ide_onboarding.md](handover/kiro_ide_onboarding.md)）：

> 只有当 1e-4@30 的 math_dev 比 3e-5@30 高 5 个点以上（256 题中约 13 题），且 code_dev 不低时，才改用 1e-4。

本记录只定义其中的"code_dev 不低"，规则的其他部分不变。比较的是 `d4_dual_lr1e-4` 与 `d4_dual`（3e-5）两个 run 的 checkpoint-30，只用 dev，不看测试集。

## math_dev 条件：已满足

（14:56 UTC 起，本节的判断由文末"补充决定"取代：math_dev 条件改按重跑后发布到 `results` 分支的分数判定，本节的 144 和 157 只作参考。）

| checkpoint-30 | Math-Verify（主口径） | 保守判分器 |
| --- | --- | --- |
| 1e-4（`d4_dual_lr1e-4`） | 157/256（61.33%） | 133/256（51.95%） |
| 3e-5（`d4_dual`） | 144/256（56.25%） | 120/256（46.88%） |
| 差值 | +13 题，+5.08 个点 | +13 题，+5.08 个点 |

判定按 Math-Verify：+5.08 个点，超过 5 个点的门槛。

来源：两者都是第三次上卡 `session3/20261110T000932Z` 的 greedy dev 评测（`colab_session_v3.sh` 的 `dev=d4_dual@30` 和 `dev=d4_dual_lr1e-4@30` 两步）。

- 3e-5：`results` 分支 `session3/20261110T000932Z/runs/eval/d4_dual-checkpoint-30-dev/math_dev.summary.json`（`accuracy_math_verify` 0.5625，`accuracy_conservative` 0.46875）。
- 1e-4：第三次上卡最后一次同步在 01:54:01 UTC（`results` 分支 `session3/20261110T000932Z/PUBLISHED.json`，当时 `state` 为 `running`）。`dev=d4_dual_lr1e-4@30` 在这之后才结束（01:54:15 UTC，下一步开始的时间），之后再没有同步，接替它的 `session3_resume` 只从 Drive 恢复了已同步的部分。所以这次评测的输出文件不在 `results` 分支或 Drive 上，Drive 上那份控制台日志也只记到这一步开始。分数只留在该 VM 任务日志（`colab_job_v3.py` 的输出，以 `[colab_job] running:` 开头）的本地副本 `~/opd_runs/session3.log`（WSL2，第 327 行）：`math_dev {"count": 256, "truncated": 21, "accuracy_conservative": 0.51953125, "accuracy_math_verify": 0.61328125}`。

## code_dev 条件：尚未判定

三个 checkpoint-30 里只有 3e-5 的 code_dev 候选文件还在（`results` 分支 `session3/20261110T000932Z/runs/eval/d4_dual-checkpoint-30-dev/code_dev.candidates.jsonl`）。1e-4 和 1e-5 的 dev 评测输出，原因同上，不在 `results` 分支、Drive 和本机上。这两个 checkpoint 的 adapter 还在 Drive 的 `opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-4/checkpoint-30/student/` 和 `.../opd_v3_d4_dual_lr1e-5/checkpoint-30/student/`，候选文件需要在 GPU 上重新生成，之后才能在本地 Docker 里打分。

code_dev 打分结果另行报告，不写进本文件。

## 补充决定：以重跑并发布的分数为准（2026-11-10 14:56 UTC，北京时间 22:56）

负责人确认，由 AI 代为记录。写于重跑上卡之前。

负责人原文（2026-11-10）：

> 学习率决策以这次重跑并发布到 results 分支的分数为准。本机日志里的 157 只作参考，和重跑结果并列报告。如果重跑后 math_dev 的差距不到 5 个点，就不换学习率。

重跑方式（负责人同日决定）：三个 checkpoint-30（3e-5、1e-4、1e-5）的 dev 评测在同一次上卡里全部重跑，任务名 `session3_dev30`，`--sync-every 5`。三个 run 目录都从 Drive 的 `session2/20261109T104721Z/runs/` 恢复。

对前文的影响：

- math_dev 条件：按重跑后发布到 `results` 分支的 Math-Verify 分数判定，1e-4 比 3e-5 高不到 5 个点就不换学习率。前文的 144（`results` 分支）和 157（本机日志）都只作参考，和重跑结果并列报告。
- code_dev 条件：定义不变（1e-4 的通过数 ≥ 3e-5 的通过数，不设容差），用这次重跑生成的候选文件在本地 Docker 里打分。

## 重跑打分结果与负责人最终决定（2026-11-10 16:37 UTC，北京时间 2026-11-11 00:37）

负责人确认，由 AI 代为记录。

1. **重跑与打分实测结果**（出处：`results` 分支 `session3_dev30/20261110T150931Z`，本地打分目录 `~/opd_scores/session3_dev30-20261110T150931Z/`）：
   - `3e-5`（`d4_dual@30`）：`math_dev.jsonl`、`code_dev.candidates.jsonl`、`code_dev.jsonl` 与 `session3/20261110T000932Z` 逐字节相同；`math_dev` Math-Verify 144/256（56.25%），保守判分器 120/256（46.88%）；`code_dev` 84/200（`function` 74/100，`stdin_stdout` 10/100，blocked 8）。
   - `1e-4`（`d4_dual_lr1e-4@30`）：`math_dev` Math-Verify 157/256（61.33%，+13 题 / +5.08 个点），保守判分器 133/256（51.95%）；`code_dev` 82/200（`function` 75/100，`stdin_stdout` 7/100，blocked 8）。
2. **负责人最终决定**：
   - 按前文预注册的字面无容差标准（`code_dev` ≥ 84），`1e-4` 的 `code_dev` 为 82/200（少 2 题 / 1.0 个点，其中 `function` 接口 75 vs 74 反超 1 题，`stdin_stdout` 7 vs 10 少 3 题），字面条件未触发。
   - 但鉴于 `1e-4` 在 `math_dev` 上显著高出 13 题（+5.08 个点），`code_dev` 2 题差距处于单题噪声范围内，且 `3e-5` 在 80 步总预算下效应量过小、不足以支撑后续消融对比检验，负责人决定**切换至 `1e-4`（方案 A）**推进 `d4_dual_lr1e-4` 续训及后续对照组（`d3_general_lr1e-4`、`d1_math_lr1e-4`、`d2_code_lr1e-4`、`d4_blocked_lr1e-4`、`d4_warmstart_lr1e-4`）。报告中同时如实披露预注册字面判定结果与负责人的切换理由。
