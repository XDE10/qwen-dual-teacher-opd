# docs/plan/ — 项目二历史方案（2026-10 冻结）

本目录的 3 篇文档是**重排前（2026-10）冻结的历史方案/契约/共用规范**，保留原文用于设计溯源，**不是**当前离线事实或当前批准范围。当前状态、目录与命令见仓库根 [README.md](../../README.md) 与 [AGENTS.md](../../AGENTS.md)。

## 文档

| 文档 | 主题 | 备注 |
| --- | --- | --- |
| [项目二_双教师在线蒸馏_详细实施方案.md](项目二_双教师在线蒸馏_详细实施方案.md) | 项目目标、研究问题、D0～D4 实验设计、评测与验收计划 | 写作时的数据数量与"待实现"状态已被后续数据发布/代码实现部分取代；正文顶部另有多条当时的状态补充 |
| [项目二_OPD提示词与张量接口契约.md](项目二_OPD提示词与张量接口契约.md) | `project2-prompt-v2-20261023`：学生模板渲染、原样 token IDs 直传教师、response mask、共同支持 KL | 机器可读版本已迁至根 `configs/project2_prompt_contract.json` |
| [共用规范_强模型合成数据与验证.md](共用规范_强模型合成数据与验证.md) | 跨项目的合成数据、验证分层、预算与来源记录规范 | 三项目共用；本项目只采用与项目二相关部分 |
| [sources/](sources/) | 上游参考源码快照（OPD 训练脚本、verl teacher loop/config 等）与 `provenance.json` | 方案引用留档，不参与运行 |

## 推荐阅读顺序

1. `项目二_双教师在线蒸馏_详细实施方案.md`：先理解研究问题与 D0～D4 设计。
2. `项目二_OPD提示词与张量接口契约.md`：再理解 token 级接口与 KL 约束（当前执行契约以 `configs/project2_prompt_contract.json` 为准）。
3. `共用规范_强模型合成数据与验证.md`：最后看数据合成/验证方法学。

## 重要提示

- 这些文档的**内部链接与相对路径写于重排前**，指向已删除或已移动的目录（如 `project_blueprints/`、`reports/handover_20261024_v2/`、`reports/strict_v8_20261024/` 等），点击**可能失效**。
- 链接失效时的查法：
  - 旧→新位置与 sha256：[docs/migration/path_mapping.json](../migration/path_mapping.json)。
  - 重排删除/迁移范围：[docs/migration/reorg_20261108.md](../migration/reorg_20261108.md)。
  - 重排前文件快照：[docs/migration/inventory_before_reorg.json](../migration/inventory_before_reorg.json)。
  - 已退役的历史校验工具：`docs/migration/legacy_tools/`（**不是**当前入口）。
- 历史文档中的数量、进度、"最新"字样都是当时快照，不能当作当前批准、当前数据量或"已完成"结论；一切以根 README、`configs/` 与 `data/` 下 manifest 为准。
