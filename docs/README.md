# docs/ 索引

本目录只存**历史方案、交接记录与迁移台账**，不参与训练/评测运行时读取。当前入口与命令见仓库根 [README.md](../README.md)，协作者规则见 [AGENTS.md](../AGENTS.md)。

## 三个子目录

当前三教师 R1 路线先读 [状态报告](../reports/latent_mopd_three_status_20261118/README.md)、[实现](latent_mopd_three.md)、[完整断点与延长训练](latent_mopd_three_continuation.md) 及 [最终评测规格](latent_mopd_three_final_evaluation.md)。`reviews/` 和根目录的6Pro审查提示词保留为2026-11-12至15日的历史审查材料，其中的状态不是实时结果。

| 子目录 | 内容 | 性质 |
| --- | --- | --- |
| [plan/](plan/README.md) | 3 篇 2026-10 冻结的项目二历史文档：详细实施方案、OPD 提示词与张量接口契约、共用规范；`plan/sources/` 为上游参考源码快照 | **历史**：内部链接与路径指向重排前目录，可能失效 |
| [handover/v2/](handover/v2/README.md) | 2026-10-24 v2 本地修订与 Colab 交接记录：CHANGELOG、测试/环境日志、`before/` 实现快照、交付哈希、去重记录 | **历史交付**：不代表 GPU 验收或正式训练完成 |
| [migration/](migration/reorg_20261108.md) | 2026-11-08 重排说明、迁移清单、`path_mapping.json`、重排前快照 `inventory_before_reorg.json`、退役工具 `legacy_tools/`、删除台账 `deletion_ledger.json` | **当前有效**：查旧路径、旧文件、删除范围先看这里 |
| [opd_v3_mopd.md](opd_v3_mopd.md) | opd-v3 设计：与 MOPD 的对应、单步流水线、损失与归一化、A100/G4 显存规划、1.5B vs 3B、实验配置矩阵、运行顺序 | **当前有效**（代码 CPU 测试通过，GPU 未验证） |
| [handover/kiro_ide_onboarding.md](handover/kiro_ide_onboarding.md) | 从 Kiro Web 转到本地 Kiro IDE 的交接：当前进度、资产位置、WSL2 初始化（venv、测试、凭据、Docker 镜像）、第三次上卡的命令、之后的排期与决策规则、续训的运行身份约束 | **当前有效**（接手从这里开始） |
| [colab_cli_wsl2.md](colab_cli_wsl2.md) | Colab 运行规范（WSL2 + zsh，google-colab-cli 0.7.5）：读写 token、开卡、rclone 挂 Drive、克隆、长任务、结果自动存到 Drive 和 `results` 分支、补发已下载的结果、Colab 预装库冲突、0.7.5 的限制 | **当前有效**（自动保存与推送未在 Colab 上实测） |

## 推荐阅读顺序

0. **接手当前工作**：`handover/kiro_ide_onboarding.md`（进度、初始化、下一步命令），再看 `colab_cli_wsl2.md` 与 `opd_v3_mopd.md`。
1. **了解重组与历史**：`migration/reorg_20261108.md` → `migration/path_mapping.json` → 根目录 `provenance/README.md`。
2. **了解项目设计**（注意是历史方案）：`plan/项目二_双教师在线蒸馏_详细实施方案.md` → `plan/项目二_OPD提示词与张量接口契约.md` → `plan/共用规范_强模型合成数据与验证.md`。
3. **了解 v2 交付边界**：`handover/v2/README.md` → `handover/v2/CHANGELOG.md`。
4. 需要核对具体文件去留时查 `migration/inventory_before_reorg.json` 与 `migration/deletion_ledger.json`。

## 注意

- `plan/` 与 `handover/v2/` 中的路径/链接写于重排前，指向已删除或已移动的目录（如 `project_blueprints/`、`reports/handover_20261024_v2/`）；旧→新位置见 [path_mapping.json](migration/path_mapping.json)。
- 历史文档里的数量、状态、"最新"字样都是当时快照，不构成当前批准或已完成结论；当前事实以根 [README.md](../README.md)、`configs/` 和 `data/` 下 manifest 为准。
- `migration/legacy_tools/` 中的 `verify_migration.py`、`verify_handover_v2.py` 已退役，仅作历史工具；当前验证链是根 README 中的 `pytest`、`tools/verify_code_repair_v4.py` 与 `tools/verify_repository.py`。
- 代码入口（训练/评测/审计/数据工程/环境）的分组与可重跑性见 [../scripts/README.md](../scripts/README.md)。
