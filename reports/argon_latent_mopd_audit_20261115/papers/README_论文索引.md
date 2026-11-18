# Argon / Gemini Latent-MOPD 论文资料索引

下载日期：2026-11-15。PDF 文件名中的 arXiv 版本号即下载版本；SHA-256 见同目录 SHA256SUMS.txt。

| 文件 | 论文 | 与目标的关系 | 官方页面 |
|---|---|---|---|
| 01_Latent-MOPD_2610.02381v1.pdf | Latent-MOPD: Latent Multi-Teacher On-Policy Distillation | **主目标论文**；三教师同源 1.5B，数学/代码/逻辑路由，62 步，末三层表征监督和 per-teacher crossfade | https://arxiv.org/abs/2610.02381 |
| 02_Replay-Continual-MOPD_2611.02746v2.pdf | Replay for LLM Continual Learning with Multi-Teacher On-Policy Distillation | 重要后续工作；含三域 joint MOPD / Joint Latent-MOPD 对照、数据/评测附录和 LastOPD 代码基础说明；不是单纯的 Latent-MOPD 原作者仓库 | https://arxiv.org/abs/2611.02746 |
| 03_Rethinking-OPD_2604.13016v2.pdf | Rethinking On-Policy Distillation of Large Language Models | token-level OPD 理论与 top-k 机制背景；配套仓库应为 Thinking-Space/Rethinking-OPD | https://arxiv.org/abs/2604.13016 |
| 04_OPRD_2606.06021v1.pdf | OPRD: On-Policy Representation Distillation | 表征蒸馏基础；OPRD-Bridge 提供部分可复用工程实现，不等于完整 Latent-MOPD | https://arxiv.org/abs/2606.06021 |
| 05_LastOPD_2609.28845v1.pdf | LastOPD: Taming Collapse in Latent On-Policy Distillation | Latent 监督崩塌分析；单教师、末层、短 crossfade，不能直接替代 Latent-MOPD 同源三层配方 | https://arxiv.org/abs/2609.28845 |
| 06_PR-OPD_2609.36642v1.pdf | PR-OPD: Privileged Representation On-policy Self-Distillation for Agentic Reinforcement Learning | Agent skill-conditioned self-distillation；方法目标与三教师 MOPD 不同，仅供相关工作/技术对照 | https://arxiv.org/abs/2609.36642 |
| 07_MOPD_2606.30406v1.pdf | MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training | 多教师 token-level 路由基线；与 Latent-MOPD 的 latent channel 不同 | https://arxiv.org/abs/2606.30406 |
| 08_Open-MOPD_2608.19098v1.pdf | Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation | 领域 token-budget 平衡与公开端到端配方；不是 Latent-MOPD 的等价实现 | https://arxiv.org/abs/2608.19098 |

## 仓库索引

- Latent-MOPD（官方占位仓库）：https://github.com/fangzy96/Latent-MOPD
- OPRD（OPRD-Bridge 分支）：https://github.com/ShenzhiYang2000/OPRD/tree/OPRD-Bridge
- Rethinking-OPD（Gemini 将其误写为 thunlp/OPD；实际仓库）：https://github.com/Thinking-Space/Rethinking-OPD
- LastOPD：https://github.com/Muyiiiii/LastOPD
- PR-OPD：https://github.com/balibata/PR-OPD
- Open-MOPD：https://github.com/BytedTsinghua-SIA/Open-MOPD

本资料包只收录论文 PDF 与校验清单；代码仓库请通过上面的官方 URL 获取，避免把未经核实的代码快照混入论文归档。
# 附件保留范围补充（2026-11-18）

Git只保存本索引与原始 `SHA256SUMS.txt`。PDF和ZIP是本地原附件，未复制进代码仓库；下文论文清单及哈希用于核对本地副本，不能据此认为论文二进制已随Git提供。上级目录的106字节审查报告是附件占位说明，完整审查结论未由该文件承载。
