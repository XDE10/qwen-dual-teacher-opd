# 项目二：数学与代码双教师在线策略蒸馏

更新：2026-10-24。本 README 是新接手者和 Codex 的项目入口，覆盖当前事实、运行入口与后续验收。这个目录可以独立迁移，不需要另外两个项目目录。原 `modern_genai` 内的项目专用内容已清理；旧原仓库绝对路径只存在于部分历史记录，不是运行依赖。

## 项目目标

研究一个 1.5B 学生如何在数学与代码两个领域，通过学生在线生成、冻结教师对同一 token 轨迹评分、reverse KL 更新，学习领域路由的专业教师组合。核心是严格 token-level **On-Policy Distillation（OPD）**，同时比较领域能力、遗忘与教师成本。

当前已具备数据、tokenizer/提示词契约和单卡分时教师训练入口；还没有学生或教师的真实 GPU 能力基线、训练收益与完整 D0～D4 结果。普通文本 API 离线生成解答不能代替本项目的全词表教师分布。

## 给 Codex 的最小阅读顺序

1. [AGENTS.md](AGENTS.md)：本目录的项目身份和阅读入口。
2. 本 README：当前状态、代码地图和实际命令。
3. [当前运行数据清单](data/portable_release/manifest.json) 与 [当前配置目录](configs)：核对实际文件、批准范围、模型和输出身份。
4. 按任务阅读下方相关源文件；再查 [完整研究方案](project_blueprints/项目二_双教师在线蒸馏_详细实施方案.md) 与 [Strict v8 实测报告](reports/strict_v8_20261024/执行报告.md)。

`data/portable_release/manifest.json` 的 `approved_training_files` / `approved_evaluation_files` 是本包当前数据准入范围。清单里的跨项目 `summary` 是原联合报告的背景，不会自动扩大授权。代码功能以实际源码为准；旧方案中的“待实现”“暂停写脚本”或旧数据数量可能已被后续实现和发布取代。具体冲突先看文件和版本，不能按段落里的“最新”二字推断。

## 当前事实与未完成事项

- 数学：**2500 train / 256 dev**。代码：**2500 train / 200 dev**；代码 train 中 function 与 stdin/stdout 各 1250，dev 各 100。
- 当前单一配置 [configs/project2_opd_smoke.json](configs/project2_opd_smoke.json) 只有 2 个更新、每域 block 1 步，用于验证数学与代码教师切换。
- 学生常驻，两个教师按领域顺序加载/卸载；这不是两教师同时驻留、熵融合或多卡实现。
- 固定共同 token 支持为 **151665 个 ID（0～151664）**；当前代码要求它连续且 token 映射一致。
- 本包内相关 CPU 测试记录为 **20 passed**，迁移 SHA 校验通过；教师真实前向、词表外概率质量、完整 logits 显存与恢复仍需 GPU 验收。
- dev 的参考实现通过源测试只是已有证据；构造型多解任务的学生输出仍可能需要语义 checker，不能全部使用固定 stdout 相等来评价。

当前只有工程 smoke 配置，没有本项目 GPU 训练产物、完整正式实验结果或已实测的 Linux/CUDA 环境锁文件。原仓库借用过的 Windows 虚拟环境已经清理；现在需要在本项目或目标训练环境重新创建依赖环境。

## 模型与数据来源

- 学生：`Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 数学教师：`Qwen/Qwen2.5-Math-7B-Instruct`，revision `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`。
- 代码教师：`Qwen/Qwen2.5-Coder-7B-Instruct`，revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`。
- [真实词表审计](reports/tokenizer_alignment_report.json)、[提示词契约](project_blueprints/configs/project2_prompt_contract.json) 和 [契约审计](reports/project2_prompt_contract_audit.json) 是模型/模板身份的主要证据。
- 原始词表输出维度不同不能直接取 `min(vocab_size)` 计算 KL；当前按审计得到的共同支持重归一化，真实移除概率质量需随训练记录。

## 当前数据与使用边界

- [数学训练题](data/portable_release/project2_math_train.jsonl) 与 [代码训练题](data/portable_release/project2_code_train.jsonl)：manifest 明确批准训练。
- [数学 dev](data/portable_release/project2_math_dev.jsonl) 与 [代码 dev](data/portable_release/project2_code_dev.jsonl)：只用于评测/调参，未批准训练。
- [dev 保护 ID](data/portable_release/reserved_dev_ids_never_train.json)：保留旧 dev 和相关候选的排除身份。

数学数据来自封存数学池，代码数据具有 function 或 stdin/stdout 的接口协议。模型只收到题面/接口提示；参考程序、标准答案、测试和审核内容保留为私有评测/溯源字段，不进入 OPD 问题提示。候选、隔离与旧 v1/v2/v3 目录不因同处本包而自动获得训练批准。

所有已批准训练行仍需要通过文件 SHA、逐行 `training_approved`、唯一 ID 和模型输入白名单检查。候选、隔离、存疑数据不可通过手改批准字段或仅更新哈希进入训练。新数据版本应保留原发布与来源证据，验证/测试内容不能进入合成生成器。

## 目录与代码地图

- [configs](configs)：本项目使用相对路径的最新工程配置，执行时从本目录根部运行。
- [data/portable_release](data/portable_release)：当前实际运行副本；原始发布和历史版本在 `data/` 其他目录中留档。
- [src](src)、[scripts](scripts)、[tests](tests)：训练、数据、评分与测试实现。迁移保留了一些共用模块和其他项目脚本作为依赖/证据；存在于本包不代表它属于本项目主线或能不加选择地运行。
- [project_blueprints](project_blueprints)：详细研究方案、共用规范和历史配置。里面旧路径配置不能替代根目录 `configs/`。
- [reports](reports)、[迁移记录](迁移记录)：逐项审核、失败/隔离证据、版本化测试与迁移验收。
- [tools/verify_migration.py](tools/verify_migration.py)：无需模型或网络的离线完整性校验。
- [math](math)、[agentic_rl](agentic_rl)、[video_mapping](video_mapping)、[tiny_posttrain](tiny_posttrain)：迁移补齐的参考学习材料。它们不是本项目的正式执行根目录。

本项目主要实现：

- [scripts/train_project2_opd.py](scripts/train_project2_opd.py)：当前 OPD 命令入口。
- [src/training/opd.py](src/training/opd.py)：学生生成、分时教师、reverse KL、梯度更新、领域位置和完整 checkpoint。
- [src/teachers/opd_contract.py](src/teachers/opd_contract.py)：学生提示、相同 token IDs/attention mask 直传教师、response mask 与前缀校验。
- [src/losses/reference.py](src/losses/reference.py)：共同 token 支持上的 KL 数值参考。
- [src/data/approved_training_inputs.py](src/data/approved_training_inputs.py)：文件授权、哈希与 OPD 输入白名单。
- [src/data/code_sandbox.py](src/data/code_sandbox.py)、[scripts/code_sandbox_worker.py](scripts/code_sandbox_worker.py)：代码源参考执行与沙箱支持。目标 Linux/Docker 环境需重验，不能把本机历史执行记录当作新环境已就绪。
- [scripts/evaluate_approved_math.py](scripts/evaluate_approved_math.py)：可复用的固定 greedy 数学评测；官方 EvalPlus 及完整混合域结果汇总还需按锁定协议接入。

## 环境与首次检查

所有下方相对路径命令均先进入本项目根目录；不使用旧 `modern_genai` 路径或旧 venv。Python 3.12 是之前 CPU 模块验证所用版本。首先只需标准库即可检查迁移完整性：

```bash
python --version
python tools/verify_migration.py
python tools/verify_migration.py --full
```

第一条校验模式检查清单文件存在性/大小、当前 release SHA、批准范围、配置依赖和图片 SHA；`--full` 额外检查所有记录文件的 SHA。脚本可在别处通过其绝对路径调用，它会自行定位包根。校验通过不证明 CUDA 训练或新环境依赖已安装。

CPU 环境参考步骤（先使用你安装的 Python 3.12 创建虚拟环境）：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

激活后恢复模块检查环境并运行相关测试：

```bash
python -m pip install -r requirements-cpu-snapshot.txt
python -m pytest tests/test_approved_training_inputs.py tests/test_training_entries_cpu.py tests/test_opd_contract.py tests/test_losses_and_masks.py -q
```

[requirements-cpu-snapshot.txt](requirements-cpu-snapshot.txt) 是旧 CPU 环境的主要包版本记录，包含 PyTorch、Transformers、TRL、PEFT 等，**不是完整依赖锁或已实测的 CUDA 安装方案**。目标 GPU 环境应单独固定与驱动匹配的 PyTorch/CUDA 和框架版本并做 import/API 验收，不要据 CPU 包安装成功宣称 GPU 可运行。[.pilot_deps](.pilot_deps) 保留数学验证器及相关纯 Python 依赖/许可证；其中 SymPy 等仍依赖实际环境安装。

最近已保存的相关 CPU 检查为 20 passed，证据见 [迁移验收记录](迁移记录/迁移验收.md) 和 [机器可读结果](迁移记录/迁移验收.json)。本次 README 更新只运行文档与离线完整性检查，没有重跑模型测试或 GPU 训练。

## 当前 GPU 入口与输出

当前 GPU 工程入口：

```bash
python -m scripts.train_project2_opd --config configs/project2_opd_smoke.json
```

数据流是：学生 tokenizer 渲染一次提示 → 学生生成轨迹 → 相同完整 `input_ids`/`attention_mask` 直接交给冻结教师 → 在共同支持上用 FP32 概率归一化计算 response-token reverse KL → 更新学生。

数学系统提示固定为逐步推理并在 `\boxed{}` 中给最终答案；代码保留封存的接口提示。教师不能重新套自己的聊天模板，不能先 decode 再重新 tokenize 学生前缀。生成温度为 1、top-p 为 1，数学最多生成 1024、代码 1536 tokens；共同支持以外的采样 logits 被屏蔽。

默认输出 `runs/strict_v8_project2_opd_smoke/`，包含 `metrics.jsonl`、`run_identity.json` 和 `checkpoint-*`。checkpoint 内保存学生 adapter、`engine.pt`、文件哈希、优化器/调度器、Python/Torch/CUDA RNG、step 与每域读取位置。恢复时添加 `resume_from_checkpoint`，其他配置保持一致；已有 checkpoint 的目录不能当作新实验直接重跑。

正式模型命令需要目标 CUDA 环境和配置中的固定模型 revision 权重；本次文档更新不会自动执行这些训练/下载命令。当前入口检查单张可见 GPU，不能直接当成多卡配置。重新开始实验使用新输出目录；恢复必须核对完整 checkpoint 身份，不能只加载 adapter 就声称恢复了优化过程。

## 完整实验与评测计划

核心计划是 D0 原始学生、D1 数学单领域/数学教师、D2 代码单领域/代码教师、D3 混合数据/通用 7B 单教师、D4 同一混合数据/数学代码专业双教师路由。当前配置只是 D4 路由机制的 2 步工程原型，尚未交付这五个状态的完整正式配置和结果。

D3/D4 应共享学生起点、混合题目顺序、更新/生成预算与评测协议；D1/D2 是领域专门化诊断。固定总步数时，各领域的实际曝光可能不同，必须同时报告。数学评测计划为 MATH-500、GSM8K；代码为锁定版本 EvalPlus 的 HumanEval+/MBPP+。内部 dev 与官方最终测试分开。

需要记录各领域准确率、跨域退化、生成和训练 tokens、教师前向成本、切换耗时、峰值显存与词表外概率质量。完整 logits 的单卡内存是否可承受尚未实测，不能承诺 A100 某个显存档一定可运行。

## 建议继续工作的顺序

1. 从当前模型 revision、真实词表审计和数学/代码提示词契约核对配置，复查训练提示白名单及 dev 隔离。
2. 建立目标 GPU 环境，先测学生和各教师各自 dev 基线；先验收单教师真实批次，再测试两个领域切换。
3. 核查 teacher `eval/no_grad`、教师无梯度、学生有梯度、前缀/位置/response shift 一致、KL 有限和真实移除概率质量。
4. 保存后同配置恢复至少一步，核对 optimizer、scheduler、RNG、领域位置和 checkpoint 哈希；记录内存及切换开销。
5. 补齐 D0～D4 正式配置、代码任务语义 checker 和官方评测接线，再按冻结预算训练；Top-K、熵融合或多卡属于后续独立扩展。

每个阶段保存配置、数据/模型身份、原始预测、评分版本、恢复记录、显存和成本。将“已实现代码”“CPU 通过”“真实模型跑通”“最终能力提升”分别陈述；未运行的检查保留未验证状态。

## 历史材料、迁移与后续维护

本项目不要求先运行数学 SFT，也不依赖另一个项目的 `runs/.../merged/`。当前 OPD 从配置中的固定 1.5B 学生起点运行。

共用 API 审核缓存的 15119 个小文件完整保存在 [shared_api_cache.zip](reports/data_release_preparation_20261024/shared_api_cache.zip)。现有训练/离线校验无需解压；需要查询原始请求或恢复历史数据审核时再解压：

```bash
python -m zipfile -e reports/data_release_preparation_20261024/shared_api_cache.zip reports/data_release_preparation_20261024/shared_api_cache
```

历史成本账本是审计事实，不是新任务的调用额度；恢复 API 数据流水线前应依据当前任务的实际范围、密钥与预算。不要自动重新生成或扩量已封存数据。

[migration_manifest.json](migration_manifest.json) 跟踪迁移负载文件的大小和哈希，数据 release 的 manifest 另行保护训练批准。本次只同步 README/AGENTS 对应的文档清单项，并在 `迁移记录/README交接补充前_20261024.md`、`迁移记录/migration_manifest_交接补充前_20261024.json` 保留旧版本。以后合法修改文档/代码应说明变更和相关校验；不能用批量重算训练数据 SHA 的方式绕过数据准入。

## 新对话可直接使用的开场说明

> 请先读取本项目 AGENTS.md、README.md、configs/ 和 data/portable_release/manifest.json。结合实际源码说明项目目标、已完成工作、未验证事项和建议的下一步。继续本项目主线，保留既有数据与审核记录；不要把历史方案或 2 步 smoke 写成已完成的正式训练结果。
