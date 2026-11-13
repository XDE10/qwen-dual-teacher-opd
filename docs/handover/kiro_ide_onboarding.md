# 交接：从 Kiro Web 转到本地 Kiro IDE（2026-11-09）

写给负责人和本地 Kiro IDE 里的 agent。仓库根目录的 `AGENTS.md` 会被 Kiro 自动加载；本文件是接手时的起点。

给 WSL2 的命令都能在 zsh 里直接粘贴：不写行尾 `#` 注释，读密码用 `printf ... && read -rs`；`colab console` 里是 VM 上的 bash。

## 1. 当前状态

| 阶段 | 状态 | 结果位置 |
| --- | --- | --- |
| 第一次上卡（smoke + 教师探针） | 完成 | `results` 分支 `first_session/20261109T065604Z/`；报告 [reports/g4_first_session_20261109](../../reports/g4_first_session_20261109/README.md) |
| 第二次上卡：探针细查 + 三个学习率各 30 步 | 完成 | `results` 分支 `session2/20261109T104721Z/`；报告 [reports/g4_session2_lr_sweep_20261109](../../reports/g4_session2_lr_sweep_20261109/README.md) |
| 第二次上卡：`eval=d0`（未训练学生的 greedy 评测）+ `teacher-outputs`（两位教师在训练题上的输出） | **状态不明**：`eval=d0` 13:20 UTC 开始；最后一次同步在 13:50 UTC，之后每 20 分钟一次的同步没有再出现（14:59 UTC 查看）。先按第 4.1 节检查 | 跑完时在同一目录的 `runs/eval/d0/`、`runs/teacher_outputs/` |
| D0 的代码分数（本地 Docker） | 未开始 | 第 4.2 节 |
| D4 80 步、D1/D2/D3、blocked、SFT、warm start | 未开始 | 第 4 节 |

已定下的事：

- **主实验用 `d4_dual`（学习率 3e-5）**，从 checkpoint-30 续跑到 80 步。理由见第二次上卡报告的"决定"一节。
- **稳定性关卡改为 v2**（`scripts/summarize_run_v3.py`）：以第 1 步为基线，选峰性看"学生熵 ÷ 教师熵"的相对变化，学生熵的绝对下降只作提示。三组 30 步在 v2 下都通过。
- `proximal_teacher_alpha` 不用于主实验；`d4_prox07`（α=0.7）只作有条件触发的消融（第 4.5 节）。

待定，以及据以决定的数据：

- 学习率是否改为 1e-4：看第三次上卡的 dev 评测（第 4.4 节）。
- OPD 的提升是不是真的：看 greedy dev 与 D0 的比较。在线准确率只是训练题上 T=1 的遥测，不能代替。

## 2. 资产在哪里

| 东西 | 位置 |
| --- | --- |
| 代码、配置、文档 | GitHub `XDE10/qwen-dual-teacher-opd` 的 `feat/opd-v3-mopd-data-v5` 分支，[PR #1](https://github.com/XDE10/qwen-dual-teacher-opd/pull/1)（不合并，等实验跑完） |
| 运行结果（日志、JSON、评测输出、代码分数） | 同一仓库的 `results` 分支。永不合并进代码分支；只放文本，不放权重 |
| checkpoint、完整 run 目录、教师输出 | Google Drive，rclone 远端 `mydrive:`，`opd_runs/<任务名>/<UTC 时间>/runs/` |
| 本地仓库 | WSL2 `/home/xde10/projects/qwen-dual-teacher-opd`（Windows：`\\wsl.localhost\Ubuntu\home\xde10\projects\qwen-dual-teacher-opd`） |
| GitHub 凭据（读写，只限本仓库） | `~/.config/opd/gh_credentials`（`https://XDE10:TOKEN@github.com`，权限 600） |
| rclone 配置 | `~/.config/rclone/rclone.conf` |
| Colab | WSL2 里的 google-colab-cli 0.7.5，会话名 `opd`；操作手册 [docs/colab_cli_wsl2.md](../colab_cli_wsl2.md) |
| Kiro Web 里存的两条长期偏好（结果走 GitHub、zsh 命令规则） | 已写进 `AGENTS.md` 第 7 节；也可以在 Kiro 的 Agent Config > Cloud Config 打开 "Apply your cloud configuration to local sessions" |

Kiro Web 沙箱已清查：工作区干净，没有未推送的提交、stash 或分支。唯一被忽略的目录是 `data/code_benchmarks/`，即用 `scripts/restore_official_benchmarks.py` 下载并校验 sha256 的 HumanEval+/MBPP+ 官方文件，可以随时恢复。

## 3. 本地初始化（只做一次）

### 3.1 Kiro IDE

- 打开文件夹 `\\wsl.localhost\Ubuntu\home\xde10\projects\qwen-dual-teacher-opd`。Kiro 文档没有专门写 WSL 文件夹的用法；如果文件监视或速度有问题，先改下一条，再看情况。
- 把默认终端配置（terminal profile）设为 WSL（Ubuntu）。Kiro 在 Windows 上用默认终端配置执行 agent 的命令，所以 agent 的命令会跑在 WSL 的 zsh 里，同样要守 zsh 规则。
- `AGENTS.md`（根目录和子目录里的）会被自动加载，不需要建 `.kiro/`。

### 3.2 仓库与 Python 环境

```zsh
cd ~/projects/qwen-dual-teacher-opd
git status --short --branch
git fetch origin
git switch feat/opd-v3-mopd-data-v5
git pull --ff-only
git log --oneline -1
```

`git status` 如果显示改动，先停下处理。仓库根目录如果还有 `first_session_20261109T030054Z/` 和 `opd_first_session.tar.gz`，先移出去，否则 `tools/verify_repository.py` 会报"unexpected top-level entry"。它们在 GitHub 和 Drive 上都有：

```zsh
mkdir -p ~/opd_runs
mv first_session_20261109T030054Z opd_first_session.tar.gz ~/opd_runs/
```

建虚拟环境（Ubuntu 的系统 Python 不允许直接 `pip install`）：

```zsh
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv ~/.venvs/opd
~/.venvs/opd/bin/python -m pip install --upgrade pip
~/.venvs/opd/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
~/.venvs/opd/bin/python -m pip install -r requirements/wsl2-cpu.txt
```

离线验证：

```zsh
~/.venvs/opd/bin/python -m scripts.restore_official_benchmarks
~/.venvs/opd/bin/python -m pytest -q
~/.venvs/opd/bin/python tools/verify_repository.py
~/.venvs/opd/bin/python tools/verify_project2_release_v5.py
```

应该看到：`189 passed, 1 skipped`（这套依赖在 Linux 沙箱里用 Python 3.11 实测过），`"layout_ok": true`，以及 `"errors": []`。

### 3.3 凭据与连接

```zsh
ls -l ~/.config/opd/gh_credentials ~/.config/rclone/rclone.conf
~/.venvs/opd/bin/python scripts/publish_results_v3.py --check --credentials ~/.config/opd/gh_credentials
rclone lsd mydrive:opd_runs
colab version
```

依次应该是：两个文件都存在且权限为 `-rw-------`；输出 `accepts pushes`；能列出 `first_session`、`session2`；`Version: 0.7.5`。

### 3.4 本地 Docker（代码评测用）

代码只能在 Docker 里执行（`AGENTS.md` 硬约束）。WSL2 里可以用 Docker Desktop 的 WSL 集成，也可以在 Ubuntu 里装 Docker Engine。

```zsh
docker info --format '{{.ServerVersion}}'
docker image inspect --format '{{.Id}}' sha256:3b0c23d092b8ff94262fd3518cf8250c4b1d65c014d6090eea3b23383145201e
```

第二条是内部 dev 评测记录在案的沙箱镜像（`reports/code_v2_sandbox_environment.json`），它当时报告 Python 3.8.20（GCC 12.2.0）。

- 用 Docker Desktop 且没删过这个镜像：直接能找到。
- 找不到：拉 `python:3.8-slim-bookworm`，先确认版本，再把它的 ID 存起来。之后的分数会记录这个新 ID；它和历史记录不是同一个镜像，报告里要注明。

```zsh
docker pull python:3.8-slim-bookworm
docker run --rm python:3.8-slim-bookworm python -c 'import sys; print(sys.version)'
docker image inspect --format '{{.Id}}' python:3.8-slim-bookworm > ~/.config/opd/code_image
```

官方 EvalPlus 评测要一个装好 evalplus 0.3.1 的镜像（只需构建一次，要联网；评测时容器没有网络）：

```zsh
docker build -t opd-evalplus:0.3.1 -f scripts/docker/evalplus-0.3.1.Dockerfile scripts/docker
docker image inspect --format '{{.Id}}' opd-evalplus:0.3.1 > ~/.config/opd/evalplus_image
cat ~/.config/opd/evalplus_image
```

`scripts/score_code_local_v3.py` 会自动读取 `~/.config/opd/code_image` 和 `~/.config/opd/evalplus_image`。这个镜像和打分脚本都还没在真实 Docker 上跑过；第一次运行时要看 `evaluate.log`，确认结果文件确实生成了。

## 4. 下一步

### 4.1 先查第二次上卡的状态

`results` 分支上最后一次同步是 13:50 UTC，那时 `eval=d0`（13:20 开始）还在生成。`generate_eval_v3` 要等全部评测集生成完才写文件，所以那时还没有 `runs/eval/` 是正常的。不正常的是之后约 14:10、14:30、14:50 的定期同步都没有出现：即使没有新文件，每次同步也会更新 `PUBLISHED.json`、产生一个提交。可能是 VM 被回收了（例如计算单元用完），也可能是 VM 还在、只是保存卡住了。

先看 GitHub 上有没有新结果：

```zsh
cd ~/projects/qwen-dual-teacher-opd
git fetch origin results
git log -3 --format='%h %cI %s' origin/results
git show origin/results:session2/20261109T104721Z/runs/sessions/20261109T104721Z/steps.jsonl
git show origin/results:session2/20261109T104721Z/PUBLISHED.json | head -12
```

跑完时 `steps.jsonl` 有 6 行，最后两行是 `eval=d0` 和 `teacher-outputs`，`status` 都是 0；`PUBLISHED.json` 里 `"state": "finished"`、`"exit_status": 0`。

仍停在 13:50 时，查 VM、Drive 和计算单元：

```zsh
colab status -s opd
colab download -s opd /content/jobs/session2/output.log ~/session2.log && tail -n 40 ~/session2.log
rclone lsf mydrive:opd_runs/session2/20261109T104721Z/runs
colab usage
```

- **VM 还在，任务还在跑**：等它跑完。日志里有 `periodic save failed`、`Drive copy incomplete` 或 `publishing to GitHub failed` 时，按其中的原因处理。
- **VM 还在，任务已结束**（日志里有 `[colab_job] command exited with status N; saving results`，之后是保存的输出）：结果应该已在 Drive 和 GitHub 上；GitHub 上没有时，看日志里推送失败的原因。
- **VM 已经不在**：三组训练的全部 checkpoint 在 13:50 那次同步时已经在 Drive 上（那次 `runs` 的增量是 0 B，日志显示 `saved to Drive`）。`eval=d0` 和 `teacher-outputs` 的结果没有保存下来，改在第三次上卡重跑，用第 4.3 节"新 VM"的第二条命令。如果 `colab usage` 显示计算单元不够，先补足，再按第 4.5 节排期。

- D0 的数学分数：`git show origin/results:session2/20261109T104721Z/runs/eval/d0/math_dev.summary.json`。
- 每个 summary 的 `seconds` 字段是实测评测耗时，用它替换下面的估计。

跑完后，VM 还在的话可以直接接第三次上卡（省去从 Drive 恢复）；不用就 `colab stop -s opd`。

### 4.2 D0 的代码分数（本地，无需 GPU）

```zsh
cd ~/projects/qwen-dual-teacher-opd
~/.venvs/opd/bin/python scripts/score_code_local_v3.py session2/20261109T104721Z/runs/eval/d0 --publish --credentials ~/.config/opd/gh_credentials
```

- 它从 `results` 分支取出 `code_dev`、`humanevalplus`、`mbppplus` 三个候选文件，在 Docker 里打分。
- 结果写到 `~/opd_scores/session2-20261109T104721Z/d0/scores.json`，并推到 `results` 分支的 `code_scores/session2-20261109T104721Z-d0/`。
- 没有 EvalPlus 镜像时，只给 code_dev 打分，另外两个标为 skipped。
- 内部 dev 分数和官方 EvalPlus 分数分开报告，不能合并。

### 4.3 第三次上卡（约 3 小时）

内容：三个 checkpoint-30 的 dev 评测 → SFT 基线（SeqKD-Mix）及其 dev 评测 → `d4_dual` 续跑到 80 步 → 第 80 步的完整评测。

**同一台 VM（第二次上卡刚跑完）**，先更新 VM 上的代码：

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
import subprocess
r = subprocess.run(['git', '-C', '/content/project2', 'pull', '--ff-only'], capture_output=True, text=True)
print(r.stdout + r.stderr, end='')
EOF
```

再启动：

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py session3 --publish --drive mydrive:opd_runs --sync-every 20 --save runs=/content/runs -- bash scripts/colab_session_v3.sh dev=d4_dual@30 dev=d4_dual_lr1e-4@30 dev=d4_dual_lr1e-5@30 sft dev=sft train=d4_dual eval=d4_dual
EOF
```

**新 VM**：先按 [docs/colab_cli_wsl2.md](../colab_cli_wsl2.md) 的"每次开卡"第 1、2 步准备好（开 G4、上传凭据、挂 Drive、克隆），再启动。步骤前面加上从 Drive 恢复。第二次上卡完整跑完（Drive 上有 `runs/teacher_outputs`）时：

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py session3 --publish --drive mydrive:opd_runs --sync-every 20 --save runs=/content/runs -- bash scripts/colab_session_v3.sh restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-4 restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-5 restore=mydrive:opd_runs/session2/20261109T104721Z/runs/teacher_outputs dev=d4_dual@30 dev=d4_dual_lr1e-4@30 dev=d4_dual_lr1e-5@30 sft dev=sft train=d4_dual eval=d4_dual
EOF
```

第二次上卡的 `eval=d0` / `teacher-outputs` 没有跑完（第 4.1 节的"VM 已经不在"）时，不恢复 `teacher_outputs`，改为重跑这两步。这一次要多用约 1.5–2 小时（估计，未实测）：

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py session3 --publish --drive mydrive:opd_runs --sync-every 20 --save runs=/content/runs -- bash scripts/colab_session_v3.sh restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-4 restore=mydrive:opd_runs/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-5 eval=d0 teacher-outputs dev=d4_dual@30 dev=d4_dual_lr1e-4@30 dev=d4_dual_lr1e-5@30 sft dev=sft train=d4_dual eval=d4_dual
EOF
```

这时 D0 的代码分数在第三次上卡之后打：把第 4.2 节命令里的 `session2/20261109T104721Z/runs/eval/d0` 换成 `$S3/runs/eval/d0`（`S3` 的取法见本节末尾）。

耗时估计（根据第二次上卡的实测步耗推算，评测部分还要用 D0 的实测时间校正）：

| 步骤 | 估计 |
| --- | --- |
| 环境引导 + 下载模型 | 约 5 分钟 |
| 从 Drive 恢复（仅新 VM） | 约 5 分钟（约 2 GB） |
| 三个 checkpoint-30 的 dev 评测（每个 456 题） | 共 20–25 分钟 |
| `sft`（约 4900 条教师输出、1 个 epoch、约 77 步） | 10–20 分钟 |
| `dev=sft` | 约 8 分钟 |
| `train=d4_dual` 第 31–80 步 | 约 80 分钟（50 × 95 s） |
| `eval=d4_dual`（6 个测试集，2817 题） | 40–45 分钟（第二次上卡的 `eval=d0` 开始 30 分钟后还没生成完） |

跑完后，在本地给代码打分。第一行取 `results` 分支上最新的 `session3/<UTC 时间>` 目录，第二行确认它：

```zsh
cd ~/projects/qwen-dual-teacher-opd
git fetch origin results
S3=$(git ls-tree --name-only origin/results session3/ | tail -n 1)
echo $S3
~/.venvs/opd/bin/python scripts/score_code_local_v3.py $S3/runs/eval/d4_dual-checkpoint-30-dev $S3/runs/eval/d4_dual_lr1e-4-checkpoint-30-dev $S3/runs/eval/d4_dual_lr1e-5-checkpoint-30-dev $S3/runs/eval/sft-final_adapter-dev $S3/runs/eval/d4_dual-checkpoint-80 --publish --credentials ~/.config/opd/gh_credentials
```

### 4.4 第三次上卡之后的判断

1. **OPD 有没有真实提升。** 比较 `d4_dual-checkpoint-30-dev`、`d4_dual-checkpoint-80` 与 D0 在 math_dev（Math-Verify 为主，保守判分器同报）和 code_dev 上的成绩。
   - 第 80 步不高于 D0：先不跑完整矩阵，改查原因（先看 warm start 和 `d4_prox07`）。
2. **学习率。** 只有当 1e-4@30 的 math_dev 比 3e-5@30 高 5 个点以上（256 题中约 13 题），且 code_dev 不低时，才改用 1e-4。
   - 改用时：续跑 `train=d4_dual_lr1e-4`，并补建 D1/D2/D3/blocked/warm start 的 `_lr1e-4` 配置。新配置要让 `tests/test_opd_v3.py` 的"只差声明因素"测试通过。
   - 否则保持 3e-5。
3. **80 步的关卡（`d4_dual/summary.json` 的 `gate`）。** 出现以下任一情况，就把 `d4_prox07` 提前到下一次上卡：
   - math 的 `entropy_vs_teacher` 低于 0.7（第 30 步是 0.74）；
   - math EOS 率比第 1 步低 0.10 以上；
   - math_dev 在第 30→80 步之间没有提高。
4. **SFT 基线。** `sft-final_adapter-dev` 与 D4 的 dev 成绩比较，回答"为什么不用离线 SFT 数据混合"。

dev 只用来做以上选择；测试集（MATH-500、GSM8K test、HumanEval+、MBPP+）只用于最终报告，不用来挑模型。

### 4.5 之后的排期（每次上卡一条命令，同样带 `--publish --drive mydrive:opd_runs --sync-every 20 --save runs=/content/runs`）

| 次序 | 步骤（`colab_session_v3.sh` 的参数） | 回答的问题 | GPU 估计 |
| --- | --- | --- | --- |
| 第四次 | `train=d3_general eval=d3_general train=d1_math eval=d1_math` | 专家教师 vs 通用教师；单数学教师与跨域保持 | 约 4.3 h |
| 第五次 | `train=d2_code eval=d2_code train=d4_warmstart eval=d4_warmstart` | 单代码教师；先 SFT 再 OPD | 约 4.6 h |
| 第六次 | `train=d4_blocked eval=d4_blocked`，可选 `train=d4_prox07 eval=d4_prox07` | 混合 vs 分块调度；近端教师 | 约 2.5 h（加 prox 约 5.3 h） |

- 每次上卡后都用第 4.3 节的方式给 `eval/` 下的新目录打代码分数。
- 新 VM 上需要的前置目录：
  - `d4_warmstart` 需要 SFT 的 adapter：在步骤最前面加 `restore=mydrive:opd_runs/session3/<UTC 时间>/runs/sft_v3_seqkd_mix`，时间戳用 `echo $S3` 显示的那个；
  - D1/D2/D3/blocked/prox 都从头训练，不需要恢复。
- 合计约 14–17 GPU 小时（不含已用的两次；每次完整评测按 40–45 分钟算）。如果第二次上卡的 `eval=d0` / `teacher-outputs` 要重跑，再加 1.5–2 小时。
- G4 每小时消耗多少计算单元没有公开数字，开卡前后各运行一次 `colab usage` 记下来。
- 日程：
  - 核心结果（D0、D4、SFT、D3、D1、D2）争取在 11/16 前完成；
  - 消融在 11/23 前完成；
  - 其余时间留给分析、写作和重跑，截止 12/3。

`d4_pcgrad`、`d4_pg`、`d4_topk` 暂不排。第二次上卡后期的 math–code 梯度余弦约为 0，没有冲突，PCGrad 不太可能起作用；PG 和 top-k 只在有余量时作为 MOPD 损失形式的对照。

### 4.6 读结果（给本地 agent）

```zsh
cd ~/projects/qwen-dual-teacher-opd
git fetch origin results
git ls-tree -r --name-only origin/results session2/20261109T104721Z | head -50
mkdir -p ~/opd_results
git archive origin/results session2 | tar -x -C ~/opd_results
~/.venvs/opd/bin/python -m scripts.compare_runs_v3 --run 3e-5=/home/xde10/opd_results/session2/20261109T104721Z/runs/opd_v3_d4_dual --run 1e-4=/home/xde10/opd_results/session2/20261109T104721Z/runs/opd_v3_d4_dual_lr1e-4
~/.venvs/opd/bin/python -m scripts.summarize_run_v3 --run-dir /home/xde10/opd_results/session2/20261109T104721Z/runs/opd_v3_d4_dual
```

- `compare_runs_v3` 输出每个 run 第一步与最后一个窗口的对比，以及同一道题上的配对在线准确率（McNemar）。
- `summarize_run_v3` 重新计算两版关卡：`gate` 是 v2，`gate_v1_first_window` 是 v1。
- `results` 分支的历史会越来越大；平时只读最新状态，用 `git fetch origin results` 即可，不需要完整克隆。

## 5. 规则与防呆

- **续训依赖运行身份。** 以下任何一项与写 checkpoint 时不同，续训都会直接报错：
  - `src/training/runtime_v3.py` 里 `SOURCES` 列出的 11 个源文件（训练循环、损失、采样、契约等）；
  - run 对应的配置文件；
  - `output_dir`，所以 `OPD_OUTPUT_ROOT` 固定用 `/content/runs`；
  - torch/transformers/peft 版本。

  `d4_dual` 的 checkpoint-30 是在 `bc02083` 写的。在它续跑到 80 步之前，不要改这些文件。改动前可以这样检查，输出为空才行：
  ```zsh
  ~/.venvs/opd/bin/python -c 'from src.training.runtime_v3 import SOURCES; print(chr(10).join(SOURCES))' > /tmp/opd_sources.txt
  git diff --stat bc02083 HEAD -- $(cat /tmp/opd_sources.txt)
  ```
- **只用 dev 做选择。** 测试集只出现在最终报告里。
- **代码只在 Docker 里执行。** 没有 Docker 时停下，不在宿主机跑候选代码。
- **Colab 环境。** 引导脚本会卸载 torchvision、torchaudio、torchao，不要把它们加回依赖。vLLM（`INSTALL_VLLM=1`）会带回自己的版本，目前评测走 HF，不需要 vLLM。
- **命令与脚本。** 交互命令守 zsh 规则；复杂逻辑写成脚本，用 `bash` 或 `python` 调用，不在命令里嵌套多层引号。每次 `colab exec` 都加 `--timeout 86400`。
- **结果回传。** 结果一律经 `colab_job_v3.py --publish --drive` 回传：文本进 `results` 分支，大文件进 Drive 的 `opd_runs/`。不让负责人手动下载或上传。

## 6. 尚未验证

- GPU 上从 checkpoint 续训、`sft` 训练、`eval=`/`dev=` 带 adapter 的评测（CPU 测试和替身命令测过流程）。
- 本地打分：`score_code_local_v3.py` 只用假的 Docker 测过流程，EvalPlus 镜像从未构建过。
- 任何能力提升。D0～D4 都还没有评测分数。
