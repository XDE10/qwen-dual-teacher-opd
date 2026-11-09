# Colab CLI 运行规范（WSL2 + zsh，google-colab-cli 0.7.5）

在 Colab G4 上跑 opd-v3 的标准步骤。合并了负责人整理的《Colab G4 实例训练初始化与避坑规范 v2.0》：rclone 挂 Drive、bash 和 zsh 的 `read` 差异、预装库冲突、`--again`、产物回传。另外，任务结束时会自动把结果存到 Drive 和 GitHub。

状态：`scripts/colab_job_v3.py`、`scripts/publish_results_v3.py` 和两个 shell 脚本在本地 Linux 上测过。测试用了假的 `rclone`、本地 git 仓库和 zsh 5.8。负责人已在 G4 上手动跑完一次第一次上卡；自动保存到 Drive 和推送到 GitHub 还没在 Colab 上跑过。

## 环境约定

| 项目 | 约定 |
| --- | --- |
| 本机 | WSL2 Ubuntu，shell 是 zsh。项目在 `~/projects/qwen-dual-teacher-opd`（Windows 下是 `\\wsl.localhost\Ubuntu\home\xde10\projects\qwen-dual-teacher-opd`），`colab` 也装在 WSL2 里 |
| VM 终端 | `colab console` 打开的是 VM 上的 bash |
| Drive | rclone 远端 `mydrive:`，VM 上挂到 `/content/mydrive`，结果放 `mydrive:opd_runs/` |
| 运行输出 | VM 本地 SSD 上的 `/content/runs`（`OPD_OUTPUT_ROOT` 的默认值） |
| 结果回传 | 任务结束时，`colab_job_v3.py` 把产物原样复制到 `mydrive:opd_runs/<任务名>/<UTC 时间>/`，并把日志和 JSON 推到 GitHub `results` 分支的 `<任务名>/<UTC 时间>/`。Kiro 直接从 `results` 分支读，不用下载再上传 |

给 WSL2 的命令都按 zsh 写，可以直接粘贴：

- 命令行里不写 `#` 注释。zsh 默认不把交互输入里的 `#` 当注释，会当成参数传给命令。
- 输入密码用 `printf '提示: ' && read -rs 变量`。zsh 的 `read -p` 意思是从协进程读，`read -rsp` 会报错。bash 写法 `read -rsp '提示' 变量` 只在 VM 的 `colab console` 里用。
- 含 `*`、`?`、`[`、`]` 的参数加引号：zsh 匹配不到文件时会报错，不会把参数原样传下去。
- `~` 只写在单独参数的开头。zsh 不展开 `--opt=~/x` 里的 `~`。

## 一次性准备（WSL2）

### GitHub token（读写，只给这个仓库）

在 GitHub 的 Settings → Developer settings → Personal access tokens → Fine-grained tokens 里新建 token，或者编辑之前那个只读 token。改权限不会改 token 值，已经写好的凭据文件不用重写。

- Repository access：Only select repositories，只选 `XDE10/qwen-dual-teacher-opd`
- Repository permissions：Contents 选 Read and write。Metadata 会自动设为 Read-only，其他保持 No access
- Expiration：选到 12 月中旬，项目结束后删掉

风险：这个 token 能推送仓库的任何分支，包括 `main`。私有仓库在 GitHub Free 下不能设分支保护；有 GitHub Pro（例如学生包）时，可以在仓库 Settings → Rules 给 `main` 加规则。脚本只往 `results` 分支推。token 只放两处：WSL2 的 `~/.config/opd/gh_credentials`，以及 VM 的 `/content/gh_credentials`（VM 释放后就没了）。不要放在 `/mnt/c`、`/mnt/d` 下，那里 `chmod` 不生效。

写凭据文件要分三次粘贴。第二行回车后输入 token（屏幕上不显示），再回车：

```zsh
mkdir -p ~/.config/opd && chmod 700 ~/.config/opd
printf 'GitHub PAT: ' && read -rs PAT && echo
printf 'https://XDE10:%s@github.com\n' "$PAT" > ~/.config/opd/gh_credentials && chmod 600 ~/.config/opd/gh_credentials && unset PAT
```

### Hugging Face token（可选，加快下载）

```zsh
printf 'HF token: ' && read -rs HFT && echo
printf '%s' "$HFT" > ~/.config/opd/hf_token && chmod 600 ~/.config/opd/hf_token && unset HFT
```

### 本地仓库切到这个分支

```zsh
cd ~/projects/qwen-dual-teacher-opd
git status --short --branch
git fetch origin
git switch feat/opd-v3-mopd-data-v5
git pull --ff-only
python3 scripts/publish_results_v3.py --check --credentials ~/.config/opd/gh_credentials
```

如果 `git status` 显示有未提交的改动，先处理掉，否则 `git switch` 可能拒绝切换。最后一行做一次 `git push --dry-run`，不会建分支；输出 `accepts pushes` 说明 token 能推送。

## 补发已经下载到本机的结果（WSL2）

适用于在这次改动之前跑完、已经下载到本机的结果，例如第一次上卡的 `opd_first_session.tar.gz`：

```zsh
cd ~/projects/qwen-dual-teacher-opd
python3 scripts/publish_results_v3.py --name first_session --source report=/home/xde10/opd_first_session.tar.gz --credentials ~/.config/opd/gh_credentials
```

`report=` 后面换成压缩包的实际位置。也可以是解压后的目录，或者 Windows 下的文件夹（例如 `report=/mnt/d/Users/Mr.Z/Desktop/opd_runs1`）。本机找不到压缩包时，用 `rclone copy mydrive:opd_runs/opd_first_session.tar.gz ~/` 从 Drive 取回。加 `--dry-run` 只列出会发布的文件，不推送。输出最后一行是 GitHub 上的链接。

## 每次开卡

### 1. 开卡，上传凭据（WSL2）

```zsh
colab new -s opd --gpu G4
colab status -s opd
colab upload -s opd ~/.config/rclone/rclone.conf /content/.colab-drive-rclone-upload.conf
colab upload -s opd ~/.config/opd/gh_credentials /content/gh_credentials
```

`colab status` 要显示 `Hardware: G4`：`--gpu` 写错时，CLI 会静默改成 A100。分配被拒时改用 `--gpu A100`。有 HF token 时再执行 `colab upload -s opd ~/.config/opd/hf_token /content/hf_token`。

### 2. VM 终端：挂 Drive、配 git、克隆（bash）

```zsh
colab console -s opd
```

出现 `root@...:/content#` 后粘贴：

```bash
mkdir -p ~/.config/rclone
mv /content/.colab-drive-rclone-upload.conf ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
command -v rclone >/dev/null || { apt-get update && apt-get install -y rclone fuse3; }
mkdir -p /content/mydrive
rclone mount mydrive: /content/mydrive --vfs-cache-mode writes --vfs-cache-max-size 10G --daemon
mountpoint /content/mydrive
chmod 600 /content/gh_credentials
git config --global credential.helper 'store --file=/content/gh_credentials'
git clone -b feat/opd-v3-mopd-data-v5 https://github.com/XDE10/qwen-dual-teacher-opd.git /content/project2
git -C /content/project2 log --oneline -1
```

`mountpoint` 要输出 `/content/mydrive is a mountpoint`。如果上传了 HF token，再执行：

```bash
mkdir -p ~/.cache/huggingface && mv /content/hf_token ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
```

最后输入 `exit` 回到 WSL2。

如果不上传 `gh_credentials`，也可以在这个终端里用 bash 写法输入 token：先执行 `read -rsp 'GitHub PAT: ' PAT; echo`，再执行 `printf 'https://XDE10:%s@github.com\n' "$PAT" > /content/gh_credentials; unset PAT`。

### 3. 运行任务（WSL2）

第一次上卡：

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py first_session --publish --drive mydrive:opd_runs --save report=/content/opd_first_session.tar.gz -- bash scripts/colab_first_session_v3.sh
EOF
```

- `--publish`：任务结束后（失败也一样），把 `job/`（`output.log`、`command.json`）和 `--save` 里的日志、JSON 推到 `results` 分支的 `first_session/<UTC 时间>/`。启动前会先试推一次（`git push --dry-run`）；token 不能推送时直接报错，任务不会启动。
- `--drive mydrive:opd_runs`：任务结束后，用 `rclone copy` 把 job 目录和 `--save` 的内容原样复制到 `mydrive:opd_runs/first_session/<UTC 时间>/`。`rclone copy` 返回时文件已经在 Drive 上。经 `/content/mydrive` 挂载点写入的文件是异步上传的，VM 释放前可能还没传完，所以脚本不经过挂载点。
- `--save report=PATH`：要保存的文件、目录或压缩包。`report` 是它在 Drive 和 GitHub 上的子目录名。
- 任务在 VM 上独立运行，关终端、Windows 睡眠、按 Ctrl-C 都不影响它。在日志出现 `[colab_job] first_session finished with exit status N` 之前，不要运行 `colab stop` 或 `colab restart-kernel`，也不要对同一会话再开 `colab exec`，否则它会排在任务后面。
- 结束时日志里有 `saved to Drive: ...` 和 GitHub 链接，之后告诉 Kiro“跑完了”即可。

在同一台 VM 上重跑同名任务时，要在任务名后加 `--again`（旧的任务目录改名保留）；不加的话，只会报告上次的结果。断线后重新执行同一条命令，会接着显示日志。

### 4. 看进度（另开一个 WSL2 终端）

```zsh
colab download -s opd /content/jobs/first_session/output.log ./first_session.log && tail -n 40 ./first_session.log
```

### 5. 收尾

```zsh
colab stop -s opd
```

## 长训练

```zsh
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py d4_dual --publish --drive mydrive:opd_runs --sync-every 30 --save run=/content/runs/opd_v3_d4_dual -- python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_d4_dual.json --output-root /content/runs
EOF
```

`--sync-every 30` 让任务每 30 分钟保存一次。Drive 上保存整个 run 目录，包括 checkpoint，只传有变化的文件。GitHub 上只保存日志和 JSON，在同一个文件夹里原地更新。VM 中途被回收时，Drive 上还有最近一次复制的 checkpoint。checkpoint 带 sha256 清单，没复制完整的 checkpoint 在续训时会被拒绝。

## Colab 环境的已知问题（脚本已处理）

- `scripts/colab_bootstrap_v3.sh` 装好 torch 2.8.0（cu128）后，会卸载 Colab 预装的 torchvision、torchaudio、torchao。它们是按 Colab 自带的 torch 编译的：torchvision 会报 `torchvision::nms` 不存在，导致 transformers 加载模型和 pytest 收集测试失败；PEFT 加 LoRA 时会拒绝 torchao 0.10（要求 ≥ 0.16）。规范 v2.0 里手动卸载的那一步不用再做。`INSTALL_VLLM=1` 时，vLLM 会装回与 torch 2.8.0 匹配的 torchvision/torchaudio。
- Python 3.13 可以用，引导脚本接受 3.10–3.13。
- 环境记录（`pip-freeze.txt`、`imports.txt`、`torch.txt`、`cpu-tests.txt` 等）放在第一次上卡报告的 `environment/` 里，会一起推到 GitHub。
- 输出目录设在 `/content/drive/...` 或 `/content/mydrive/...` 下、而对应位置没挂载时，脚本会直接报错退出，不会把文件写进 VM 本地的同名目录。

## 0.7.5 的限制

| 限制 | 后果 | 做法 |
| --- | --- | --- |
| `colab exec` 的 `--timeout` 默认 30 秒，是整次执行的期限。0.7.5 固定的 jupyter-kernel-client 0.9.0 不执行这个期限，但过了期限后，本地 `colab` 进程会占满一个 CPU 核（[#183](https://github.com/googlecolab/google-colab-cli/issues/183)） | 长任务期间本机一个核满载；换成新版库后会直接超时失败 | 每次 `colab exec` 都加 `--timeout 86400` |
| 代码在 VM 的 Jupyter 内核里执行，同一会话的 `colab exec` 按顺序排队 | 任务运行时再开 `colab exec`，它会一直等 | 任务运行时只用 `colab download` 看日志 |
| `--env KEY=VALUE` 会把值拼进要执行的代码，代码原样写进本地 `~/.config/colab-cli/history/<会话>.jsonl` | 令牌会明文写进本地文件 | 令牌用 `colab upload` 传文件 |
| Colab Secrets（`userdata.get`）需要网页前端 | CLI 里拿不到 | 同上 |
| `--gpu` 的值写错时，静默改成 A100 | 可能开错卡 | 开完用 `colab status` 确认 |
| `colab download` 只能下载单个文件 | 不能直接下载整个报告目录 | 任务结束时自动存到 Drive 和 GitHub |
