# 用 Colab CLI（WSL2）跑 opd-v3

适用于 [google-colab-cli](https://github.com/googlecolab/google-colab-cli) 0.7.5，命令名是 `colab`。CLI 只支持 Linux 和 macOS，所以在 WSL2 里用。下面的命令都在 WSL2 终端里执行，`opd` 是会话名，可以换成别的。

状态：`scripts/colab_job_v3.py` 和 `scripts/colab_first_session_v3.sh` 的流程只在本地 Linux 上用替身命令测过，覆盖了启动、跟随日志、跟随进程被杀后重新跟随、内核中断、某一步失败后照样打包这几种情况。还没在 Colab 上跑过。

## 0.7.5 的限制和对应做法

| 限制 | 后果 | 做法 |
| --- | --- | --- |
| `colab exec` 的 `--timeout` 默认 30 秒，是整次执行的期限。0.7.5 固定使用的 jupyter-kernel-client 0.9.0 不执行这个期限，但过了期限后，本地 `colab` 进程会占满一个 CPU 核（[#183](https://github.com/googlecolab/google-colab-cli/issues/183)） | 长任务期间本机一个核满载；换成新版库后会直接超时失败 | 每次 `colab exec` 都加 `--timeout 86400` |
| 代码在 VM 的 Jupyter 内核里执行；同一会话的 `colab exec` 按顺序排队 | 任务运行时再开一个 `colab exec`，它会一直等 | 任务运行时只用 `colab download` 看日志 |
| `--env KEY=VALUE` 会把 `os.environ[...] = '...'` 拼进要执行的代码，代码原样写进本地 `~/.config/colab-cli/history/<会话>.jsonl` | 令牌会明文写进本地文件 | 令牌用 `colab upload` 传文件，历史里只记录路径 |
| Colab Secrets（`userdata.get`）需要网页前端 | CLI 里拿不到 | 同上 |
| `--gpu` 的值写错时，静默改成 A100 | 可能开错卡 | 开完用 `colab status` 确认 |
| `colab download` 只能下载单个文件 | 不能直接下载整个报告目录 | 第一次上卡的脚本退出时，把报告目录打包成一个文件 |

## 一次性准备

```bash
colab version                     # 应显示 0.7.5
mkdir -p ~/.config/opd && chmod 700 ~/.config/opd
# GitHub fine-grained PAT：Repository access 只选 XDE10/qwen-dual-teacher-opd，Permissions 里 Contents 设为 Read-only
read -rsp 'GitHub PAT: ' PAT; echo
printf 'https://XDE10:%s@github.com\n' "$PAT" > ~/.config/opd/gh_credentials; unset PAT
chmod 600 ~/.config/opd/gh_credentials
```

Hugging Face 令牌是可选的，只用来加快下载：

```bash
read -rsp 'HF token: ' HFT; echo; printf '%s' "$HFT" > ~/.config/opd/hf_token; unset HFT; chmod 600 ~/.config/opd/hf_token
```

令牌文件放在 WSL2 的 Linux 目录（`~` 下）。不要放在 `/mnt/c` 或 `/mnt/d`：那里 `chmod` 不生效，Windows 上的程序也能读到。

## 每个会话

### 1. 开 G4

```bash
colab new -s opd --gpu G4
colab status -s opd               # 要看到 Hardware: G4
```

分配被拒（412 表示容量不足，400 表示账号没有这种卡）时改用 `colab new -s opd --gpu A100`。

### 2. 挂 Drive

```bash
colab drivemount -s opd
```

第一次会打印一个授权链接：复制到 Windows 浏览器里打开并同意，再回到 WSL2 终端按回车。挂不上时有两个办法：

- `colab url -s opd` 打印一个链接，在浏览器里打开的 Colab 页面连的是同一台 VM，在单元格里运行 `from google.colab import drive; drive.mount('/content/drive')`；
- 不用 Drive，见第 4 步。

### 3. 上传令牌，克隆分支

```bash
colab upload -s opd ~/.config/opd/gh_credentials /content/gh_credentials
colab upload -s opd ~/.config/opd/hf_token /content/hf_token        # 可选
colab exec -s opd --timeout 86400 <<'EOF'
import os, shutil, subprocess
os.chmod('/content/gh_credentials', 0o600)
if os.path.exists('/content/hf_token'):
    os.makedirs('/root/.cache/huggingface', exist_ok=True)
    shutil.move('/content/hf_token', '/root/.cache/huggingface/token')
for cmd in (['git', 'config', '--global', 'credential.helper', 'store --file=/content/gh_credentials'],
            ['git', 'clone', '-b', 'feat/opd-v3-mopd-data-v5',
             'https://github.com/XDE10/qwen-dual-teacher-opd.git', '/content/project2'],
            ['git', '-C', '/content/project2', 'log', '--oneline', '-1']):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout + r.stderr, end='')
    r.check_returncode()
EOF
```

最后一行是分支的最新提交。

- VM 直接从 GitHub 克隆。不要从 Windows 检出的目录上传脚本：Windows 上的 Git 可能把 `.sh` 的换行改成 CRLF，bash 会报 `$'\r': command not found`。
- 这里用 Python 的 `subprocess`，没有用 `!git ...`。`!` 由 Colab 自己的 shell 实现处理，它会向网页前端发请求；0.7.5 的 CLI 只应答 Drive 授权这一种请求，所以这里不依赖 `!`。

### 4. 第一次上卡（约 35–50 分钟）

```bash
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py first_session -- bash scripts/colab_first_session_v3.sh
EOF
```

没挂 Drive 时，把 `--` 后面换成 `env OPD_OUTPUT_ROOT=/content/runs bash scripts/colab_first_session_v3.sh`。这样结果只在 VM 本地，`colab stop` 之前一定要下载。输出目录设在 `/content/drive/...` 而 Drive 没挂载时，脚本会直接报错退出，不会把文件写到本地同名目录里。

`colab_job_v3.py` 让任务在 VM 上自己的会话里运行，输出写进 `/content/jobs/first_session/output.log`，同时把日志打印到你的终端，直到任务结束：

- 关掉终端、Windows 睡眠、按 Ctrl-C，任务都会继续跑。
- 任务结束前不要运行 `colab restart-kernel` 或 `colab stop`。
- 任务运行时不要对同一会话再开 `colab exec`，它会排在任务后面。断线后重新运行上面同一条命令，它会等任务结束，然后打印退出状态和最后 40 行日志。

### 5. 看进度（另开一个 WSL2 终端）

```bash
colab download -s opd /content/jobs/first_session/output.log ./first_session.log && tail -n 40 ./first_session.log
```

`colab download` 走 Jupyter 的文件接口，不经过内核，任务运行时也能用。

### 6. 取回结果

```bash
colab download -s opd /content/opd_first_session.tar.gz ./opd_first_session.tar.gz
tar -xzf opd_first_session.tar.gz
```

脚本退出时（失败时也一样）会把报告目录打包成这个文件。把里面的 `smoke_summary.json` 和 `teacher_fit.json` 发给我；某一步失败时发 `console.log`。挂了 Drive 的话，同样的文件也在 Drive 的 `project2/runs/first_session_<时间>/` 下。

### 7. 更新代码、重跑、收尾

我推送修复后，在 VM 上更新代码：

```bash
colab exec -s opd --timeout 86400 <<'EOF'
import subprocess
r = subprocess.run(['git', '-C', '/content/project2', 'pull', '--ff-only'], capture_output=True, text=True)
print(r.stdout + r.stderr, end='')
EOF
```

重跑第一次上卡时，在任务名后加 `--again`，旧的任务目录会改名保留：

```bash
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py first_session --again -- bash scripts/colab_first_session_v3.sh
EOF
```

用完释放 VM。G4 按使用时长扣计算单元，余额用 `colab usage` 查看。

```bash
colab stop -s opd
```

## 以后的长任务

训练和评测也通过 `colab_job_v3.py` 运行，每个任务用一个名字，例如：

```bash
colab exec -s opd --timeout 86400 <<'EOF'
%run /content/project2/scripts/colab_job_v3.py d4_dual -- python -m scripts.train_project2_opd_v3 --config configs/project2_opd_v3_d4_dual.json --output-root /content/drive/MyDrive/project2/runs
EOF
```

VM 会被回收，运行时长也有上限，所以长训练的 `--output-root` 要放在 Drive 上，中断后用 checkpoint 续训（配置里的 `resume_from_checkpoint`）。
