# results 分支

这个分支只放运行结果里的小型文本文件（日志、JSON、JSONL），由 `scripts/publish_results_v3.py`
写入。代码在其他分支；不要把这个分支合并进任何代码分支。

目录是 `<任务名>/<UTC 时间>/`：

- `PUBLISHED.json`：发布时间；任务状态和退出码；产生结果的代码提交；GPU；已发布文件的字节数和 sha256；
  没发布的文件和原因。
- `job/`：`scripts/colab_job_v3.py` 的任务日志 `output.log` 和命令 `command.json`。
- 其他子目录：任务的 `--save` 内容，例如第一次上卡的 `report/`。

权重、checkpoint、压缩包和隐藏文件不会发布。看起来像 GitHub、Hugging Face、Google 凭据的字符串和
URL 里的密码会被替换成 `[REDACTED]`。完整产物在 Google Drive 的 `opd_runs/<任务名>/<UTC 时间>/`。
