# 三教师 Latent-MOPD

用数学、代码、逻辑三位 1.5B 教师蒸馏一个 **DeepSeek-R1-Distill-Qwen-1.5B** 学生。学生在线生成回答，当步对应的教师在同一串 token 上评分，训练结合 Top-16 候选优势与最后三层隐状态对齐。部署时只加载学生。

本项目独立复现 [论文同家族 last-3 路线](https://arxiv.org/html/2610.02381v1#S4.T1)。从本地 60、70、80 步中选出 **70 步**做九项终评。[查看最终成绩、原始分和重测说明](reports/latent_mopd_three_closeout_20261121/README.md)。

## 下载后先跑一次

在 Linux 或 Windows WSL2 上使用 **Python 3.12**。也可解压 GitHub 的 Download ZIP，在解压目录从 `python3.12 -m venv` 开始。安装依赖需要联网。

```bash
git clone --depth 1 https://github.com/XDE10/qwen-three-teacher-latent-mopd.git
cd qwen-three-teacher-latent-mopd

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/wsl2-cpu.txt

OMP_NUM_THREADS=2 python -m pytest -q tests/test_latent_mopd_three.py -k one_update
```

看到 **1 passed** 就跑通了真实训练器的一次反向和 AdamW 更新。测试使用随机初始化的小型 Qwen2，在 CPU 内创建数据，安装完成后可离线运行。完整 CPU 回归使用 `OMP_NUM_THREADS=2 python -m pytest -q`。发布文件巡检使用 `python tools/verify_repository.py`。

## 正式运行

- [从公开源构建数据到单卡训练](docs/latent_mopd_three.md)。需要一张支持 BF16 的 CUDA GPU、四个模型的权重及经审查的数据清单。
- [运行三域开发集与代码判分](docs/latent_mopd_three_final_evaluation.md)。GPU 生成，代码候选用 Docker 执行测试。
- [恢复断点和延长训练](docs/latent_mopd_three_continuation.md)。用与断点配套的运行包和完整训练状态。
- [按执行顺序读项目结构](docs/README.md)。从一条题目进训练器开始，追到一次更新、一个 checkpoint 和最后的分数。

本次训练的 70 步权重由项目运行归档提供。Git 下载包含源码、运行所需的小型固定输入和成绩记录；可直接跑 CPU 示例，也可从公开底座和数据源开始训练自己的模型。
