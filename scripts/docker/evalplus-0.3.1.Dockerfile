# Official EvalPlus v0.3.1 evaluator image for scripts/evaluate_evalplus_v2.py (HumanEval+ / MBPP+ pass@1).
# Build once with network access; evaluation runs with --network none, a read-only root file system and an
# unprivileged user (src/evaluation/evalplus_v2.py). Not built or run in the repository's checks yet.
#   docker build -t opd-evalplus:0.3.1 -f scripts/docker/evalplus-0.3.1.Dockerfile scripts/docker
#   docker image inspect --format '{{.Id}}' opd-evalplus:0.3.1
# Pass that sha256:... ID as --evalplus-image to scripts/score_code_local_v3.py; it is recorded with every score.
FROM python:3.10-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install "evalplus==0.3.1" \
 && python -c "import importlib.metadata as m; assert m.version('evalplus') == '0.3.1'"
