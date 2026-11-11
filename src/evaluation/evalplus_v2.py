"""Official EvalPlus v0.3.1 adapter; no internal comparator substitution."""
from pathlib import Path
import json
from src.evaluation.code_v2 import extract_code


def convert_predictions(predictions, tasks):
    expected = {r['task_id'] for r in tasks}
    if len({r['task_id'] for r in predictions}) != len(predictions):
        raise ValueError('pass@1 requires one candidate per task')
    if {r['task_id'] for r in predictions} != expected:
        raise ValueError('Official task ID set mismatch')
    # 'solution' is a complete program; never append official reference code/tests.
    def safe_extract(text):
        try:
            return extract_code(text)
        except Exception:
            return text
    return [{'task_id': r['task_id'], 'solution': safe_extract(r['text'])} for r in predictions]


def docker_command(image, dataset, samples_dir, official_file, docker='docker'):
    if not image.startswith('sha256:') or dataset not in {'humaneval', 'mbpp'}:
        raise ValueError('Pin the official-evaluator Docker image ID and dataset')
    override = 'HUMANEVAL_OVERRIDE_PATH' if dataset == 'humaneval' else 'MBPP_OVERRIDE_PATH'
    entry = (
        "import importlib.metadata as m,subprocess,sys; "
        "assert m.version('evalplus')=='0.3.1'; "
        "sys.exit(subprocess.call([sys.executable, '-m', 'evalplus.evaluate', *sys.argv[1:]]))"
    )
    # --pids-limit counts threads too: one BLAS/OpenMP thread per process and two evaluation workers
    # (matching --cpus 2) keep EvalPlus's worker processes well inside it.
    return [docker, 'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', '6g', '--cpus', '2',
        '--user', '65534:65534', '--tmpfs', '/tmp:rw,nosuid,size=1g',
        '-e', 'HOME=/tmp', '-e', 'XDG_CACHE_HOME=/tmp', '-e', override + '=/dataset.jsonl',
        '-e', 'EVALPLUS_MAX_MEMORY_BYTES=1073741824',
        '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
        '--mount', f'type=bind,source={Path(samples_dir).resolve().as_posix()},target=/results',
        '--mount', f'type=bind,source={Path(official_file).resolve().as_posix()},target=/dataset.jsonl,readonly',
        '--entrypoint', 'python', image, '-c', entry, '--dataset', dataset, '--samples', '/results/samples.jsonl',
        '--parallel', '1']
