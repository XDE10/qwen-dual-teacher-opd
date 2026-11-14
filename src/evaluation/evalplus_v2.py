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


MEMORY_POLICIES = ('vms-headroom-v1', 'legacy-fixed-1g')
DEFAULT_MEMORY_POLICY = MEMORY_POLICIES[0]


def container_entry(memory_policy=DEFAULT_MEMORY_POLICY):
    """Versioned resource policy; official tests and scoring are unchanged."""
    if memory_policy not in MEMORY_POLICIES:
        raise ValueError('Unknown EvalPlus memory policy')
    common = """import importlib.metadata as m
import ctypes
import os
import evalplus.evaluate as cli
import evalplus.eval as evaluator
assert m.version('evalplus')=='0.3.1'
# WSL's piped core handler can ignore RLIMIT_CORE. Disable dumping only for
# this evaluator and its forked children; never change host core_pattern.
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
    raise OSError(ctypes.get_errno(), 'Could not disable evaluator core dumps')
"""
    if memory_policy == 'legacy-fixed-1g':
        return common + '\ncli.main()\n'
    return common + """
def memory_with_headroom():
    # Called inside each evaluation child, AFTER datasets/imports were inherited.
    # Lowering RLIMIT_AS below inherited VMS prevents even stack growth.
    with open('/proc/self/statm', encoding='ascii') as statm:
        baseline = int(statm.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    headroom = int(os.environ['OPD_EVALPLUS_EXTRA_MEMORY_BYTES'])
    if headroom <= 0:
        raise ValueError('EvalPlus extra memory budget must be positive')
    return baseline + headroom

evaluator.query_maximum_memory_bytes = memory_with_headroom
cli.main()
"""

def docker_command(image, dataset, samples_dir, official_file, docker='docker', memory_policy=DEFAULT_MEMORY_POLICY):
    if not image.startswith('sha256:') or dataset not in {'humaneval', 'mbpp'}:
        raise ValueError('Pin the official-evaluator Docker image ID and dataset')
    override = 'HUMANEVAL_OVERRIDE_PATH' if dataset == 'humaneval' else 'MBPP_OVERRIDE_PATH'
    entry = container_entry(memory_policy)
    # --pids-limit counts threads too: one BLAS/OpenMP thread per process and two evaluation workers
    # (matching --cpus 2) keep EvalPlus's worker processes well inside it.
    return [docker, 'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', '6g', '--cpus', '2',
        '--ulimit', 'core=0:0', '--user', '65534:65534', '--tmpfs', '/tmp:rw,nosuid,size=1g',
        '-e', 'HOME=/tmp', '-e', 'XDG_CACHE_HOME=/tmp', '-e', override + '=/dataset.jsonl',
        '-e', 'EVALPLUS_MAX_MEMORY_BYTES=1073741824',
        '-e', 'OPD_EVALPLUS_EXTRA_MEMORY_BYTES=1073741824',
        '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
        '--mount', f'type=bind,source={Path(samples_dir).resolve().as_posix()},target=/results',
        '--mount', f'type=bind,source={Path(official_file).resolve().as_posix()},target=/dataset.jsonl,readonly',
        '--entrypoint', 'python', image, '-c', entry, '--dataset', dataset, '--samples', '/results/samples.jsonl',
        '--parallel', '1']
