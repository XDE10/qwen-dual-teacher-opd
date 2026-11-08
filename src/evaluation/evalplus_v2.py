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
    return [{'task_id': r['task_id'], 'solution': extract_code(r['text'])} for r in predictions]


def docker_command(image, dataset, samples_dir, official_file, docker='docker'):
    if not image.startswith('sha256:') or dataset not in {'humaneval', 'mbpp'}:
        raise ValueError('Pin the official-evaluator Docker image ID and dataset')
    override = 'HUMANEVAL_OVERRIDE_PATH' if dataset == 'humaneval' else 'MBPP_OVERRIDE_PATH'
    entry = "import importlib.metadata as m,runpy; assert m.version('evalplus')=='0.3.1'; runpy.run_module('evalplus.evaluate',run_name='__main__')"
    return [docker, 'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', '4g', '--cpus', '2',
        '--user', '65534:65534', '--tmpfs', '/tmp:rw,nosuid,size=1g',
        '-e', 'HOME=/tmp', '-e', 'XDG_CACHE_HOME=/tmp', '-e', override + '=/dataset.jsonl',
        '--mount', f'type=bind,source={Path(samples_dir).resolve().as_posix()},target=/results',
        '--mount', f'type=bind,source={Path(official_file).resolve().as_posix()},target=/dataset.jsonl,readonly',
        '--entrypoint', 'python', image, '-c', entry, '--dataset', dataset, '--samples', '/results/samples.jsonl']
