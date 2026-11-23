"""Check the inputs and entry points shipped in a clone or source ZIP."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reviewed_registry_v3 import validate_registry

RELEASE = ROOT / 'data/project2_release_v5_20261108/release'
RESULTS = ROOT / 'reports/latent_mopd_three_closeout_20261121/scores.json'
TRAINER_SHA = 'c68797a2063ad5df7bf1d9e187ad94c0c37bbbf12894b26a14a518052fa29eb7'
MATH500_SHA = '4a8ada6bff87e172a59d78467764e781ae92460a4c6a3ec23f11e99ecd97c04e'
BENCHMARKS = {'aime24', 'aime25', 'minerva', 'mbpp', 'mbpp_plus', 'lcb_easy', 'bbh', 'musr', 'gym'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_digest(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'SHA-256 differs for {path.relative_to(ROOT)}')
    return actual


def main():
    for required in (
        'requirements/wsl2-cpu.txt',
        'requirements/linux-cuda-v3.txt',
        'scripts/prepare_latent_mopd_three.py',
        'scripts/evaluate_latent_mopd_three_dev.py',
        'scripts/score_code_local_v3.py',
        'tests/test_latent_mopd_three.py',
    ):
        if not (ROOT / required).is_file():
            raise ValueError(f'Missing public entry or dependency file: {required}')

    verify_digest(ROOT / 'src/training/latent_mopd_three.py', TRAINER_SHA)
    verify_digest(ROOT / 'data/math/math500_test.jsonl', MATH500_SHA)

    manifest = read_json(RELEASE / 'manifest.json')
    for name in ('project2_math_dev.jsonl', 'project2_code_dev.jsonl'):
        if name not in manifest['approved_evaluation_files']:
            raise ValueError(f'{name} is outside the approved evaluation scope')
        meta = manifest['files'][name]
        verify_digest(RELEASE / name, meta['sha256'] if isinstance(meta, dict) else meta)

    code_dev = RELEASE / 'project2_code_dev.jsonl'
    rows = [json.loads(line) for line in code_dev.read_text(encoding='utf-8').splitlines() if line.strip()]
    registry = read_json(ROOT / 'data/evaluation/reviewed_comparators.json')
    validate_registry(rows, registry, digest(code_dev))

    scores = read_json(RESULTS)
    if scores['checkpoint_step'] != 70 or set(scores['benchmarks']) != BENCHMARKS:
        raise ValueError('The closing score record must describe step 70 and the nine benchmarks')
    print('Verified trainer, protected data, two dev sets, Docker comparator registry and nine score records.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from None
