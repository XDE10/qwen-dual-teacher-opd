"""Prepare official full-solution JSONL and optionally invoke sealed Docker EvalPlus."""
import argparse
import json
from pathlib import Path
import subprocess
from src.evaluation.evalplus_v2 import convert_predictions, docker_command, MEMORY_POLICIES, DEFAULT_MEMORY_POLICY
from src.data.code_sandbox_v2 import docker_binary
from src.training.runtime_v2 import sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['humaneval', 'mbpp'], required=True)
    p.add_argument('--predictions', required=True, help='JSONL task_id/text; complete Python solutions')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--image-id', required=True, help='prebuilt image sha256 with evalplus==0.3.1; never auto pull')
    p.add_argument('--execute-docker', action='store_true')
    p.add_argument('--memory-policy', choices=MEMORY_POLICIES, default=DEFAULT_MEMORY_POLICY)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    official = Path('data/code_benchmarks/official') / ('HumanEvalPlus-v0.1.10.jsonl' if a.dataset == 'humaneval' else 'MbppPlus-v0.2.0.jsonl')
    tasks = [json.loads(l) for l in official.read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in Path(a.predictions).read_text(encoding='utf-8').splitlines()]
    samples = convert_predictions(predictions, tasks)
    destination = out / 'samples.jsonl'
    with destination.open('x', encoding='utf-8') as f:
        f.write(''.join(json.dumps(s) + '\n' for s in samples))
    command = docker_command(a.image_id, a.dataset, out, official, docker_binary(), memory_policy=a.memory_policy)
    identity = {'protocol': 'official-evalplus-v0.3.1', 'dataset_sha256': sha(official),
        'resource_protocol': a.memory_policy, 'core_dump_limit_bytes': 0, 'evaluator_dumpable': False,
        'extra_virtual_memory_budget_bytes': 1073741824 if a.memory_policy == DEFAULT_MEMORY_POLICY else None,
        'comparability_note': 'Resource policy differs from legacy runs; do not silently pool results.',
        'samples_sha256': sha(destination), 'command': command, 'task_ids': [s['task_id'] for s in samples],
        'status': '未验证', 'scores_separate_from_internal_dev': True}
    if a.execute_docker:
        subprocess.run([docker_binary(), 'image', 'inspect', a.image_id], check=True, stdout=subprocess.PIPE)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / 'evalplus.log').write_text(completed.stdout, encoding='utf-8')
        identity.update(exit_code=completed.returncode, status='executed' if completed.returncode == 0 else 'failed')
    write(out / 'official_protocol.json', identity)


if __name__ == '__main__':
    main()
