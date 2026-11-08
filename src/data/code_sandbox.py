"""Invoke a restricted Linux container; dataset code never runs on the host."""
import json
import subprocess
from pathlib import Path
from src.data.screening_io import json_write,sha

ENVIRONMENT=Path('reports/code_v2_sandbox_environment.json')


def environment():
    # v2 runtime relocation; never rewrite historical environment JSON.
    from src.data.code_sandbox_v2 import environment as runtime_environment
    return runtime_environment()


def run_jobs(jobs):
    meta=environment()
    if meta['worker_sha256']!=sha('scripts/code_sandbox_worker.py'):raise ValueError('Sandbox worker version changed')
    process=subprocess.run(meta['command'],input=json.dumps(jobs,ensure_ascii=False),encoding='utf-8',
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=max(60,len(jobs)*30))
    if process.returncode:raise RuntimeError('Sandbox infrastructure failure: '+process.stderr[-2000:])
    response=json.loads(process.stdout)
    return response


def function_job(row):
    return {'kind':'function','sample_id':row['sample_id'],'reference_code':row['reference_code'],
            'entry_point':row['entry_point'],'assertions':[t['assertion'] for t in row['test_cases']],
            'constant_expected':row['test_cases'][0]['expected_repr']}
