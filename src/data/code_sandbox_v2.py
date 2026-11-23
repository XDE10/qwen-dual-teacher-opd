"""Docker-only runtime commands for the internal code dev scorer."""
import json
from pathlib import Path
import shutil
import subprocess

from src.training.runtime_v2 import sha, write

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = 'sha256:3b0c23d092b8ff94262fd3518cf8250c4b1d65c014d6090eea3b23383145201e'


def docker_binary():
    found = shutil.which('docker')
    fallback = Path('C:/Program Files/Docker/Docker/resources/bin/docker.exe')
    return found or (str(fallback) if fallback.exists() else 'docker')


def environment(worker=None, image=None, record_path=None):
    worker = Path(worker or ROOT / 'scripts/code_sandbox_worker.py').resolve()
    image = image or DEFAULT_IMAGE
    if not image.startswith('sha256:'):
        raise ValueError('Pin Docker image by digest/ID')
    command = [docker_binary(), 'run', '--rm', '-i', '--network', 'none', '--read-only',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64',
        '--memory', '512m', '--cpus', '1', '--user', '65534:65534',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
        '--mount', f'type=bind,source={worker.as_posix()},target=/worker.py,readonly',
        '--entrypoint', 'python', image, '-I', '/worker.py']
    result = {'schema': 'code-sandbox-v2', 'image_id': image, 'worker_sha256': sha(worker),
              'worker_path': worker.as_posix(), 'command': command,
              'boundary': 'Docker Linux: no network, read-only, unprivileged, resource limited; no host-code fallback'}
    if record_path:
        write(record_path, result)
    return result


def run_jobs(jobs, worker=None, image=None, record_path=None):
    meta = environment(worker, image, record_path)
    # Never pull implicitly. Missing daemon/image is an infrastructure failure.
    try:
        subprocess.run([meta['command'][0], 'image', 'inspect', meta['image_id']],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        p = subprocess.run(meta['command'], input=json.dumps(jobs, ensure_ascii=False),
            encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=max(60, len(jobs) * 30))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('Docker infrastructure unavailable; no reduced-isolation fallback') from error
    if p.returncode:
        raise RuntimeError('Sandbox infrastructure failure: ' + p.stderr[-2000:])
    return json.loads(p.stdout)
