"""Versioned identity, sealed checkpoint files, resume policy and phase telemetry."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time
from contextlib import contextmanager

VERSION = 'opd-runtime-v2'
REQUIRED = {'engine.pt', 'run_identity.json', 'student/adapter_config.json', 'student/adapter_model.safetensors'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def file_identity(paths, root):
    root = Path(root).resolve()
    return {Path(p).resolve().relative_to(root).as_posix(): sha(p) for p in sorted(paths)}


def build_identity(config, generation, root):
    import torch
    root = Path(root).resolve()
    stable = {k: v for k, v in config.items() if k not in {'resume_from_checkpoint', 'resume_policy'}}
    # Output location is part of experiment identity; Drive path must be resolved first.
    sources = [root / p for p in ['src/training/opd_v2.py', 'src/training/sampling_v2.py',
                'src/training/runtime_v2.py', 'src/losses/chunked_opd_v2.py',
                'src/losses/reference.py', 'src/teachers/opd_contract.py',
                'src/data/approved_training_inputs.py', 'scripts/train_project2_opd_v2.py',
                'scripts/train_project2_opd.py', 'src/training/opd.py']]
    tokenizer = root / config['tokenizer_path']
    files = [p for p in tokenizer.rglob('*') if p.is_file() and '.cache' not in p.parts]
    if not files:
        raise ValueError('Empty tokenizer directory')
    return {'schema': VERSION, 'config': stable,
            'manifest_sha': sha(root / config['data_manifest']),
            'audit_sha': sha(root / config['tokenizer_audit']),
            'prompt_contract_sha': sha(root / 'project_blueprints/configs/project2_prompt_contract.json'),
            'tokenizer_files': file_identity(files, root), 'source_files': file_identity(sources, root),
            'generation': generation,
            'runtime': {'python': sys.version, 'platform': platform.platform(),
                        'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            'dependencies': dict(sorted((d.metadata['Name'].lower(), d.version)
                                        for d in importlib.metadata.distributions() if d.metadata['Name']))}


def seal_checkpoint(path):
    path = Path(path)
    files = {p.relative_to(path).as_posix(): sha(p) for p in path.rglob('*') if p.is_file() and p.name != 'checkpoint_manifest.json'}
    if not REQUIRED <= files.keys():
        raise ValueError('Checkpoint missing required files')
    write(path / 'checkpoint_manifest.json', {'schema': VERSION, 'files': files})


def validate_checkpoint(path, identity):
    path = Path(path).resolve()
    document = read(path / 'checkpoint_manifest.json')
    files = document.get('files', {})
    if document.get('schema') != VERSION or not files or not REQUIRED <= files.keys():
        raise ValueError('Empty/incomplete checkpoint manifest')
    for name in files:
        relative = Path(name)
        if '\\' in name or relative.is_absolute() or ':' in name or '..' in relative.parts or relative.as_posix() != name:
            raise ValueError('Checkpoint path must be canonical relative path')
        raw = path / relative
        target = raw.resolve()
        if not target.is_relative_to(path) or raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p != path):
            raise ValueError('Checkpoint path escapes root')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p.name != 'checkpoint_manifest.json'}
    if actual != set(files) or any(sha(path / n) != h for n, h in files.items()):
        raise ValueError('Checkpoint closed set or digest mismatch')
    if read(path / 'run_identity.json') != identity:
        raise ValueError('Run identity mismatch')
    # Caller may torch.load ONLY after this function returns.
    return document


def prepare_output(out, identity, resume_step=None, policy='reject'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / 'run_identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Output identity changed')
    checkpoints = list(out.glob('checkpoint-*'))
    log = out / 'metrics.jsonl'
    rows = [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines() if l.strip()] if log.exists() else []
    if resume_step is None and (checkpoints or rows):
        raise ValueError('Existing run requires explicit resume or new output')
    if resume_step is not None:
        later = [p for p in checkpoints if int(p.name.split('-')[-1]) > resume_step]
        later_rows = [r for r in rows if r['step'] > resume_step]
        if later or later_rows:
            if policy != 'archive_future_v2':
                raise ValueError('Later checkpoints/log steps require archive_future_v2')
            archive = out / ('resume_archive_' + str(time.time_ns()))
            archive.mkdir()
            for p in later:
                p.rename(archive / p.name)
            if log.exists():
                log.rename(archive / log.name)
            retained = [r for r in rows if r['step'] <= resume_step]
            log.write_text(''.join(json.dumps(r) + '\n' for r in retained), encoding='utf-8')
    write(identity_path, identity)


class PhaseRecorder:
    def __init__(self):
        self.records = []

    @contextmanager
    def phase(self, name):
        import torch
        import psutil
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            self.records.append({'phase': name, 'seconds': time.monotonic() - start,
                'cpu_rss_before': rss_before, 'cpu_rss_after': process.memory_info().rss,
                'allocated': torch.cuda.memory_allocated() if cuda else None,
                'reserved': torch.cuda.memory_reserved() if cuda else None,
                'peak_allocated': torch.cuda.max_memory_allocated() if cuda else None,
                'peak_reserved': torch.cuda.max_memory_reserved() if cuda else None,
                'gpu_status': 'observed' if cuda else '未验证'})


def teacher_key(spec):
    return spec['model'], spec['revision']
