"""opd-v3 run identity: hard fields block resume, soft environment facts only warn.

v2 put every installed package and the platform string into one identity, so a
new Colab session with a refreshed base image could never resume. v3 splits it.
Hard: config, data/audit/contract/tokenizer/source digests, the resolved
sampling contract, and the pinned torch/transformers/peft versions. Soft:
Python/platform/GPU/CUDA strings and the full dependency list.
"""
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from src.training.runtime_v2 import REQUIRED, file_identity, read, sha, write

VERSION = 'opd-runtime-v3'
PINNED_PACKAGES = ('torch', 'transformers', 'peft')
SOURCES = ['src/training/opd_v3.py', 'src/training/runtime_v3.py', 'src/training/sampler_v3.py',
           'src/training/rollout_v3.py', 'src/training/teachers_v3.py', 'src/training/sampling_v2.py',
           'src/training/runtime_v2.py', 'src/losses/mopd_v3.py', 'src/teachers/opd_contract.py',
           'src/data/approved_training_inputs.py', 'scripts/train_project2_opd_v3.py']
NON_IDENTITY_KEYS = {'resume_from_checkpoint', 'resume_policy'}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_identity(config, generation, root, sources=SOURCES):
    root = Path(root).resolve()
    tokenizer = root / config['tokenizer_path']
    tokenizer_files = [p for p in tokenizer.rglob('*') if p.is_file() and '.cache' not in p.parts]
    if not tokenizer_files:
        raise ValueError('Empty tokenizer directory')
    hard = {'config': {k: v for k, v in config.items() if k not in NON_IDENTITY_KEYS},
            'manifest_sha': sha(root / config['data_manifest']),
            'audit_sha': sha(root / config['tokenizer_audit']),
            'prompt_contract_sha': sha(root / 'configs/project2_prompt_contract.json'),
            'tokenizer_files': file_identity(tokenizer_files, root),
            'source_files': file_identity([root / s for s in sources], root),
            'generation': generation,
            'pinned_packages': {name: package_version(name) for name in PINNED_PACKAGES}}
    soft = {'python': sys.version, 'platform': platform.platform(), 'runtime': runtime_facts(),
            'dependencies': dict(sorted((d.metadata['Name'].lower(), d.version)
                                        for d in importlib.metadata.distributions() if d.metadata['Name']))}
    return {'schema': VERSION, 'hard': hard, 'soft': soft}


def runtime_facts():
    try:
        import torch
    except ImportError:
        return {}
    cuda = torch.cuda.is_available()
    return {'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if cuda else None,
            'gpu': torch.cuda.get_device_name(0) if cuda else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if cuda else None,
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if cuda else None}


def soft_differences(saved, current):
    keys = sorted(set(saved.get('soft', {})) | set(current.get('soft', {})))
    return [k for k in keys if saved.get('soft', {}).get(k) != current.get('soft', {}).get(k)]


def check_identity(saved, current):
    """Raise on any hard difference; return the names of differing soft facts."""
    if saved.get('schema') != VERSION or current.get('schema') != VERSION:
        raise ValueError('Run identity schema mismatch')
    if saved['hard'] != current['hard']:
        differing = sorted(k for k in set(saved['hard']) | set(current['hard'])
                           if saved['hard'].get(k) != current['hard'].get(k))
        raise ValueError('Run identity mismatch (hard fields): ' + ', '.join(differing))
    return soft_differences(saved, current)


def seal_checkpoint(path):
    path = Path(path)
    files = {p.relative_to(path).as_posix(): sha(p) for p in path.rglob('*')
             if p.is_file() and p.name != 'checkpoint_manifest.json'}
    if not REQUIRED <= files.keys():
        raise ValueError('Checkpoint missing required files')
    write(path / 'checkpoint_manifest.json', {'schema': VERSION, 'files': files})


def validate_checkpoint(path, identity):
    """Verify the closed file set and digests BEFORE any torch.load, then identity."""
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
        if not raw.resolve().is_relative_to(path) or raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p != path):
            raise ValueError('Checkpoint path escapes root')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p.name != 'checkpoint_manifest.json'}
    if actual != set(files) or any(sha(path / n) != h for n, h in files.items()):
        raise ValueError('Checkpoint closed set or digest mismatch')
    return check_identity(read(path / 'run_identity.json'), identity)


def prepare_output(out, identity, resume_step=None, policy='reject'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / 'run_identity.json'
    warnings = check_identity(read(identity_path), identity) if identity_path.exists() else []
    checkpoints = list(out.glob('checkpoint-*'))
    log = out / 'metrics.jsonl'
    rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()] if log.exists() else []
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
            log.write_text(''.join(json.dumps(r) + '\n' for r in rows if r['step'] <= resume_step), encoding='utf-8')
    if warnings:
        write(out / ('resume_soft_differences_' + str(time.time_ns()) + '.json'),
              {'differing_soft_fields': warnings, 'policy': 'warn_only'})
    write(identity_path, identity)
    return warnings
