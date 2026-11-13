"""Stage pinned Hugging Face models on local disk, with an optional Drive mirror."""

import fcntl
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

FILES = {
    'LICENSE', 'README.md', 'config.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
    'chat_template.jinja', 'merges.txt', 'vocab.json', 'model.safetensors',
}
MARKER = 'shared_model_verified_v1.json'


def official_manifest(spec):
    from huggingface_hub import HfApi

    repo, revision = spec['model'], spec['revision']
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f'Unexpected revision for {repo}')
    files = sorted(({'path': f.rfilename, 'size': f.size,
                     'lfs_sha256': f.lfs.sha256 if f.lfs else None, 'git_blob_id': f.blob_id}
                    for f in info.siblings if f.rfilename in FILES), key=lambda f: f['path'])
    if not {'config.json', 'model.safetensors'} <= {f['path'] for f in files}:
        raise ValueError(f'Missing required files for {repo}')
    return {'repo_id': repo, 'revision': revision, 'files': files, 'official_hashes_verified': True}


def verify(folder, manifest):
    for file in manifest['files']:
        path = folder / file['path']
        if not path.is_file() or path.stat().st_size != file['size']:
            return False
        with path.open('rb') as stream:
            if file['lfs_sha256']:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != file['lfs_sha256']:
                    return False
            else:
                blob = hashlib.sha1(f"blob {file['size']}\0".encode())
                blob.update(stream.read())
                if blob.hexdigest() != file['git_blob_id']:
                    return False
    return True


def copy_model(source, destination, files):
    result = subprocess.run(['rclone', 'copy', str(source), str(destination), '--checksum',
                             '--files-from-raw', '-', '--contimeout', '10s', '--timeout', '60s',
                             '--retries', '1'], check=False, input='\n'.join(files) + '\n',
                            capture_output=True, text=True)
    return result.returncode == 0


def stage_model(spec, root, drive):
    from huggingface_hub import snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"{spec['model'].split('/')[-1]}--{spec['revision']}"
    folder = root / name
    remote = f"{drive.rstrip('/')}/{name}" if drive and shutil.which('rclone') else None
    started = time.monotonic()
    manifest = official_manifest(spec)
    files = [file['path'] for file in manifest['files']]
    with (root / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if verify(folder, manifest):
            source = 'local'
        else:
            with tempfile.TemporaryDirectory(prefix='stage-', dir=root) as temporary:
                staged = Path(temporary) / name
                if remote and copy_model(remote, staged, files) and verify(staged, manifest):
                    source = 'drive'
                else:
                    source = 'hugging_face'
                    shutil.rmtree(staged, ignore_errors=True)
                    snapshot_download(spec['model'], revision=spec['revision'], local_dir=staged,
                                      allow_patterns=files)
                    if not verify(staged, manifest):
                        raise ValueError(f'Official file hashes failed for {name}')
                (staged / MARKER).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                if folder.exists():
                    shutil.rmtree(folder)
                staged.replace(folder)
        print(json.dumps({'model': spec['model'], 'model_source': source,
                          'model_stage_seconds': round(time.monotonic() - started, 2)}), flush=True)
        if remote and source != 'drive':
            if not (folder / MARKER).is_file():
                (folder / MARKER).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            started = time.monotonic()
            saved = copy_model(folder, remote, [*files, MARKER])
            print(json.dumps({'model': spec['model'], 'saved_to_drive': saved,
                              'drive_copy_seconds': round(time.monotonic() - started, 2)}), flush=True)
    return folder
