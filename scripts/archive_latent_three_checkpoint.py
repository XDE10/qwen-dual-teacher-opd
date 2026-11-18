"""Seal and archive a complete continuation checkpoint before reporting success."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

BASE = 'p2drive:opd_runs/latent-three-formal62-checkpoints/20261116T061609Z/recovery_materials_20261118'
BASE_SHA = '5b0a67bc191c5f5f91f7b6ae4421b6e76ae70f4605272c355f807e81cbbf8012'
DEFAULT_ARCHIVE = 'p2drive:opd_runs/latent-three-extended-checkpoints'


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def command(*arguments):
    reply = subprocess.run(arguments,capture_output=True,text=True,timeout=1800)
    if reply.returncode:
        # Do not surface credential-bearing connection errors.
        raise RuntimeError('Archival command failed: '+arguments[0]+' '+arguments[1])
    return reply.stdout


def validate_state(saved, step):
    required = {'optimizer','step','python_rng','numpy_rng','torch_rng','cuda_rng'}
    if not required <= saved.keys() or saved['step']!=step or not saved['cuda_rng']:
        raise ValueError('Missing optimizer, step or RNG in terminal checkpoint')
    optimizer = saved['optimizer']
    if any(not {'step','exp_avg','exp_avg_sq'} <= value.keys() for value in optimizer['state'].values()):
        raise ValueError('Terminal AdamW moments are incomplete')
    cursors = [int(value['step'].item()) for value in optimizer['state'].values()]
    if len(cursors)!=339 or max(cursors)!=step:
        raise ValueError('Terminal AdamW states do not match the completed step')
    if not optimizer['param_groups'] or any(
            g['lr']!=1e-5 or not g.get('fused') or tuple(g.get('betas',()))!=(0.9,0.999)
            or g.get('weight_decay')!=0.01 or g.get('eps')!=1e-8 for g in optimizer['param_groups']):
        raise ValueError('Terminal optimizer recipe differs')
    return len(cursors)


def commit_receipt(metadata, local, remote):
    # The verified marker is created only AFTER all checkpoint and recovery
    # artifact uploads and comparisons in archive() have succeeded.
    pending = local.with_name(local.name+'.pending')
    write(pending,metadata)
    command('rclone','copyto',str(pending),remote)
    if json.loads(command('rclone','cat',remote))!=metadata:
        raise RuntimeError('Drive terminal receipt readback differs')
    pending.replace(local)


def archive(trainer, output, step, entrypoint, archive_root=DEFAULT_ARCHIVE):
    checkpoint = output/f'checkpoint-{step}'
    identity = json.loads((checkpoint/'run_identity.json').read_text())
    saved = trainer.restore_checkpoint(checkpoint,identity)
    states = validate_state(saved,step)
    del saved
    versions = {k:importlib.metadata.version(k) for k in identity['hard']['pinned_packages']}
    if versions!=identity['hard']['pinned_packages']:
        raise ValueError('Installed environment differs from the sealed checkpoint')
    history = read_rows(output/'metrics.jsonl')
    if [r['step'] for r in history]!=list(range(1,step+1)) or any(r['optimizer_updates']!=1 for r in history):
        raise ValueError('Continuation history is incomplete')
    lineage = identity['hard']['continuation']
    entrypoint = Path(entrypoint).resolve()
    if trainer.file_digest(entrypoint)!=lineage['entrypoint_sha256'] or trainer.file_digest(Path(__file__))!=lineage['archival_entrypoint_sha256']:
        raise ValueError('Continuation or preservation entrypoint differs')
    base = json.loads(command('rclone','cat',BASE+'/capture_receipt.json'))
    if base['archive_sha256']!=BASE_SHA:
        raise ValueError('Base reconstruction bundle differs')
    key = hashlib.sha256(json.dumps(identity['hard'],sort_keys=True).encode()).hexdigest()
    destination = archive_root.rstrip('/')+'/'+key
    evidence = output/'preservation'
    evidence.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'checkpoint-{step}-',dir=evidence) as workspace:
        metadata = Path(workspace)
        for name in ('metrics.jsonl','phase_metrics.jsonl','extension_schedule.jsonl'):
            shutil.copyfile(output/name,metadata/name)
        shutil.copyfile(entrypoint,metadata/entrypoint.name)
        shutil.copyfile(Path(__file__),metadata/Path(__file__).name)
        for name,expected in identity['hard']['source_files'].items():
            source = trainer.opd.ROOT/name
            if trainer.file_digest(source)!=expected:
                raise ValueError('Runtime source differs: '+name)
            target = metadata/'sources'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        freeze = command(sys.executable,'-m','pip','freeze','--all')
        freeze = re.sub(r'(https?://)[^/\s]+@',r'\1[redacted]@',freeze)
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|ya29\.[A-Za-z0-9_-]{40,})',freeze):
            raise ValueError('Credential-like content in dependency output')
        (metadata/'pip-freeze-all.txt').write_text(freeze)
        (metadata/'nvidia-smi.txt').write_text(command('nvidia-smi'))
        write(metadata/'environment.json',{'python':sys.version,'pinned_packages':versions,
              'checkpoint_identity':identity,'base_reconstruction_sha256':BASE_SHA,'is_vm_image':False})
        payloads = {p.relative_to(metadata).as_posix():trainer.file_digest(p) for p in metadata.rglob('*') if p.is_file()}
        write(metadata/'RECOVERY_MANIFEST.json',payloads)
        command('rclone','copy',str(checkpoint),destination+f'/checkpoint-{step}','--transfers','4')
        command('rclone','check',str(checkpoint),destination+f'/checkpoint-{step}')
        recovery = destination+f'/recovery-{step}'
        command('rclone','copy',str(metadata),recovery+'/metadata')
        command('rclone','check',str(metadata),recovery+'/metadata')
        shared = '/recovery_environment_and_sources.tar.gz'
        command('rclone','copyto',BASE+shared,recovery+shared,'--checksum')
        source_md5 = command('rclone','md5sum',BASE+shared).split()[0]
        destination_md5 = command('rclone','md5sum',recovery+shared).split()[0]
        if source_md5!=destination_md5:
            raise RuntimeError('Reconstruction bundle copy differs')
        receipt = {'step':step,'target_budget':identity['hard']['config']['steps'],'drive_verified':True,
                   'checkpoint_manifest_sha256':trainer.file_digest(checkpoint/'checkpoint_manifest.json'),
                   'optimizer_states':states,'optimizer_step_max':step,'rng_state_saved':True,
                   'recovery_metadata_sha256':trainer.file_digest(metadata/'RECOVERY_MANIFEST.json'),
                   'base_reconstruction_sha256':BASE_SHA,'checkpoint':destination+f'/checkpoint-{step}',
                   'recovery':recovery,'pinned_versions':versions,'entrypoints_archived':True,
                   'retention':'Keep this verified checkpoint and recovery bundle for further derived continuation; no cloud pruning.',
                   'gpu_cold_resume_from_this_step':'not yet tested','verified_epoch':time.time()}
        local = evidence/f'checkpoint-{step}_verified.json'
        commit_receipt(receipt,local,destination+f'/checkpoint-{step}_verified.json')
    print(json.dumps({'event':'continuation_checkpoint_drive_verified','step':step,'checkpoint':receipt['checkpoint']}),flush=True)
    return receipt
