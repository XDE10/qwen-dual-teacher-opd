"""Explicit derived continuation from a sealed checkpoint, defaulting to 62->80.

Run on the reconstructed, unchanged parent GPU environment. --plan-only works
locally without torch or CUDA. The derived run binds this entrypoint and its
appended schedule into its new hard identity.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.archive_latent_three_checkpoint import archive, DEFAULT_ARCHIVE

TRAINER_SHA = 'c68797a2063ad5df7bf1d9e187ad94c0c37bbbf12894b26a14a518052fa29eb7'
RELEASE_SHA = 'e5f4c579f8a4514795e6ba80e1e4c2df034e724346bbfce7a2ea150d7d969252'
PARENT_SCHEDULE_SHA = 'fb63303496426c95085f552961f4e2c733efa33de88dfc689bddde6fdb6e68bb'
DOMAINS = ('math', 'code', 'logic')
TARGET = 80


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    # Physical JSONL lines: str.splitlines() also splits Unicode separators
    # inside otherwise valid code prompts.
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_plan(manifest, target=TARGET):
    if target < 62:
        raise ValueError('Continuation plan must preserve the original 62 updates')
    if sha(manifest) != RELEASE_SHA:
        raise ValueError('Only the unchanged approved pool-fix release is supported')
    doc = json.loads(manifest.read_text())
    if not doc.get('approval', {}).get('approver'):
        raise ValueError('No owner approval in parent release')
    prior = manifest.with_name('schedule.jsonl')
    if sha(prior) != PARENT_SCHEDULE_SHA:
        raise ValueError('Parent schedule differs')
    protected_path = manifest.parent/doc['protected_dev_ids_file']
    if sha(protected_path) != doc['files'][protected_path.name]['sha256']:
        raise ValueError('Protected IDs differ')
    protected = set(json.loads(protected_path.read_text())['sample_ids'])
    pools, seen = {}, set()
    for domain in DOMAINS:
        path = manifest.with_name(domain+'_train.jsonl')
        if sha(path) != doc['files'][path.name]['sha256']:
            raise ValueError('Approved training pool differs: '+domain)
        rows = read_rows(path)
        for row in rows:
            sample = row['sample_id']
            if row.get('training_approved') is not True or row['domain'] != domain or sample in seen or sample in protected:
                raise ValueError('Unapproved, duplicate or protected training row')
            seen.add(sample)
        pools[domain] = rows
    plan = []
    for step in range(1, target+1):
        domain = DOMAINS[(step-1)%3]
        offset = ((step-1)//3)*32
        rows = pools[domain][offset:offset+32]
        if len(rows) != 32:
            raise ValueError('Insufficient approved rows')
        plan.append({'step':step,'domain':domain,'sample_ids':[r['sample_id'] for r in rows]})
    if read_rows(prior) != plan[:62]:
        raise ValueError('Continuation does not preserve the first 62 updates')
    return plan


def derive_hard(parent_hard, child_hard, lineage, output, target=TARGET):
    expected = copy.deepcopy(parent_hard)
    # Prior continuation metadata is retained transitively through the parent
    # hard-identity digest, while this child records its immediate parent.
    expected.pop('continuation',None)
    expected['config']['steps'] = target
    expected['config']['output_dir'] = str(output.resolve())
    if child_hard != expected:
        raise ValueError('Continuation changed more than declared steps and output directory')
    result = copy.deepcopy(child_hard)
    result['continuation'] = copy.deepcopy(lineage)
    return result


def segment_stops(parent_step, stop, checkpoint_every):
    """Fixed-budget save boundaries; validation scores never control this list."""
    if checkpoint_every < 1 or stop <= parent_step:
        raise ValueError('Expected a positive interval and an explicit later stop')
    current = parent_step
    while current < stop:
        current = min((current // checkpoint_every + 1) * checkpoint_every, stop)
        yield current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('/content/project2'))
    parser.add_argument('--parent-checkpoint',type=Path,default=Path('/content/runs/latent_three_step34_scoring_memory_20261116/full/latent_mopd_three_latent/checkpoint-62'))
    parser.add_argument('--output-root',type=Path,default=Path('/content/runs/latent_three_extended80_20261118/full'))
    parser.add_argument('--target-step',type=int,default=TARGET)
    parser.add_argument('--stop-at-step',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=5)
    parser.add_argument('--model-drive',default='mydrive:models')
    parser.add_argument('--archive-root',default=DEFAULT_ARCHIVE)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--plan-output',type=Path)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('Checkpoint interval must be positive')
    root = args.project_root.resolve()
    manifest = root/'runs/latent_mopd_three/release_poolfix_20261115/manifest.json'
    plan = make_plan(manifest,args.target_step)
    stop = args.stop_at_step if args.stop_at_step is not None else args.target_step
    plan_bytes = ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in plan).encode()
    if args.plan_output:
        args.plan_output.write_bytes(plan_bytes)
    if args.plan_only:
        print(json.dumps({'approved_plan_steps':len(plan),'parent_prefix_preserved':True,
                          'additional_prompt_positions':(args.target_step-62)*32,'additional_responses':(args.target_step-62)*32*4,
                          'schedule_sha256':hashlib.sha256(plan_bytes).hexdigest()}))
        return
    parent_path = args.parent_checkpoint.resolve()
    label = parent_path.name.removeprefix('checkpoint-')
    if not parent_path.name.startswith('checkpoint-') or not label.isdigit() or int(label)<62:
        parser.error('Expected a sealed parent checkpoint at step62 or later')
    parent_step = int(label)
    if not parent_step < stop <= args.target_step:
        parser.error('Explicit target and stop must be beyond the completed parent step')
    output = args.output_root.resolve()/'latent_mopd_three_latent'
    if output.is_relative_to(parent_path.parent) or parent_path.parent.is_relative_to(output):
        parser.error('Derived output must be independent of the parent checkpoint')
    if sha(root/'src/training/latent_mopd_three.py') != TRAINER_SHA:
        raise ValueError('Reconstruct the actual cache-on/scoring-memory trainer first')
    sys.path.insert(0,str(root))
    from src.training import latent_mopd_three as trainer
    from src.training.runtime_v3 import build_identity, check_identity
    parent = json.loads((parent_path/'run_identity.json').read_text())
    cfg = parent['hard']['config']
    if cfg['steps'] < parent_step or cfg['variant'] != 'latent' or cfg['seed'] != 42:
        raise ValueError('Unexpected parent experiment')
    expected = {'prompts':32,'responses':4,'rollout_batch':4,'max_new_tokens':16384,
                'packing_tokens':24576,'worker_responses':16}
    if any(cfg.get(k)!=v for k,v in expected.items()) or Path(cfg['data_manifest'])!=manifest:
        raise ValueError('Parent recipe differs')
    # Verify source/environment/data identities before any optimizer deserialization.
    tokenizer, tokenizer_path, audit = trainer.tokenizer_audit(Path(cfg['tokenizer_path']).parent)
    current = build_identity(cfg,parent['hard']['generation'],root,sources=trainer.SOURCE_FILES)
    core_parent = copy.deepcopy(parent)
    prior = core_parent['hard'].pop('continuation',None)
    check_identity(core_parent,current)
    if prior:
        if (prior.get('schema') not in ('explicit-parent62-continuation-1','explicit-chain-continuation-1')
                or prior.get('target_step')!=cfg['steps'] or not prior.get('parent_step',parent_step)<parent_step):
            raise ValueError('Parent continuation lineage is inconsistent')
        prior_plan = make_plan(manifest,cfg['steps'])
        prior_bytes = ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in prior_plan).encode()
        if hashlib.sha256(prior_bytes).hexdigest()!=prior.get('schedule_sha256'):
            raise ValueError('Parent continuation schedule differs')
    elif parent_step!=62 or cfg['steps']!=62:
        raise ValueError('A later parent requires declared continuation lineage')
    seal = trainer.verify_checkpoint_files(parent_path)
    schedule = trainer.read_schedule(manifest,tokenizer,args.target_step,32)
    if [{'step':n,'domain':domain,'sample_ids':samples} for n,(domain,_,samples) in enumerate(schedule,1)] != plan:
        raise ValueError('Rendered extension schedule differs from the sealed plan')
    parent_history = parent_path.parent/'metrics.jsonl'
    history = read_rows(parent_history)
    if [r['step'] for r in history] != list(range(1,parent_step+1)):
        raise ValueError('Parent history does not match the completed parent step')
    for row,planned in zip(history,plan[:parent_step],strict=True):
        if row['domain']!=planned['domain'] or row['sample_ids']!=planned['sample_ids'] or row['optimizer_updates']!=1:
            raise ValueError('Parent history differs from the approved schedule')
    lineage = {'schema':'explicit-chain-continuation-1','parent_checkpoint_manifest_sha256':seal,
               'parent_hard_sha256':hashlib.sha256(json.dumps(parent['hard'],sort_keys=True).encode()).hexdigest(),
               'parent_step':parent_step,'target_step':args.target_step,'entrypoint_sha256':sha(Path(__file__)),
               'archival_entrypoint_sha256':sha(Path(__file__).with_name('archive_latent_three_checkpoint.py')),
               'schedule_sha256':hashlib.sha256(plan_bytes).hexdigest(),
               'recipe':'unchanged; teacher-local crossfade remains complete; no optimizer or RNG reset'}
    output.mkdir(parents=True,exist_ok=True)
    for name,source in [('metrics.jsonl',parent_history),('phase_metrics.jsonl',parent_path.parent/'phase_metrics.jsonl')]:
        target = output/name
        if not target.exists() and source.is_file():
            shutil.copyfile(source,target)
    (output/'extension_schedule.jsonl').write_bytes(plan_bytes)
    original_latest = trainer.latest_resumable_checkpoint

    def child_identity(config,generation,project_root,sources):
        identity = build_identity(config,generation,project_root,sources=sources)
        identity['hard'] = derive_hard(parent['hard'],identity['hard'],lineage,output,args.target_step)
        return identity

    def resume_child_or_parent(folder,identity):
        if any(folder.glob('checkpoint-*')):
            return original_latest(folder,identity)
        if (folder/'run_identity.json').exists():
            check_identity(json.loads((folder/'run_identity.json').read_text()),identity)
        saved = trainer.restore_checkpoint(parent_path,parent)
        optimizer = saved['optimizer']
        if saved['step'] != parent_step or not saved['cuda_rng']:
            raise ValueError('Missing final optimizer or CUDA RNG state')
        if max(int(v['step'].item()) for v in optimizer['state'].values()) != parent_step:
            raise ValueError('Parent optimizer step differs')
        if any(g['lr']!=1e-5 or not g.get('fused') for g in optimizer['param_groups']):
            raise ValueError('Parent AdamW recipe differs')
        print(json.dumps({'event':'explicit_checkpoint_continuation','parent_step':parent_step,'target':args.target_step,'parent_seal':seal}),flush=True)
        return parent_path,saved

    trainer.build_identity = child_identity
    trainer.latest_resumable_checkpoint = resume_child_or_parent
    completed = parent_step
    if any(output.glob('checkpoint-*')):
        expected = copy.deepcopy(current)
        expected['hard']['config'].update(steps=args.target_step,output_dir=str(output.resolve()))
        expected['hard'] = derive_hard(parent['hard'],expected['hard'],lineage,output,args.target_step)
        _, resumed = original_latest(output,expected)
        completed = resumed['step']
        del resumed
    if completed > stop:
        raise ValueError('The derived run has already passed the requested stop')
    run_args = argparse.Namespace(variant='latent',manifest=manifest,output_root=args.output_root.resolve(),
                                  seed=42,steps=args.target_step,prompts=32,responses=4,rollout_batch=4,
                                  max_new_tokens=16384,model_drive=args.model_drive,resume=True,
                                  stop_at_step=stop,checkpoint_every=args.checkpoint_every,keep_checkpoints=2)
    for boundary in segment_stops(completed,stop,args.checkpoint_every) if completed < stop else []:
        run_args.stop_at_step = boundary
        trainer.run(run_args,tokenizer,tokenizer_path,audit,schedule,audit.parent)
        archive(trainer,output,boundary,Path(__file__),args.archive_root)
        gc.collect()
        trainer.torch.cuda.empty_cache()
    if completed == stop:
        archive(trainer,output,stop,Path(__file__),args.archive_root)
    print(json.dumps({'event':'continuation_segment_complete','step':stop,'parent_step':parent_step}),flush=True)


if __name__ == '__main__':
    main()
