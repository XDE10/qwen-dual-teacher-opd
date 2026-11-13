"""Independent dual-teacher experiment based on Latent-MOPD (arXiv:2610.02381)."""

import argparse
import json
import os
import time
from pathlib import Path

from src.teachers.opd_contract import load_contract
from src.teachers.tokenizer_audit import compare_vocabularies
from src.training import opd_v3 as opd
from src.training.runtime_v2 import write

STUDENT = {'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
           'revision': 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'}
TEACHERS = {
    'math': {'model': 'hbx/JustRL-DeepSeek-1.5B', 'revision': '0637e4096c789c67f9eecbe8355e0bdeddede1c2'},
    'code': {'model': 'Fate-Zero/Archer2.0-Code-1.5B-Preview', 'revision': 'db1e1ff9f3c36f7177f3896358ce26edeb62ce56'},
}
DATA = {'data_manifest': 'data/project2_release_v5_20261108/release/manifest.json',
        'data_files': {'math': 'project2_math_train.jsonl', 'code': 'project2_code_train.jsonl'}}
ASSETS = Path('runs/latent_opd_assets')
AUDIT = ASSETS / 'tokenizer_audit.json'


def audit_inputs():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizers, configs = [], []
    models = [STUDENT, *TEACHERS.values()]
    for spec in models:
        folder = ASSETS / spec['model'].split('/')[-1]
        snapshot_download(spec['model'], revision=spec['revision'], local_dir=folder,
                          allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                          'special_tokens_map.json', 'chat_template.jinja'])
        tokenizers.append(AutoTokenizer.from_pretrained(folder, local_files_only=True))
        configs.append(json.loads((folder / 'config.json').read_text()))
    support, differences = compare_vocabularies([tok.get_vocab() for tok in tokenizers])
    if differences or support['size'] != 151665 or not support['contiguous_from_zero']:
        raise ValueError('Model vocabularies do not share the required 151665 contiguous token IDs')
    shape = {(cfg['model_type'], cfg['num_hidden_layers'], cfg['hidden_size']) for cfg in configs}
    if shape != {('qwen2', 28, 1536)} or any(cfg['vocab_size'] < support['size'] for cfg in configs):
        raise ValueError('Model shapes do not match same-lineage Qwen2 latent alignment')
    tok = tokenizers[0]
    if tok.eos_token_id not in support['ids'] or tok.pad_token_id not in support['ids']:
        raise ValueError('Student EOS and padding must be in the audited token support')
    write(AUDIT, {'models': models, 'common_support': support, 'mapping_difference_count': len(differences)})
    pools = opd.load_pools(DATA)
    prefixes = opd.render_prefixes(pools, tok, load_contract())
    return tok, support['size'], pools, prefixes


def train_step(trainer, variant, prompts, responses, microbatch_tokens, progress=None):
    import torch

    from src.losses.latent_opd import loss_weights, response_batch, reward_scale, routed_loss, score_candidates

    started = time.monotonic()
    cuda = torch.device(trainer.device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(trainer.device)
    step = trainer.state['step']
    domain = opd.DOMAINS[step % len(opd.DOMAINS)]
    token_weight, latent_weight = loss_weights(variant, domain, step // len(opd.DOMAINS))

    def event(phase, **detail):
        if progress is not None:
            progress.event(step + 1, domain, phase, **detail)

    event('step_start', token_weight=token_weight, latent_weight=latent_weight)
    rows = trainer.samplers[domain].next(prompts)
    prefixes = [trainer.prefixes[row['sample_id']] for row in rows for _ in range(responses)]
    trainer.engine.sync(trainer.student)
    rollouts = (progress.rollouts(trainer.engine, prefixes, step + 1, domain) if progress is not None
                else trainer.engine.generate(prefixes, domain))
    rollout_seconds = time.monotonic() - started
    targets = [len(item['response_ids']) for item in rollouts]
    if min(targets) < 1:
        raise RuntimeError('Empty on-policy response')
    sequences = [prefix + item['response_ids'] for prefix, item in zip(prefixes, rollouts)]
    target_count = sum(targets)
    sorted_lengths = sorted(targets)
    event('rollout_done', response_tokens=target_count, seconds=rollout_seconds,
          p50_response_tokens=sorted_lengths[(len(targets) - 1) // 2],
          p95_response_tokens=sorted_lengths[(95 * len(targets) - 1) // 100],
          truncated_responses=sum(not item['ended_with_eos'] for item in rollouts))
    teacher = trainer.teachers.get(domain)
    batches = []
    for group in opd.pack_by_tokens([len(ids) for ids in sequences], microbatch_tokens):
        ids, attention = opd.right_pad([sequences[i] for i in group], trainer.pad_id, trainer.device)
        batch = response_batch(ids, attention, [prefixes[i] for i in group], trainer.support)
        batches.append((batch, len(group)))

    scoring_started = time.monotonic()
    if token_weight:
        trainer.student.eval()
        scored = []
        for number, (batch, _) in enumerate(batches, 1):
            event('candidate_scoring_start', microbatch=number, total_microbatches=len(batches))
            began = time.monotonic()
            scored.append(score_candidates(trainer.student, teacher, batch, trainer.support))
            if cuda and progress is not None:
                torch.cuda.synchronize(trainer.device)
            event('candidate_scoring_done', microbatch=number, seconds=time.monotonic() - began)
        scale = reward_scale(scored)
    else:
        scored = [(None, None)] * len(batches)
        scale = 1.0
    if cuda:
        torch.cuda.synchronize(trainer.device)
    scoring_seconds = time.monotonic() - scoring_started

    trainer.student.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for number, ((batch, count), (candidates, rewards)) in enumerate(zip(batches, scored), 1):
        event('forward_start', microbatch=number, total_microbatches=len(batches))
        began = time.monotonic()
        advantages = rewards / scale if token_weight else None
        loss = routed_loss(trainer.student, teacher, batch, trainer.support, token_weight, latent_weight,
                           candidates, advantages) * (count / len(rollouts))
        if cuda and progress is not None:
            torch.cuda.synchronize(trainer.device)
        event('backward_start', microbatch=number, forward_seconds=time.monotonic() - began)
        began = time.monotonic()
        loss.backward()
        total_loss += float(loss.detach())
        del loss
        event('backward_done', microbatch=number, seconds=time.monotonic() - began)
    norm = torch.nn.utils.clip_grad_norm_(trainer.params, 1.0, error_if_nonfinite=True)
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.state['step'] = step + 1
    event('optimizer_done', gradient_norm=float(norm))
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(trainer.device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(trainer.device)} if cuda else {}
    return {'step': step + 1, 'domain': domain, 'token_weight': token_weight, 'latent_weight': latent_weight,
            'prompts': prompts, 'responses': len(rollouts), 'response_tokens': target_count,
            'truncated_responses': sum(not item['ended_with_eos'] for item in rollouts),
            'loss': total_loss, 'gradient_norm': float(norm), 'seconds': time.monotonic() - started,
            'rollout_seconds': rollout_seconds, 'candidate_scoring_seconds': scoring_seconds, **memory}


def run(args, assets):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    from src.training.latent_progress import RunProgress
    from src.training.model_drive_cache import stage_model
    from src.training.rollout_v3 import GenerationCopy, HFRolloutEngine
    from src.training.runtime_v3 import SOURCES, build_identity, prepare_output, validate_checkpoint
    from src.training.sampler_v3 import StratifiedSampler
    from src.training.sampling_v2 import generation_config, resolved_contract
    from src.training.teachers_v3 import TeacherPool

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One visible CUDA GPU is required for training')
    tok, support, pools, prefixes = assets
    specs = [STUDENT, *TEACHERS.values()]
    paths = {spec['model']: stage_model(spec, ASSETS / 'weights', args.model_drive) for spec in specs}

    def local_spec(spec):
        return {**spec, 'model': str(paths[spec['model']])}

    set_seed(args.seed)
    device = torch.device('cuda')
    output = Path(args.output_root) / ('latent_mopd_top16_' + args.variant)
    config = {**DATA, 'implementation_version': 'latent-mopd-top16-replay', 'student': STUDENT,
              'teachers': TEACHERS, 'variant': args.variant,
              'tokenizer_path': str(ASSETS / STUDENT['model'].split('/')[-1]), 'tokenizer_audit': str(AUDIT),
              'seed': args.seed, 'prompts': args.prompts, 'responses': args.responses, 'steps': args.steps,
              'max_new_tokens': args.max_new_tokens, 'rollout_batch': args.rollout_batch,
              'microbatch_tokens': args.microbatch_tokens, 'output_dir': str(output)}
    base = opd.load_model(local_spec(STUDENT), device)
    base.name_or_path = STUDENT['model']
    generation = GenerationCopy(base)
    configs = {d: generation_config(tok.eos_token_id, tok.pad_token_id, args.max_new_tokens) for d in opd.DOMAINS}
    contracts = {d: resolved_contract(generation.model, cfg, support, device) for d, cfg in configs.items()}
    identity = build_identity(config, contracts, opd.ROOT, sources=[*SOURCES, 'src/training/latent_opd.py',
                              'src/losses/latent_opd.py', 'src/teachers/tokenizer_audit.py',
                              'src/training/model_drive_cache.py', 'src/training/latent_progress.py'])
    checkpoint = opd.latest_valid_checkpoint(output, identity) if args.resume else None
    if args.resume and checkpoint is None and not (output / 'run_identity.json').is_file():
        raise ValueError(f'No saved run in {output}')
    if checkpoint is not None:
        validate_checkpoint(checkpoint, identity)
        saved = torch.load(checkpoint / 'engine.pt', map_location='cpu', weights_only=False)
        student = PeftModel.from_pretrained(base, checkpoint / 'student', is_trainable=True)
    else:
        saved = None
        student = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=opd.LORA_TARGETS,
                                                 task_type='CAUSAL_LM'), revision=STUDENT['revision'])
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.enable_input_require_grads()
    student.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-5,
                                  betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    teachers = TeacherPool(TEACHERS, lambda spec: opd.frozen_load(local_spec(spec), device), device, 'resident')
    engine = HFRolloutEngine(generation.model, configs, contracts, support, tok.pad_token_id, tok.eos_token_id,
                             args.rollout_batch, sync=generation.sync_from)
    samplers = {d: StratifiedSampler(rows, d, args.seed, opd.DEFAULTS['stratify_keys'][d])
                for d, rows in pools.items()}
    trainer = opd.OPDTrainerV3({'prompts_per_step': dict.fromkeys(opd.DOMAINS, args.prompts)}, student,
                               teachers, engine, samplers, prefixes, support, tok.pad_token_id, device,
                               optimizer, scheduler)
    if saved is not None:
        opd.restore_engine(trainer, saved)
    prepare_output(output, identity, trainer.state['step'] if args.resume else None, 'archive_future_v2')
    progress = RunProgress(output, identity, device)
    last = min(args.steps, args.stop_at_step) if args.stop_at_step else args.steps
    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        while trainer.state['step'] < last:
            record = train_step(trainer, args.variant, args.prompts, args.responses, args.microbatch_tokens, progress)
            step = record['step']
            if step == 1 or step % 10 == 0 or step == last:
                progress.event(step, record['domain'], 'checkpoint_start')
                opd.save_checkpoint(output / f'checkpoint-{step}', trainer, identity, record)
                progress.event(step, record['domain'], 'checkpoint_saved')
            print(json.dumps(record), file=log, flush=True)
            print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=('audit', 'token', 'latent', 'math20'))
    parser.add_argument('--output-root', default=os.environ.get('OPD_OUTPUT_ROOT', 'runs'))
    parser.add_argument('--model-drive', default=os.environ.get('OPD_MODEL_DRIVE', 'mydrive:models'),
                        help='Drive cache for pinned model snapshots (empty to disable)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at-step', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=62)
    parser.add_argument('--prompts', type=int, default=32)
    parser.add_argument('--responses', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=16384)
    parser.add_argument('--rollout-batch', type=int, default=16)
    parser.add_argument('--microbatch-tokens', type=int, default=24576)
    args = parser.parse_args()
    if min(args.steps, args.prompts, args.responses, args.max_new_tokens, args.rollout_batch,
           args.microbatch_tokens) < 1:
        parser.error('Training and generation budgets must be positive')
    if args.stop_at_step is not None and not 1 <= args.stop_at_step <= args.steps:
        parser.error('--stop-at-step must be between 1 and --steps')
    assets = audit_inputs()
    print(json.dumps({'common_token_ids': assets[1], 'approved_prompts': {d: len(p) for d, p in assets[2].items()}}),
          flush=True)
    if args.variant != 'audit':
        run(args, assets)


if __name__ == '__main__':
    main()
