"""Measure a full step-4-shaped joint backward with sealed checkpoint-2 and synthetic responses."""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import latent_mopd_three as trainer
from src.training.runtime_v2 import read, sha, write
from src.training.runtime_v3 import build_identity, check_identity

CHECKPOINT_SHA = '98ebf250c46f87b95e1ccb3ee5145bbb4c02a14ee344c0a282f37694337c3c84'
MANIFEST_SHA = 'e5f4c579f8a4514795e6ba80e1e4c2df034e724346bbfce7a2ea150d7d969252'
TRAINER_SHA = '2b7736f37c0e740ffcbfa9c69e368d6f4efd4a678760c7f717bad789d104bf6d'
STEP = 4
RESPONSE_TOKENS = 1_339_641
TRUNCATED_RESPONSES = 12
SEED = 42


def build_plan(manifest, tokenizer, stress_rollout=False):
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError('Expected the frozen poolfix release')
    schedule = manifest.with_name('schedule.jsonl')
    if sha(schedule) != read(manifest)['files'][schedule.name]['sha256']:
        raise ValueError('Schedule hash differs from the release')
    row = json.loads(schedule.read_text().splitlines()[STEP - 1])
    domain, prefixes, prompt_ids = trainer.read_schedule(manifest, tokenizer, STEP, trainer.PROMPTS)[-1]
    if row['step'] != STEP or row['domain'] != domain or domain != 'math' or row['sample_ids'] != prompt_ids:
        raise ValueError('Expected the sealed mathematics step 4')
    prefixes = [prefix for prefix in prefixes for _ in range(trainer.RESPONSES)]
    count = len(prefixes)
    # Spread cap-length responses across the production workers deterministically.
    long_slots = {i * count // TRUNCATED_RESPONSES for i in range(TRUNCATED_RESPONSES)}
    lengths = [trainer.MAX_RESPONSE if i in long_slots else 0 for i in range(count)]
    if stress_rollout:
        # Reserve a full two-sequence packing group within each worker.
        for start in range(0, count, trainer.WORKER_RESPONSES):
            worker = range(start, start + trainer.WORKER_RESPONSES)
            pair = [i for i in worker if not lengths[i]][:2]
            for i in pair:
                lengths[i] = trainer.PACK_TOKENS // 2 - len(prefixes[i])
    regular_slots = [i for i, length in enumerate(lengths) if not length]
    regular, remainder = divmod(RESPONSE_TOKENS - sum(lengths), len(regular_slots))
    for rank, i in enumerate(regular_slots):
        lengths[i] = regular + int(rank < remainder)
    full_lengths = [len(prefix) + length for prefix, length in zip(prefixes, lengths, strict=True)]
    groups = list(trainer.microbatches(full_lengths))
    if (count != 128 or sum(lengths) != RESPONSE_TOKENS
            or not all(0 < length <= trainer.MAX_RESPONSE for length in lengths)
            or lengths.count(trainer.MAX_RESPONSE) != TRUNCATED_RESPONSES
            or sorted(i for group in groups for i in group) != list(range(count))):
        raise ValueError('Synthetic workload differs from the sealed response budget')
    packed = []
    for group in groups:
        padded_tokens = len(group) * max(full_lengths[i] for i in group)
        if (len({i // trainer.WORKER_RESPONSES for i in group}) != 1
                or padded_tokens > trainer.PACK_TOKENS):
            raise ValueError('Invalid production packing group')
        packed.append({'indices': group, 'sequence_lengths': [full_lengths[i] for i in group],
                       'response_tokens': sum(lengths[i] for i in group), 'padded_tokens': padded_tokens})
    if stress_rollout and sum(g['padded_tokens'] == trainer.PACK_TOKENS for g in packed) != count // trainer.WORKER_RESPONSES:
        raise ValueError('Expected one full packing group per worker')
    token_weight, latent_weight = trainer.loss_weights('latent', domain, (STEP - 1) // len(trainer.DOMAINS))
    if (token_weight, latent_weight) != (0.1, 0.9):
        raise ValueError('Step 4 has unexpected loss weights')
    return prefixes, lengths, {
        'target_step': STEP, 'domain': domain, 'prompt_ids': prompt_ids, 'responses': count,
        'response_lengths': lengths, 'response_tokens': sum(lengths), 'truncated_responses': len(long_slots),
        'token_weight': token_weight, 'latent_weight': latent_weight, 'groups': packed,
        'worker_responses': trainer.WORKER_RESPONSES, 'packing_tokens': trainer.PACK_TOKENS,
        'max_padded_tokens': max(g['padded_tokens'] for g in packed), 'manifest_sha256': sha(manifest),
        'stress_rollout': stress_rollout,
    }


def synthetic_sequences(prefixes, lengths, tokenizer):
    rng = random.Random(SEED)
    excluded = set(tokenizer.all_special_ids)
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or not 0 <= eos < trainer.SUPPORT:
        raise ValueError('EOS is outside the audited token support')
    sequences = []
    for prefix, length in zip(prefixes, lengths, strict=True):
        capped = length == trainer.MAX_RESPONSE
        response = []
        while len(response) < length - int(not capped):
            token = rng.randrange(trainer.SUPPORT)
            if token not in excluded:
                response.append(token)
        if not capped:
            response.append(eos)
        sequences.append(prefix + response)
    return sequences


def stress_rollout(student, tokenizer, prefixes, domain, device, emit):
    # The common-support mask prevents this EOS marker from being sampled.
    # This keeps every sequence live for the full cap without adding a logits processor.
    eos = trainer.SUPPORT
    if eos >= student.config.vocab_size:
        raise ValueError('No masked model token available as a stress-only EOS marker')
    config = trainer.generation_config(eos, tokenizer.pad_token_id, trainer.MAX_RESPONSE)
    contract = trainer.resolved_contract(student, config, trainer.SUPPORT, device)
    engine = trainer.HFRolloutEngine(student, {domain: config}, {domain: contract}, trainer.SUPPORT,
                                     tokenizer.pad_token_id, eos, trainer.RESPONSES)
    prefix = max(prefixes, key=len)
    emit('stress_rollout_start', responses=trainer.RESPONSES, prompt_tokens=len(prefix))
    started = time.monotonic()
    device = torch.device(device)
    autocast = (torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=True)
                if device.type == 'cuda' else nullcontext())
    with autocast:
        rows = engine.generate([prefix] * trainer.RESPONSES, domain)
    lengths = [len(row['response_ids']) for row in rows]
    if lengths != [trainer.MAX_RESPONSE] * trainer.RESPONSES or any(row['ended_with_eos'] for row in rows):
        raise ValueError('Stress rollout did not reach the full token cap for every response')
    result = {'responses': len(rows), 'response_lengths': lengths, 'response_tokens': sum(lengths),
              'prompt_tokens': len(prefix), 'cache_enabled': True, 'masked_eos_token_id': eos,
              'seconds': time.monotonic() - started}
    emit('stress_rollout_done', **result)
    return result


def joint_backward(student, teacher, sequences, prefixes, groups, weights, pad_id, device, emit,
                   support=trainer.SUPPORT):
    # Mirrors the accumulation in trainer.update. The probe has no optimizer argument.
    device = torch.device(device)

    def autocast():
        if device.type == 'cuda':
            return torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)
        return nullcontext()

    def batch(group):
        ids, mask = trainer.opd.right_pad([sequences[i] for i in group], pad_id, device)
        return trainer.response_batch(ids, mask, [prefixes[i] for i in group], support)

    def group_details(index, group):
        return {'microbatch': index + 1, 'total_microbatches': len(groups),
                'group_size': len(group), 'response_tokens': sum(len(sequences[i]) - len(prefixes[i]) for i in group),
                'padded_tokens': max(len(sequences[i]) for i in group) * len(group)}

    student.eval()
    scored = []
    for index, group in enumerate(groups):
        detail = group_details(index, group)
        started = time.monotonic()
        emit('score_start', **detail)
        with autocast():
            ids, rewards = trainer.score(student, teacher, batch(group), support)
        scored.append((ids.cpu(), rewards.cpu()))
        del ids, rewards
        emit('score_done', **detail, seconds=time.monotonic() - started)
    scale = trainer.reward_scale(scored)
    emit('score_all_done', reward_std=float(scale))

    student.train()
    student.zero_grad(set_to_none=True)
    loss_value = token_value = rep_value = 0.0
    for index, (group, (ids, rewards)) in enumerate(zip(groups, scored, strict=True)):
        detail = group_details(index, group)
        started = time.monotonic()
        emit('backward_start', **detail)
        weight = len(group) / len(prefixes)
        with autocast():
            loss, token, rep = trainer.objective(student, teacher, batch(group), ids.to(device),
                                                   (rewards / scale).to(device), *weights, support)
            loss = weight * loss
        loss.backward()
        loss_value += float(loss.detach())
        token_value += weight * float(token)
        rep_value += weight * float(rep)
        emit('backward_done', **detail, seconds=time.monotonic() - started)
    norm = torch.nn.utils.get_total_norm([p.grad for p in student.parameters() if p.grad is not None],
                                        error_if_nonfinite=True)
    stats = {'loss': loss_value, 'token_loss': token_value, 'raw_rep_loss': rep_value,
             'reward_std': float(scale), 'gradient_norm': float(norm), 'optimizer_updates': 0}
    if not all(map(math.isfinite, stats.values())) or stats['gradient_norm'] <= 0:
        raise ValueError(f'Invalid losses or gradients: {stats}')
    return stats


def load_inputs(checkpoint, manifest, tokenizer_path, audit, tokenizer, device):
    if checkpoint.name != 'checkpoint-2' or sha(checkpoint / 'checkpoint_manifest.json') != CHECKPOINT_SHA:
        raise ValueError('Expected the verified step 2 checkpoint seal')
    archived = read(checkpoint / 'run_identity.json')
    hard = archived['hard']
    config = {**hard['config'], 'data_manifest': str(manifest), 'tokenizer_audit': str(audit),
              'tokenizer_path': str(tokenizer_path)}
    if hard['config'] != config or hard['source_files']['src/training/latent_mopd_three.py'] != TRAINER_SHA:
        raise ValueError('Probe requires the checkpoint-2 tokenizer, release and cached trainer identity')
    current = build_identity(config, hard['generation'], trainer.opd.ROOT, sources=trainer.SOURCE_FILES)
    check_identity(archived, current)
    saved = trainer.restore_checkpoint(checkpoint, current)
    if saved['step'] != 2 or not saved['cuda_rng'] or not saved['optimizer']['state']:
        raise ValueError('Checkpoint-2 is missing the saved optimizer or CUDA RNG state')
    student = AutoModelForCausalLM.from_pretrained(
        checkpoint / 'student', dtype=torch.float32, attn_implementation='sdpa',
        local_files_only=True, device_map={'': device},
    )
    gen = trainer.generation_config(tokenizer.eos_token_id, tokenizer.pad_token_id, trainer.MAX_RESPONSE)
    live = trainer.resolved_contract(student, gen, trainer.SUPPORT, device)
    if dict.fromkeys(trainer.DOMAINS, live) != hard['generation']:
        raise ValueError('Live generation contract differs from checkpoint-2')
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                  weight_decay=0.01, eps=1e-8, fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    return student, optimizer, saved


def run(args, evidence):
    device = torch.device('cuda', 0)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expected one BF16-compatible CUDA device')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def emit(phase, **details):
        row = {'phase': phase, 'timestamp': time.time(), **details,
               'allocated_bytes': torch.cuda.memory_allocated(device),
               'reserved_bytes': torch.cuda.memory_reserved(device),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        line = json.dumps(row)
        with (evidence / 'phase_metrics.jsonl').open('a') as log:
            print(line, file=log)
        print(line, flush=True)

    manifest, checkpoint = args.manifest.resolve(), args.checkpoint.resolve()
    assets = trainer.opd.ROOT / 'runs/latent_mopd_three_assets'
    emit('tokenizer_audit_start')
    tokenizer, tokenizer_path, audit = trainer.tokenizer_audit(assets)
    tokenizer_path, audit = tokenizer_path.resolve(), audit.resolve()
    prefixes, lengths, plan = build_plan(manifest, tokenizer, args.stress_rollout)
    write(evidence / 'workload.json', plan)
    emit('checkpoint_restore_start')
    student, optimizer, saved = load_inputs(checkpoint, manifest, tokenizer_path, audit, tokenizer, device)
    emit('checkpoint_restored', adamw_parameter_states=len(optimizer.state))
    teachers = {}
    for domain, spec in trainer.TEACHERS.items():
        emit('teacher_load_start', domain=domain)
        teacher_path = trainer.stage_weights(spec, assets / 'weights', args.model_drive)
        teachers[domain] = AutoModelForCausalLM.from_pretrained(
            teacher_path, dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True, device_map={'': device},
        ).eval().requires_grad_(False)
        emit('teacher_loaded', domain=domain)
    trainer.opd.restore_rng(saved)
    del saved
    rollout = stress_rollout(student, tokenizer, prefixes, plan['domain'], device, emit) if args.stress_rollout else None
    if rollout is not None:
        write(evidence / 'stress_rollout.json', rollout)
    missing = [(name, p) for name, p in student.named_parameters()
               if p.requires_grad and p not in optimizer.state]
    extra_moments = [torch.zeros_like(p) for _, p in missing for _ in range(2)]
    if any(t.device != device for t in extra_moments):
        raise ValueError('Missing AdamW moments were not reserved on CUDA')
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    emit('moments_reserved', names=[name for name, _ in missing],
         bytes=sum(t.numel() * t.element_size() for t in extra_moments))
    sequences = synthetic_sequences(prefixes, lengths, tokenizer)
    groups = [part['indices'] for part in plan['groups']]
    compute_started = time.monotonic()
    try:
        stats = joint_backward(student, teachers[plan['domain']], sequences, prefixes, groups,
                               (plan['token_weight'], plan['latent_weight']), tokenizer.pad_token_id,
                               device, emit)
        if any(p.grad is None for _, p in missing):
            raise ValueError('Joint backward did not exercise all reserved missing AdamW states')
        torch.cuda.synchronize(device)
        result = {**stats, 'restored_step': 2, 'target_step': STEP, 'domain': plan['domain'],
                  'synthetic_responses': True, 'stress_rollout': rollout,
                  'responses': plan['responses'], 'response_tokens': plan['response_tokens'],
                  'truncated_responses': plan['truncated_responses'], 'microbatches': len(groups),
                  'token_weight': plan['token_weight'], 'latent_weight': plan['latent_weight'],
                  'resident_teachers': list(teachers), 'checkpoint_manifest_sha256': CHECKPOINT_SHA,
                  'reserved_missing_adam_moment_bytes': sum(t.numel() * t.element_size() for t in extra_moments),
                  'reserved_missing_adam_moment_names': [name for name, _ in missing],
                  'device': torch.cuda.get_device_name(device), 'device_total_bytes': torch.cuda.get_device_properties(device).total_memory,
                  'baseline_allocated_bytes': baseline, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                  'compute_seconds': time.monotonic() - compute_started, 'total_seconds': time.monotonic() - started}
        write(evidence / 'step4_probe.json', result)
        emit('probe_complete', **stats)
    finally:
        student.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-drive', default='mydrive:models')
    parser.add_argument('--stress-rollout', action='store_true',
                        help='Run one forced-cap rollout batch before joint backward and fill packing groups')
    parser.add_argument('--plan-tokenizer', type=Path,
                        help='Write only the workload plan using a local tokenizer and no CUDA')
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.plan_tokenizer is None):
        parser.error('Specify either --checkpoint or --plan-tokenizer')
    if args.plan_tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.plan_tokenizer, local_files_only=True)
        _, _, plan = build_plan(args.manifest, tokenizer, args.stress_rollout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, plan)
        print(f"{plan['responses']} responses, {plan['response_tokens']} tokens, {len(plan['groups'])} microbatches")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        run(args, args.output)


if __name__ == '__main__':
    main()
