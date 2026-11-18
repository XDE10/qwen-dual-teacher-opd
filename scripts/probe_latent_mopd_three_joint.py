"""Measure 16K joint-loss memory and backward on fixed IDs, without an optimizer update."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.losses.latent_opd import loss_weights, response_batch, reward_scale
from src.training import opd_v3 as opd
from src.training.latent_mopd_three import (
    MAX_RESPONSE, PACK_TOKENS, PROMPTS, RESPONSES, STEPS, STUDENT, SUPPORT, TEACHERS,
    objective, read_schedule, restore_checkpoint, score, stage_weights, tokenizer_audit,
)
from src.training.runtime_v2 import read, sha


def fixed_response(tokenizer):
    pattern = tokenizer.encode(' Follow the given assumptions and check each inference.',
                               add_special_tokens=False)
    repeats = (MAX_RESPONSE - 2) // len(pattern) + 1
    return (pattern * repeats)[:MAX_RESPONSE - 1] + [tokenizer.eos_token_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='32 × 4 single-step checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-drive', default='mydrive:models')
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The joint probe requires one visible BF16 CUDA GPU')
    assets = opd.ROOT / 'runs/latent_mopd_three_assets'
    tokenizer, _, _ = tokenizer_audit(assets)
    schedule = read_schedule(args.manifest, tokenizer, STEPS, PROMPTS)
    prefix, sample_id = max(
        ((ids, name) for domain, prefixes, names in schedule if domain == 'logic'
         for ids, name in zip(prefixes, names, strict=True)),
        key=lambda item: len(item[0]),
    )
    response = fixed_response(tokenizer)
    if len(prefix) + len(response) > PACK_TOKENS:
        raise ValueError('Probe sequence exceeds the training packing budget')

    identity = read(args.checkpoint / 'run_identity.json')
    hard = identity['hard']
    config = hard['config']
    expected = {'implementation': 'three-teacher-full-parameter', 'variant': 'latent', 'student': STUDENT,
                'teachers': TEACHERS, 'prompts': PROMPTS, 'responses': RESPONSES,
                'max_new_tokens': MAX_RESPONSE}
    if (args.checkpoint.name != 'checkpoint-1' or hard['manifest_sha'] != sha(args.manifest)
            or config.get('steps') not in (1, STEPS)
            or any(config.get(k) != v for k, v in expected.items())):
        raise ValueError('Probe requires the matching approved-release 32 × 4 single-step checkpoint')

    device = torch.device('cuda')
    print('Verifying single-step checkpoint', flush=True)
    saved = restore_checkpoint(args.checkpoint, identity)
    student = AutoModelForCausalLM.from_pretrained(
        args.checkpoint / 'student', dtype=torch.float32, attn_implementation='sdpa', local_files_only=True,
        device_map={'': device},
    )
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                  weight_decay=0.01, eps=1e-8, fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    if not optimizer.state:
        raise ValueError('Single-step checkpoint lacks initialized AdamW state')
    del saved
    # A pure latent step may not initialize moments for parameters such as the LM head.
    # Reserve the equivalent storage while measuring the joint step's backward.
    extra_moments = [torch.empty_like(p) for p in student.parameters()
                     if p.requires_grad and p not in optimizer.state for _ in range(2)]
    teachers = {
        domain: AutoModelForCausalLM.from_pretrained(
            stage_weights(spec, assets / 'weights', args.model_drive), dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True, device_map={'': device},
        ).eval().requires_grad_(False)
        for domain, spec in TEACHERS.items()
    }

    ids, mask = opd.right_pad([prefix + response], tokenizer.pad_token_id, device)
    batch = response_batch(ids, mask, [prefix], SUPPORT)
    token_weight, latent_weight = loss_weights('latent', 'logic', 1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    started = time.monotonic()
    print(f'Candidate scoring: {len(prefix)} prompt tokens and {len(response)} fixed response tokens', flush=True)
    student.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        candidates, rewards = score(student, teachers['logic'], batch, SUPPORT)
    candidates, rewards = candidates.cpu(), rewards.cpu()
    scale = reward_scale([(candidates, rewards)])
    scoring_seconds = time.monotonic() - started

    print('Joint objective and backward', flush=True)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        loss, token_loss, rep_loss = objective(student, teachers['logic'], batch, candidates.to(device),
                                               (rewards / scale).to(device), token_weight, latent_weight, SUPPORT)
    loss.backward()
    grad_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in student.parameters() if p.grad is not None], error_if_nonfinite=True,
    )
    torch.cuda.synchronize(device)
    stats = {'loss': float(loss.detach()), 'token_loss': float(token_loss), 'raw_rep_loss': float(rep_loss),
             'reward_std': float(scale), 'gradient_norm': float(grad_norm)}
    if (not all(math.isfinite(value) for value in stats.values())
            or stats['gradient_norm'] <= 0 or stats['raw_rep_loss'] <= 0):
        raise RuntimeError(f'Invalid joint objective or gradients: {stats}')
    result = {
        **stats, 'probe': 'joint_16k_fixed_response', 'domain': 'logic', 'sample_id': sample_id,
        'prompt_tokens': len(prefix), 'response_tokens': len(response), 'optimizer_updates': 0,
        'token_weight': token_weight, 'latent_weight': latent_weight, 'resident_teachers': list(teachers),
        'checkpoint_manifest_sha256': sha(args.checkpoint / 'checkpoint_manifest.json'),
        'device': torch.cuda.get_device_name(device), 'baseline_allocated_bytes': baseline_bytes,
        'reserved_missing_adam_moment_bytes': sum(t.numel() * t.element_size() for t in extra_moments),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'scoring_seconds': scoring_seconds, 'total_seconds': time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    print('LATENT_THREE_PROBE_OK joint16k', flush=True)


if __name__ == '__main__':
    main()
