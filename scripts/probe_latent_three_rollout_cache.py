"""Compare HF generation with BF16 autocast weight caching disabled and enabled."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed

from src.training.latent_mopd_three import (
    DOMAINS, MAX_RESPONSE, PROMPTS, RESPONSES, STEPS, STUDENT, SUPPORT, read_schedule, stage_weights, tokenizer_audit,
)
from src.training.opd_v3 import ROOT
from src.training.rollout_v3 import HFRolloutEngine
from src.training.runtime_v2 import sha
from src.training.sampling_v2 import generation_config, resolved_contract


def run_arm(engine, prompts, domain, enabled, seed):
    set_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=enabled):
        rows = engine.generate(prompts, domain)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    responses = [row['response_ids'] for row in rows]
    counts = [len(ids) for ids in responses]
    if len(responses) != len(prompts) or not all(counts) or any(max(ids) >= SUPPORT for ids in responses):
        raise ValueError('Missing responses or invalid rollout token IDs')
    return responses, {
        'autocast_cache': enabled, 'seconds': seconds, 'tokens': sum(counts),
        'tokens_per_second': sum(counts) / seconds, 'lengths': counts,
        'ended_by_eos': [row['ended_with_eos'] for row in rows],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


@torch.no_grad()
def fixed_log_probs(student, prompt, generated, enabled):
    scores = []
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=enabled):
        for length in (0, min(8, len(generated))):
            ids = torch.tensor([prompt + generated[:length]], device=student.device)
            logits = student(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :SUPPORT]
            scores.append(logits.float().log_softmax(-1).cpu())
    return torch.cat(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', choices=DOMAINS, default='code')
    parser.add_argument('--prompt-index', type=int, default=1)
    parser.add_argument('--new-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-drive', default='mydrive:models')
    args = parser.parse_args()
    if not 1 <= args.prompt_index <= PROMPTS or not 1 <= args.new_tokens <= MAX_RESPONSE:
        parser.error('Use a valid formal prompt index and a response cap in 1..16384')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This probe requires one visible BF16 CUDA GPU')

    assets = ROOT / 'runs/latent_mopd_three_assets'
    tokenizer, _, audit = tokenizer_audit(assets)
    schedule = read_schedule(args.manifest, tokenizer, STEPS, PROMPTS)
    _, prefixes, sample_ids = next(row for row in schedule if row[0] == args.domain)
    prompt, sample_id = prefixes[args.prompt_index - 1], sample_ids[args.prompt_index - 1]
    prompts = [prompt] * RESPONSES
    device = torch.device('cuda')
    checkpoint = stage_weights(STUDENT, assets / 'weights', args.model_drive)
    student = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation='sdpa', local_files_only=True,
        device_map={'': device},
    ).eval()
    student.config.use_cache = False

    def engine(cap):
        generation = generation_config(tokenizer.eos_token_id, tokenizer.pad_token_id, cap)
        contract = resolved_contract(student, generation, SUPPORT, device)
        return HFRolloutEngine(student, {args.domain: generation}, {args.domain: contract}, SUPPORT,
                               tokenizer.pad_token_id, tokenizer.eos_token_id, RESPONSES)

    warmup = engine(min(32, args.new_tokens))
    for enabled in (False, True):
        run_arm(warmup, prompts, args.domain, enabled, args.seed)
    rollout = engine(args.new_tokens)
    measurements, generated = [], []
    for enabled in (False, True, True, False):
        responses, record = run_arm(rollout, prompts, args.domain, enabled, args.seed)
        measurements.append(record)
        generated.append(responses)
        print(json.dumps(record), flush=True)

    off = fixed_log_probs(student, prompt, generated[0][0], False)
    on = fixed_log_probs(student, prompt, generated[0][0], True)
    if not torch.isfinite(off).all() or not torch.isfinite(on).all():
        raise ValueError('Nonfinite fixed-prefix log probabilities')
    rates = {enabled: statistics.median(row['tokens_per_second'] for row in measurements
                                        if row['autocast_cache'] == enabled) for enabled in (False, True)}
    result = {
        'student': STUDENT, 'manifest_sha256': sha(args.manifest), 'tokenizer_audit_sha256': sha(audit),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'support': SUPPORT,
        'sample_id': sample_id, 'domain': args.domain, 'concurrent_responses': RESPONSES,
        'seed': args.seed, 'max_new_tokens': args.new_tokens, 'optimizer_updates': 0,
        'measurements': measurements, 'identical_sampled_token_ids': all(rows == generated[0] for rows in generated),
        'fixed_prefix_logprob_max_abs_diff': float((off - on).abs().max()),
        'median_tokens_per_second_cache_off': rates[False], 'median_tokens_per_second_cache_on': rates[True],
        'throughput_ratio_cache_on_over_off': rates[True] / rates[False],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
