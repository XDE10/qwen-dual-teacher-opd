"""Measure real 128-response rollouts and joint backward from checkpoint-2 without updating it."""

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed

from scripts import probe_latent_three_step4_full_batch as joint
from src.training.runtime_v2 import write


class DeadlineReached(Exception):
    pass


def following_cases():
    return [(4, True, 42), (4, False, 42), (5, True, 42), (6, True, 42)]


def run_case(student, optimizer, teachers, tokenizer, schedule, spec, device, emit):
    trainer = joint.trainer
    step, cache, seed = spec
    domain, prompt_prefixes, sample_ids = schedule[step - 1]
    prefixes = [prefix for prefix in prompt_prefixes for _ in range(trainer.RESPONSES)]
    weights = trainer.loss_weights('latent', domain, (step - 1) // len(trainer.DOMAINS))
    if len(prefixes) != 128 or weights != (0.1, 0.9):
        raise ValueError('Expected a full first-joint-weight domain batch')

    student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    gen = trainer.generation_config(tokenizer.eos_token_id, tokenizer.pad_token_id, trainer.MAX_RESPONSE)
    contract = trainer.resolved_contract(student, gen, trainer.SUPPORT, device)
    engine = trainer.HFRolloutEngine(student, {domain: gen}, {domain: contract}, trainer.SUPPORT,
                                     tokenizer.pad_token_id, tokenizer.eos_token_id, trainer.RESPONSES)
    emit('rollout_start', responses=0, total=128)
    started = time.monotonic()
    rollouts = []
    for start in range(0, len(prefixes), trainer.RESPONSES):
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=cache):
            rows = engine.generate(prefixes[start:start + trainer.RESPONSES], domain)
        rollouts.extend(rows)
        emit('rollout_progress', responses=len(rollouts), total=128,
             batch_response_lengths=[len(row['response_ids']) for row in rows])
    rollout_seconds = time.monotonic() - started
    if len(rollouts) != len(prefixes) or any(
        row['prompt_ids'] != prefix or not row['response_ids']
        for prefix, row in zip(prefixes, rollouts, strict=True)
    ):
        raise ValueError('Incomplete rollout or unexpected prompt tokens')
    lengths = [len(row['response_ids']) for row in rollouts]
    digests = [hashlib.sha256(json.dumps(row['response_ids'], separators=(',', ':')).encode()).hexdigest()
               for row in rollouts]
    eos = [row['ended_with_eos'] for row in rollouts]
    rollout_stats = {'target_step': step, 'domain': domain, 'cache_enabled': cache, 'seed': seed,
                     'sample_ids': sample_ids, 'response_lengths': lengths, 'response_sha256': digests,
                     'ended_by_eos': eos, 'response_tokens': sum(lengths),
                     'truncated_responses': eos.count(False), 'rollout_seconds': rollout_seconds,
                     'rollout_tokens_per_second': sum(lengths) / rollout_seconds,
                     'rollout_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                     'rollout_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    emit('rollout_done', **rollout_stats)
    # The on/off math cases compare generation. The cached cases also test joint backward.
    if not cache:
        return {**rollout_stats, 'optimizer_updates': 0, 'total_seconds': time.monotonic() - started}

    missing = [p for p in student.parameters() if p.requires_grad and p not in optimizer.state]
    extra_moments = [torch.zeros_like(p) for p in missing for _ in range(2)]
    reserved_bytes = sum(t.numel() * t.element_size() for t in extra_moments)
    emit('moments_reserved', bytes=reserved_bytes)
    sequences = [prefix + row['response_ids'] for prefix, row in zip(prefixes, rollouts, strict=True)]
    groups = list(trainer.microbatches(list(map(len, sequences))))
    if sorted(i for g in groups for i in g) != list(range(128)):
        raise ValueError('Production packing dropped or duplicated responses')
    update_started = time.monotonic()
    try:
        stats = joint.joint_backward(student, teachers[domain], sequences, prefixes, groups, weights,
                                     tokenizer.pad_token_id, device, emit)
        if any(p.grad is None for p in missing):
            raise ValueError('Joint backward did not exercise the reserved AdamW state')
        if sum(t.numel() * t.element_size() for t in extra_moments) != reserved_bytes:
            raise ValueError('AdamW reserve changed before backward completed')
        return {**rollout_stats, **stats, 'microbatches': len(groups),
                'joint_compute_seconds': time.monotonic() - update_started,
                'token_weight': weights[0], 'latent_weight': weights[1],
                'reserved_missing_adam_moment_bytes': reserved_bytes,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'total_seconds': time.monotonic() - started}
    finally:
        student.zero_grad(set_to_none=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'checkpoint', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--early-finish', required=True, help='ISO timestamp for the start of the desired finish window')
    p.add_argument('--deadline', required=True, help='ISO timestamp for ceasing new batches')
    p.add_argument('--resume-report', type=Path, help='Read completed immutable cases from the prior run report')
    args = p.parse_args()
    early_finish = datetime.fromisoformat(args.early_finish)
    deadline = datetime.fromisoformat(args.deadline)
    if early_finish.tzinfo is None or deadline.tzinfo is None or not early_finish < deadline:
        p.error('Provide a time-zone-aware finish window')
    if deadline.timestamp() <= time.time():
        p.error('The finish window has elapsed')
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = joint.trainer
    device = torch.device('cuda', 0)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expected one BF16-compatible CUDA device')
    extra_cases = [(4, True, 43), (4, False, 43), (5, True, 43), (6, True, 43)]
    report = {'checkpoint_manifest_sha256': joint.CHECKPOINT_SHA, 'optimizer_updates': 0,
              'desired_finish_from': args.early_finish, 'batch_deadline': args.deadline,
              'scheduled_cases': following_cases(), 'optional_cases': list(extra_cases),
              'matched_math_cache_comparisons': [], 'cases': [], 'status': 'initializing'}
    if args.resume_report is not None:
        previous = json.loads(args.resume_report.read_text())
        keys = ('checkpoint_manifest_sha256', 'optimizer_updates', 'desired_finish_from', 'batch_deadline')
        if any(previous[k] != report[k] for k in keys):
            raise ValueError('Previous report does not match this run')
        expected = following_cases() + extra_cases
        actual = [(case['target_step'], case['cache_enabled'], case['seed']) for case in previous['cases']]
        if actual != expected[:len(actual)] or any(
            len(case['response_sha256']) != 128 or case['optimizer_updates'] != 0
            for case in previous['cases']
        ):
            raise ValueError('Previous report contains an invalid prefix of completed cases')
        report['cases'] = previous['cases']
        report['matched_math_cache_comparisons'] = previous['matched_math_cache_comparisons']
        report['cases_reused'] = len(actual)
        extra_cases = extra_cases[max(0, len(actual) - len(following_cases())):]
    report_path = args.output / 'overnight_results.json'
    write(report_path, report)
    active = None

    def emit(phase, **detail):
        row = {'phase': phase, 'case': active, 'timestamp': time.time(), **detail,
               'allocated_bytes': torch.cuda.memory_allocated(device),
               'reserved_bytes': torch.cuda.memory_reserved(device)}
        line = json.dumps(row)
        with (args.output / 'phase_metrics.jsonl').open('a') as stream:
            print(line, file=stream)
        preview = {k: v for k, v in row.items() if k not in ('response_sha256', 'response_lengths', 'sample_ids', 'ended_by_eos')}
        print(json.dumps(preview), flush=True)
        if phase.endswith('_start') or phase == 'rollout_progress':
            if time.time() >= deadline.timestamp():
                raise DeadlineReached

    try:
        if trainer.sha(args.manifest) != joint.MANIFEST_SHA:
            raise ValueError('Expected the sealed poolfix release')
        assets = trainer.opd.ROOT / 'runs/latent_mopd_three_assets'
        emit('tokenizer_start')
        tokenizer, tokenizer_path, audit = trainer.tokenizer_audit(assets)
        schedule = trainer.read_schedule(args.manifest, tokenizer, 6, trainer.PROMPTS)
        frozen = args.manifest.with_name('schedule.jsonl')
        if trainer.sha(frozen) != trainer.read(args.manifest)['files'][frozen.name]['sha256']:
            raise ValueError('Schedule differs from the sealed release')
        rows = [json.loads(line) for line in frozen.read_text().splitlines()[:6]]
        for number, ((domain, _, ids), row) in enumerate(zip(schedule, rows, strict=True), 1):
            if (number, domain, ids) != (row['step'], row['domain'], row['sample_ids']):
                raise ValueError(f'Step {number} differs from the approved schedule')
        emit('checkpoint_restore_start')
        student, optimizer, saved = joint.load_inputs(args.checkpoint.resolve(), args.manifest.resolve(),
                                                        tokenizer_path.resolve(), audit.resolve(), tokenizer, device)
        del saved
        teachers = {}
        for domain, spec in trainer.TEACHERS.items():
            emit('teacher_load_start', domain=domain)
            weights = trainer.stage_weights(spec, assets / 'weights', 'mydrive:models')
            teachers[domain] = AutoModelForCausalLM.from_pretrained(
                weights, dtype=torch.bfloat16, attn_implementation='sdpa',
                local_files_only=True, device_map={'': device},
            ).eval().requires_grad_(False)
        report['resident_teachers'] = list(teachers)
        report['status'] = 'running'
        pending = following_cases()[len(report['cases']):]

        def add_optional_case():
            if pending or not extra_cases or time.time() >= early_finish.timestamp():
                return
            candidate = extra_cases.pop(0)
            seconds = [c['total_seconds'] for c in report['cases'] if c['cache_enabled'] == candidate[1]]
            if time.time() + statistics.median(seconds) * 1.1 < deadline.timestamp():
                pending.append(candidate)

        if report['cases']:
            add_optional_case()
        while pending:
            active = pending.pop(0)
            emit('case_start')
            result = run_case(student, optimizer, teachers, tokenizer, schedule, active, device, emit)
            report['cases'].append(result)
            if not result['cache_enabled']:
                cached = next(case for case in report['cases']
                              if case['target_step'] == active[0] and case['seed'] == active[2]
                              and case['cache_enabled'])
                report['matched_math_cache_comparisons'].append({
                    'seed': active[2],
                    'identical_token_ids_by_response': cached['response_sha256'] == result['response_sha256'],
                    'identical_eos': cached['ended_by_eos'] == result['ended_by_eos'],
                    'throughput_ratio_on_over_off': cached['rollout_tokens_per_second'] / result['rollout_tokens_per_second'],
                })
            write(report_path, report)
            emit('case_done', case_seconds=result['total_seconds'])
            add_optional_case()
        report['status'] = 'completed'
    except DeadlineReached:
        report['status'] = 'stopped_at_deadline'
        report['partial_case'] = active
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        report['partial_case'] = active
        raise
    finally:
        write(report_path, report)
    print('LATENT_THREE_OVERNIGHT_PROBE_COMPLETE', report['status'], len(report['cases']), flush=True)


if __name__ == '__main__':
    main()
