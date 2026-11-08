"""OPD v2: exact sampling, bounded KL workspace, token weighting, sealed resume."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

from src.data.approved_training_inputs import load_approved, model_view
from src.teachers.opd_contract import render_student_prompt, teacher_batch_from_student
from src.training.opd import response_batch
from src.training.runtime_v2 import (read, write, build_identity, seal_checkpoint,
    validate_checkpoint, prepare_output, PhaseRecorder, teacher_key)

ROOT = Path(__file__).resolve().parents[2]


def update_loss(student, teacher, ids, attention, prefix, support,
                sequence_chunk=128, row_chunk=16, recorder=None):
    import torch
    from src.losses.chunked_opd_v2 import chunked_reverse_kl, removed_mass_values
    recorder = recorder or PhaseRecorder()
    batch, mask = teacher_batch_from_student(ids, attention, [len(prefix)], [prefix], support)
    with recorder.phase('student_forward'):
        s = student(**batch, use_cache=False).logits
    with recorder.phase('teacher_forward'):
        teacher.eval()
        with torch.no_grad():
            t = teacher(**batch, use_cache=False).logits
    with recorder.phase('kl_and_mass'):
        loss = chunked_reverse_kl(s, t, mask, support, sequence_chunk, row_chunk)
        mass = {name: removed_mass_values(logits, mask, support, sequence_chunk, row_chunk)
                for name, logits in [('student', s), ('teacher', t)]}
    return loss, mass, int(mask[:, 1:].sum())


def save_checkpoint(path, student, optimizer, scheduler, state, identity, telemetry):
    import numpy as np
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(path / 'student', safe_serialization=True)
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'state': state, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, path / 'engine.pt')
    write(path / 'run_identity.json', identity)
    write(path / 'telemetry.json', telemetry)
    seal_checkpoint(path)


def run(config_path, output_root=None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model, PeftModel
    from src.training.sampling_v2 import generation_config, resolved_contract, generate_exact
    from src.losses.chunked_opd_v2 import summarize_mass, gradient_normalizer

    c = read(config_path)
    if c.get('implementation_version') != 'opd-v2':
        raise ValueError('Use a versioned opd-v2 configuration')
    c.setdefault('loss_normalization', 'response_tokens_v2')
    if c['loss_normalization'] not in {'response_tokens_v2', 'microbatch_mean_v1'}:
        raise ValueError('Unknown loss normalization')
    output_root = output_root or os.environ.get('OPD_OUTPUT_ROOT')
    if output_root:
        c['output_dir'] = (Path(output_root) / Path(c['output_dir']).name).as_posix()
    out = Path(c['output_dir'])
    pools = {domain: load_approved(c['data_manifest'], name) for domain, name in c['data_files'].items()}
    if c.get('training_exclusions'):
        from src.data.training_exclusions_v3 import filter_training_pools
        pools = filter_training_pools(pools, c['data_manifest'], c['training_exclusions'])
    protected_path = Path(c['data_manifest']).parent
    m = read(c['data_manifest'])
    protected = set(read(protected_path / m['protected_dev_ids_file'])['sample_ids'])
    seen = set()
    for domain, rows in pools.items():
        for row in rows:
            if row['domain'] != domain or row['sample_id'] in protected or row['sample_id'] in seen:
                raise ValueError('Domain/duplicate/protected training ID')
            seen.add(row['sample_id'])
    audit = read(c['tokenizer_audit'])
    support = len(audit['common_support']['ids'])
    if audit['common_support']['ids'] != list(range(support)) or audit['mapping_difference_count']:
        raise ValueError('Requires identical audited contiguous token IDs')
    expected = {r['model']: r['revision'] for r in audit['models']}
    for spec in [c['student'], *c['teachers'].values()]:
        if expected.get(spec['model']) != spec['revision']:
            raise ValueError('Model revision differs from audit')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required; CPU tests use tiny models only')
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['tokenizer_path'], local_files_only=True)
    tok.padding_side = 'left'
    recorder = PhaseRecorder()
    spec = c['student']
    with recorder.phase('student_load'):
        student = AutoModelForCausalLM.from_pretrained(spec['model'], revision=spec['revision'],
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    configs = {d: generation_config(tok.eos_token_id, tok.pad_token_id, n)
               for d, n in c['max_new_tokens'].items()}
    generation = {d: resolved_contract(student, config, support, student.device)
                  for d, config in configs.items()}
    identity = build_identity(c, generation, ROOT)
    resume = c.get('resume_from_checkpoint')
    saved = None
    if resume:
        validate_checkpoint(resume, identity)  # All files verified BEFORE torch.load.
        saved = torch.load(Path(resume) / 'engine.pt', map_location='cpu', weights_only=False)
    prepare_output(out, identity, saved['state']['step'] if saved else None, c.get('resume_policy', 'reject'))
    write(out / 'resolved_generation.json', generation)
    write(out / ('startup_' + str(time.time_ns()) + '.json'), recorder.records)
    if saved:
        student = PeftModel.from_pretrained(student, Path(resume) / 'student', is_trainable=True)
    else:
        student = get_peft_model(student, LoraConfig(r=c.get('lora_r', 16), lora_alpha=32,
            lora_dropout=0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.enable_input_require_grads()
    student.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=c['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0., 1 - s / c['max_steps']))
    state = {'step': 0, 'positions': {d: 0 for d in pools},
             'exposure': {d: {'prompts': 0, 'response_tokens': 0} for d in pools}}
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['state']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    teacher = None
    current_key = None
    domains = c['domain_order']
    accum = c.get('gradient_accumulation_steps', 4)
    if accum < 1:
        raise ValueError('Positive gradient_accumulation_steps required')
    with (out / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for step in range(state['step'], c['max_steps']):
            recorder.records = []
            wall_start = time.monotonic()
            domain = domains[(step // c.get('domain_block_steps', 10)) % len(domains)]
            spec = c['teachers'][domain]
            switched = current_key != teacher_key(spec)
            if switched:
                with recorder.phase('teacher_unload'):
                    if teacher is not None:
                        del teacher
                        teacher = None
                        gc.collect()
                        torch.cuda.empty_cache()
                # Also preserve Python/NumPy RNG used by possible loading internals.
                py_rng, np_rng = random.getstate(), np.random.get_state()
                try:
                    with recorder.phase('teacher_load'), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        teacher = AutoModelForCausalLM.from_pretrained(spec['model'], revision=spec['revision'],
                            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
                finally:
                    random.setstate(py_rng)
                    np.random.set_state(np_rng)
                current_key = teacher_key(spec)
            optimizer.zero_grad(set_to_none=True)
            total_tokens, numerator = 0, 0.0
            masses = {'student': [], 'teacher': []}
            samples = []
            for _ in range(accum):
                position = state['positions'][domain]
                row = pools[domain][position % len(pools[domain])]
                state['positions'][domain] += 1
                view = model_view(row, 'opd')
                prefix = render_student_prompt(tok, view['input']['prompt'], domain)
                x = torch.tensor([prefix], dtype=torch.long, device='cuda')
                student.eval()
                with recorder.phase('rollout'), torch.no_grad():
                    sample = generate_exact(student, x, torch.ones_like(x), configs[domain], support,
                                            generation[domain]['logits_processors'])
                full, attention, _ = response_batch(sample, prefix, tok.eos_token_id)
                student.train()
                loss, mass, count = update_loss(student, teacher, full, attention, prefix, support,
                    c.get('sequence_chunk_size', 128), c.get('response_row_chunk_size', 16), recorder)
                token_mode = c['loss_normalization'] == 'response_tokens_v2'
                weight = count if token_mode else 1
                with recorder.phase('backward'):
                    (loss * weight).backward()
                numerator += loss.detach().item() * weight
                total_tokens += count
                for name in masses:
                    masses[name].append(mass[name])
                samples.append({'sample_id': row['sample_id'], 'position': position,
                    'interface': row.get('interface'), 'prompt_tokens': len(prefix),
                    'response_tokens': count, 'ended_with_eos': int(full[0, -1]) == tok.eos_token_id})
                del loss, sample, full, attention, x
            denominator = gradient_normalizer(c['loss_normalization'], total_tokens, accum)
            with recorder.phase('optimizer'):
                for p in student.parameters():
                    if p.grad is not None:
                        p.grad.div_(denominator)
                norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            state['step'] = step + 1
            state['exposure'][domain]['prompts'] += accum
            state['exposure'][domain]['response_tokens'] += total_tokens
            if state['step'] % c.get('save_steps', 20) == 0 or state['step'] == c['max_steps']:
                with recorder.phase('checkpoint_save'):
                    save_checkpoint(out / f"checkpoint-{state['step']}", student, optimizer, scheduler, state, identity, recorder.records)
            compute_names = {'rollout', 'student_forward', 'teacher_forward', 'kl_and_mass', 'backward', 'optimizer'}
            record = {'step': state['step'], 'domain': domain, 'teacher_identity': list(current_key),
                'teacher_reloaded': switched, 'loss': numerator / denominator, 'gradient_norm': float(norm),
                'loss_normalization': c['loss_normalization'], 'samples': samples, 'exposure': state['exposure'],
                'total_wall_seconds': time.monotonic() - wall_start,
                'pure_compute_seconds': sum(r['seconds'] for r in recorder.records if r['phase'] in compute_names),
                'mass': {k: summarize_mass(v) for k, v in masses.items()}, 'phases': recorder.records}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(record, flush=True)
    return state


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', help='e.g. /content/drive/MyDrive/project2/runs')
    args = parser.parse_args()
    run(args.config, args.output_root)
