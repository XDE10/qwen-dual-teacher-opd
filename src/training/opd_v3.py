"""opd-v3: MOPD-style dual-teacher on-policy distillation on a single GPU.

Changes from opd-v2 (which stays unchanged for provenance):

* Each optimizer step mixes domains, e.g. 32 math + 32 code prompts. Every
  rollout is routed to its domain teacher. Block alternation is kept only as
  the ``schedule.type = blocked`` ablation.
* Teachers stay resident on 80/96 GB cards, or move in from host memory
  inside a step (``cpu_swap``). Nothing is reloaded from disk per block.
* Rollouts are batched with left padding through the audited sampling-v2 chain,
  on a frozen generation copy that receives the FP32 out-of-place LoRA merge.
* Logits are computed only on response rows and only on the audited support.
* Losses: exact full-support reverse KL, MOPD top-k, or MOPD policy gradient.
  Optional proximal teacher (TROPD). Normalisations: token, sequence (MOPD
  1/|y|) or per-domain token mean. Optional math/code gradient cosine and PCGrad.
* Seeded stratified sampler (v2 read files in order) and prompt budgets checked
  before any GPU work.
* Hard/soft run identity (runtime_v3), so a new Colab image does not block resume.

CPU tests exercise all of this with tiny Qwen2 models. Real Qwen models, GPU
memory, throughput and capability gains are NOT verified by those tests.
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from src.data.approved_training_inputs import load_approved, model_view
from src.teachers.opd_contract import load_contract, render_student_prompt
from src.training.runtime_v2 import PhaseRecorder, read, write

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ('math', 'code')
NORMALIZATIONS = ('token_mean', 'sequence_mean', 'domain_token_mean')
SCHEDULES = ('mixed', 'blocked')
ENGINES = ('hf_generation_copy', 'hf_peft')
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
DEFAULTS = {
    'schedule': {'type': 'mixed'},
    'stratify_keys': {'math': 'source', 'code': 'interface'},
    'teacher_placement': 'resident',
    'rollout_engine': 'hf_generation_copy',
    'rollout_batch_size': 64,
    'train_microbatch_tokens': 16384,
    'loss': {'mode': 'exact_rkl', 'topk': 64, 'pg_advantage_clip': 10.0, 'proximal_teacher_alpha': 1.0,
             'normalization': 'sequence_mean', 'domain_weights': None, 'row_chunk': 64},
    'gradient_telemetry': True,
    'gradient_surgery': 'none',
    'lora': {'r': 16, 'alpha': 32, 'dropout': 0.0, 'target_modules': LORA_TARGETS},
    'learning_rate': 2e-5,
    'warmup_steps': 0,
    'lr_schedule': 'constant',
    'max_grad_norm': 1.0,
    'save_steps': 10,
    'save_rollouts_every': 10,
    'rollouts_saved_per_domain': 2,
    'resume_policy': 'reject',
}


def with_defaults(config):
    c = json.loads(json.dumps(config))
    for key, value in DEFAULTS.items():
        if isinstance(value, dict):
            c[key] = {**value, **c.get(key, {})}
        else:
            c.setdefault(key, value)
    return c


def validate_config(c, contract):
    if c.get('implementation_version') != 'opd-v3':
        raise ValueError('Use a versioned opd-v3 configuration')
    domains = list(c['prompts_per_step'])
    if not domains or not set(domains) <= set(DOMAINS):
        raise ValueError('prompts_per_step must name math and/or code')
    for key in ('data_files', 'teachers', 'max_new_tokens'):
        if set(c[key]) != set(domains):
            raise ValueError(key + ' must cover exactly the scheduled domains')
    for d in domains:
        if not 0 < c['max_new_tokens'][d] <= contract[d + '_max_new_tokens']:
            raise ValueError('max_new_tokens exceeds the prompt contract reserve for ' + d)
        if c['prompts_per_step'][d] < 1:
            raise ValueError('Positive prompts_per_step required')
    loss = c['loss']
    if loss['mode'] not in ('exact_rkl', 'topk_rkl', 'pg') or loss['normalization'] not in NORMALIZATIONS:
        raise ValueError('Unknown loss mode or normalization')
    if not 0 < loss['proximal_teacher_alpha'] <= 1:
        raise ValueError('proximal_teacher_alpha must be in (0, 1]')
    if c['schedule']['type'] not in SCHEDULES or (c['schedule']['type'] == 'blocked' and c['schedule'].get('block_steps', 0) < 1):
        raise ValueError('Unknown schedule or missing block_steps')
    if c['rollout_engine'] not in ENGINES or c['gradient_surgery'] not in ('none', 'pcgrad'):
        raise ValueError('Unknown rollout engine or gradient surgery')
    if c['gradient_surgery'] == 'pcgrad' and (len(domains) < 2 or c['schedule']['type'] != 'mixed'):
        raise ValueError('PCGrad needs two domains in every step')
    if c['max_steps'] < 1 or c['rollout_batch_size'] < 1 or c['train_microbatch_tokens'] < 1:
        raise ValueError('Positive max_steps / batch sizes required')
    return c


def load_config(path):
    return validate_config(with_defaults(read(path)), load_contract())


def load_pools(c):
    """Approved rows per domain plus protected-ID, domain and duplicate checks."""
    if not Path(c['data_manifest']).exists():
        raise FileNotFoundError(f"{c['data_manifest']} not found. The v5 release exists only after the owner runs "
                                "scripts/approve_project2_release_v5.py (reports/project2_release_v5_20261108/README.md).")
    pools = {d: load_approved(c['data_manifest'], name) for d, name in c['data_files'].items()}
    manifest = read(c['data_manifest'])
    protected = set(read(Path(c['data_manifest']).parent / manifest['protected_dev_ids_file'])['sample_ids'])
    seen = set()
    for domain, rows in pools.items():
        for row in rows:
            if row['domain'] != domain or row['sample_id'] in protected or row['sample_id'] in seen:
                raise ValueError('Domain/duplicate/protected training ID: ' + row['sample_id'])
            seen.add(row['sample_id'])
    return pools


def audited_support(c):
    audit = read(c['tokenizer_audit'])
    support = len(audit['common_support']['ids'])
    if audit['common_support']['ids'] != list(range(support)) or audit['mapping_difference_count']:
        raise ValueError('Requires identical audited contiguous token IDs')
    expected = {r['model']: r['revision'] for r in audit['models']}
    for spec in [c['student'], *c['teachers'].values()]:
        if expected.get(spec['model']) != spec['revision']:
            raise ValueError('Model revision differs from tokenizer audit: ' + spec['model'])
    return support


def render_prefixes(pools, tokenizer, contract):
    """Render every prompt once before GPU work; list all over-budget rows at once."""
    prefixes, failures = {}, []
    for domain, rows in pools.items():
        for row in rows:
            try:
                prefixes[row['sample_id']] = render_student_prompt(
                    tokenizer, model_view(row, 'opd')['input']['prompt'], domain, contract)
            except ValueError as error:
                failures.append((row['sample_id'], str(error)))
    if failures:
        raise ValueError(f'{len(failures)} prompts fail the contract, e.g. {failures[:5]}')
    return prefixes


def step_plan(c, step):
    domains = list(c['prompts_per_step'])
    if c['schedule']['type'] == 'mixed':
        return {d: c['prompts_per_step'][d] for d in domains}
    active = domains[(step // c['schedule']['block_steps']) % len(domains)]
    return {active: sum(c['prompts_per_step'].values())}


def sequence_weights(lengths, mode, domain_weights=None):
    """Per-sequence weight applied to each of its response-token rows; rows sum to 1."""
    if any(n < 1 for v in lengths.values() for n in v):
        raise ValueError('Every rollout needs at least one response token')
    tokens = sum(sum(v) for v in lengths.values())
    sequences = sum(len(v) for v in lengths.values())
    if mode == 'token_mean':
        return {d: [1. / tokens] * len(v) for d, v in lengths.items()}
    if mode == 'sequence_mean':
        return {d: [1. / (n * sequences) for n in v] for d, v in lengths.items()}
    if mode == 'domain_token_mean':
        weights = {d: (domain_weights or {}).get(d, 1.) for d in lengths}
        total = sum(weights.values())
        return {d: [weights[d] / total / sum(v)] * len(v) for d, v in lengths.items()}
    raise ValueError('Unknown normalization ' + mode)


def pack_by_tokens(lengths, max_tokens):
    """Index groups whose padded size (count x longest) stays within max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, current, width = [], [], 0
    for i in order:
        new_width = max(width, lengths[i])
        if current and new_width * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, new_width = [], lengths[i]
        current.append(i)
        width = new_width
    if current:
        batches.append(current)
    return batches


def right_pad(sequences, pad_id, device):
    import torch
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[i, :len(sequence)] = 1
    return ids.to(device), attention.to(device)


def flat_grad(params):
    import torch
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])


def assign_flat_grad(params, vector):
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = vector[offset:offset + n].view_as(p).to(p.dtype).clone()
        offset += n


def pcgrad(a, b):
    """Gradient surgery (Yu et al., 2020) for two tasks; returns combined vector."""
    dot = float((a * b).sum())
    if dot >= 0:
        return a + b, False
    a2 = a - dot / max(float((b * b).sum()), 1e-30) * b
    b2 = b - dot / max(float((a * a).sum()), 1e-30) * a
    return a2 + b2, True


class DomainAccumulator:
    """Token-weighted telemetry for one domain within one step."""

    def __init__(self):
        import torch
        self.sums = Counter()
        self.tokens = 0
        self.deciles = [[0., 0] for _ in range(10)]
        self.removed = {'student': [], 'teacher': []}
        self.microbatches = 0
        self._torch = torch

    def add(self, detail, prompt_lengths, response_lengths):
        from src.losses.mopd_v3 import STAT
        torch = self._torch
        stats = detail['stats'].detach().float().cpu()
        b = detail['batch_index'].cpu()
        positions = detail['positions'].cpu()
        self.tokens += stats.shape[0]
        self.microbatches += 1
        for name, i in STAT.items():
            if not name.endswith('_lse'):
                self.sums[name] += float(stats[:, i].sum())
        prompt = torch.tensor(prompt_lengths)[b]
        response = torch.tensor(response_lengths)[b]
        bucket = (((positions + 1 - prompt).float() / response.float()) * 10).floor().clamp(0, 9).long()
        kl = stats[:, STAT['kl_exact']]
        for k in range(10):
            selected = bucket == k
            self.deciles[k][0] += float(kl[selected].sum())
            self.deciles[k][1] += int(selected.sum())
        self.removed['student'].append(detail['removed_student'].detach().float().cpu())
        self.removed['teacher'].append(detail['removed_teacher'].detach().float().cpu())

    def summary(self):
        torch = self._torch
        out = {name + '_token_mean': value / max(self.tokens, 1) for name, value in self.sums.items()}
        out['response_rows'] = self.tokens
        out['microbatches'] = self.microbatches
        out['kl_by_relative_position_decile'] = [s / n if n else None for s, n in self.deciles]
        for who, values in self.removed.items():
            x = torch.cat(values).double() if values else torch.zeros(1, dtype=torch.float64)
            out['removed_mass_' + who] = {'mean': float(x.mean()), 'max': float(x.max()),
                                          'p99': float(x.quantile(.99)) if x.numel() > 1 else float(x.max())}
        return out


class OPDTrainerV3:
    """One optimizer step = sample -> sync copy -> rollouts -> routed teacher scoring -> update."""

    def __init__(self, config, student, teachers, engine, samplers, prefixes, support, pad_id, device,
                 optimizer, scheduler, recorder=None):
        self.c, self.student, self.teachers, self.engine = config, student, teachers, engine
        self.samplers, self.prefixes, self.support = samplers, prefixes, support
        self.pad_id, self.device = pad_id, device
        self.optimizer, self.scheduler = optimizer, scheduler
        self.recorder = recorder or PhaseRecorder()
        self.params = [p for p in student.parameters() if p.requires_grad]
        domains = list(config['prompts_per_step'])
        self.state = {'step': 0, 'exposure': {d: {'prompts': 0, 'response_tokens': 0, 'eos': 0} for d in domains}}

    def train_step(self):
        import torch
        from src.losses.mopd_v3 import microbatch_loss
        c, phase = self.c, self.recorder.phase
        self.recorder.records = []
        wall = time.monotonic()
        step = self.state['step']
        plan = step_plan(c, step)
        rows = {d: self.samplers[d].next(n) for d, n in plan.items()}
        with phase('sync_generation_copy'):
            merged_layers = self.engine.sync(self.student)
        rollouts = {}
        for d in plan:
            with phase('rollout_' + d):
                rollouts[d] = self.engine.generate([self.prefixes[r['sample_id']] for r in rows[d]], d)
        lengths = {d: [len(x['response_ids']) for x in rollouts[d]] for d in plan}
        weights = sequence_weights(lengths, c['loss']['normalization'], c['loss'].get('domain_weights'))
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        track = (c['gradient_telemetry'] or c['gradient_surgery'] == 'pcgrad') and len(plan) > 1
        loss_value, domain_stats, domain_grads, previous = 0., {}, {}, None
        for d in plan:
            with phase('teacher_route_' + d):
                teacher = self.teachers.get(d)
            accumulator = DomainAccumulator()
            items = rollouts[d]
            for group in pack_by_tokens([len(x['prompt_ids']) + len(x['response_ids']) for x in items],
                                        c['train_microbatch_tokens']):
                chosen = [items[i] for i in group]
                ids, attention = right_pad([x['prompt_ids'] + x['response_ids'] for x in chosen], self.pad_id, self.device)
                w = torch.tensor([weights[d][i] for i in group], dtype=torch.float32, device=self.device)
                with phase('forward_' + d):
                    loss, detail = microbatch_loss(self.student, teacher, ids, attention,
                                                   [len(x['prompt_ids']) for x in chosen],
                                                   [x['prompt_ids'] for x in chosen], self.support, w, c['loss'])
                with phase('backward_' + d):
                    loss.backward()
                loss_value += float(loss.detach())
                accumulator.add(detail, [len(x['prompt_ids']) for x in chosen], [len(x['response_ids']) for x in chosen])
                del loss, detail, ids, attention
            domain_stats[d] = accumulator.summary()
            if track:
                current = flat_grad(self.params)
                domain_grads[d] = current - previous if previous is not None else current
                previous = current
        cosine, surgery = None, False
        if len(domain_grads) == 2:
            a, b = domain_grads.values()
            cosine = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
            if c['gradient_surgery'] == 'pcgrad':
                combined, surgery = pcgrad(a, b)
                assign_flat_grad(self.params, combined)
        del domain_grads, previous
        with phase('optimizer'):
            norm = torch.nn.utils.clip_grad_norm_(self.params, c['max_grad_norm'], error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
        self.state['step'] = step + 1
        samples = []
        for d in plan:
            exposure = self.state['exposure'][d]
            exposure['prompts'] += len(rows[d])
            exposure['response_tokens'] += sum(lengths[d])
            exposure['eos'] += sum(x['ended_with_eos'] for x in rollouts[d])
            key = c['stratify_keys'].get(d)
            for row, x in zip(rows[d], rollouts[d]):
                samples.append({'sample_id': row['sample_id'], 'domain': d, 'stratum': row.get(key),
                                'prompt_tokens': len(x['prompt_ids']), 'response_tokens': len(x['response_ids']),
                                'ended_with_eos': x['ended_with_eos']})
        for d in plan:
            domain_stats[d].update(prompts=len(rows[d]), response_tokens=sum(lengths[d]),
                                   mean_response_tokens=sum(lengths[d]) / len(lengths[d]),
                                   eos_rate=sum(x['ended_with_eos'] for x in rollouts[d]) / len(rollouts[d]))
        compute = {'forward_', 'backward_', 'rollout_', 'optimizer', 'sync_generation_copy'}
        record = {'step': self.state['step'], 'domains_in_step': plan, 'loss': loss_value,
                  'loss_mode': c['loss']['mode'], 'normalization': c['loss']['normalization'],
                  'gradient_norm': float(norm), 'gradient_cosine_math_code': cosine,
                  'gradient_surgery_applied': surgery, 'learning_rate': self.scheduler.get_last_lr()[0],
                  'merged_lora_layers': merged_layers, 'domains': domain_stats, 'samples': samples,
                  'exposure': json.loads(json.dumps(self.state['exposure'])),
                  'sampler_state': {d: s.state() for d, s in self.samplers.items()},
                  'teacher_swaps': getattr(self.teachers, 'swaps', 0),
                  'total_wall_seconds': time.monotonic() - wall,
                  'pure_compute_seconds': sum(r['seconds'] for r in self.recorder.records
                                              if any(r['phase'].startswith(p) for p in compute)),
                  'phases': self.recorder.records}
        return record, rollouts


def lr_lambda(c):
    def factor(step):
        if c['warmup_steps'] and step < c['warmup_steps']:
            return (step + 1) / c['warmup_steps']
        if c['lr_schedule'] == 'linear':
            return max(0., 1 - step / c['max_steps'])
        return 1.
    return factor


def rng_state():
    import numpy as np
    import torch
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if saved.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def save_checkpoint(path, trainer, identity, telemetry):
    import torch
    from src.training.runtime_v3 import seal_checkpoint
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    trainer.student.save_pretrained(path / 'student', safe_serialization=True)
    torch.save({'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
                'state': trainer.state, 'samplers': {d: s.state() for d, s in trainer.samplers.items()},
                **rng_state()}, path / 'engine.pt')
    write(path / 'run_identity.json', identity)
    write(path / 'telemetry.json', telemetry)
    seal_checkpoint(path)


def restore_engine(trainer, saved):
    trainer.optimizer.load_state_dict(saved['optimizer'])
    trainer.scheduler.load_state_dict(saved['scheduler'])
    trainer.state = saved['state']
    for d, state in saved['samplers'].items():
        trainer.samplers[d].load_state(state)
    restore_rng(saved)


def load_model(spec, device):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(spec['model'], revision=spec['revision'], torch_dtype=torch.bfloat16,
                                                attn_implementation=spec.get('attn_implementation', 'sdpa')).to(device)


def frozen_load(spec, device):
    """Load a teacher without perturbing Python/NumPy/Torch/CUDA RNG streams."""
    import numpy as np
    import torch
    py, npy = random.getstate(), np.random.get_state()
    try:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            return load_model(spec, device)
    finally:
        random.setstate(py)
        np.random.set_state(npy)


def save_rollout_texts(out, step, rollouts, decode, per_domain):
    folder = Path(out) / 'rollouts'
    folder.mkdir(exist_ok=True)
    with (folder / f'step-{step:06d}.jsonl').open('w', encoding='utf-8') as f:
        for d, items in rollouts.items():
            for x in items[:per_domain]:
                f.write(json.dumps({'domain': d, 'ended_with_eos': x['ended_with_eos'],
                                    'response_tokens': len(x['response_ids']),
                                    'prompt': decode(x['prompt_ids']), 'response': decode(x['response_ids'])},
                                   ensure_ascii=False) + '\n')


def run(config_path, output_root=None):
    import torch
    from transformers import AutoTokenizer, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    from src.training.rollout_v3 import GenerationCopy, HFRolloutEngine
    from src.training.runtime_v3 import build_identity, prepare_output, validate_checkpoint
    from src.training.sampler_v3 import StratifiedSampler
    from src.training.sampling_v2 import generation_config, resolved_contract
    from src.training.teachers_v3 import TeacherPool

    c = load_config(config_path)
    output_root = output_root or os.environ.get('OPD_OUTPUT_ROOT')
    if output_root:
        c['output_dir'] = (Path(output_root) / Path(c['output_dir']).name).as_posix()
    out = Path(c['output_dir'])
    pools = load_pools(c)
    support = audited_support(c)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required; CPU tests use tiny models only')
    device = torch.device('cuda')
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['tokenizer_path'], local_files_only=True)
    contract = load_contract()
    prefixes = render_prefixes(pools, tok, contract)
    recorder = PhaseRecorder()
    with recorder.phase('student_load'):
        base = load_model(c['student'], device)
    configs = {d: generation_config(tok.eos_token_id, tok.pad_token_id, n) for d, n in c['max_new_tokens'].items()}
    gen_copy = GenerationCopy(base) if c['rollout_engine'] == 'hf_generation_copy' else None
    probe = gen_copy.model if gen_copy else base
    contracts = {d: resolved_contract(probe, cfg, support, device) for d, cfg in configs.items()}
    identity = build_identity(c, contracts, ROOT)
    saved = None
    if c.get('resume_from_checkpoint'):
        validate_checkpoint(c['resume_from_checkpoint'], identity)  # all files verified BEFORE torch.load
        saved = torch.load(Path(c['resume_from_checkpoint']) / 'engine.pt', map_location='cpu', weights_only=False)
    prepare_output(out, identity, saved['state']['step'] if saved else None, c['resume_policy'])
    write(out / 'resolved_generation.json', contracts)
    budget = {d: {'rows': len(rows), 'max_prompt_tokens': max(len(prefixes[r['sample_id']]) for r in rows)}
              for d, rows in pools.items()}
    write(out / ('startup_' + str(time.time_ns()) + '.json'), {'phases': recorder.records, 'prompt_budget': budget})
    if saved:
        student = PeftModel.from_pretrained(base, Path(c['resume_from_checkpoint']) / 'student', is_trainable=True)
    else:
        lora = c['lora']
        student = get_peft_model(base, LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                                                  target_modules=lora['target_modules'], task_type='CAUSAL_LM'))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.enable_input_require_grads()
    student.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=c['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(c))
    with recorder.phase('teacher_load'):
        teachers = TeacherPool(c['teachers'], lambda spec: frozen_load(spec, 'cpu' if c['teacher_placement'] == 'cpu_swap' else device),
                               device, c['teacher_placement'])
    engine = HFRolloutEngine(gen_copy.model if gen_copy else student, configs, contracts, support, tok.pad_token_id,
                             tok.eos_token_id, c['rollout_batch_size'], sync=gen_copy.sync_from if gen_copy else None)
    samplers = {d: StratifiedSampler(rows, d, c['seed'], c['stratify_keys'].get(d)) for d, rows in pools.items()}
    trainer = OPDTrainerV3(c, student, teachers, engine, samplers, prefixes, support, tok.pad_token_id, device,
                           optimizer, scheduler, PhaseRecorder())
    if saved:
        restore_engine(trainer, saved)
    write(out / 'teacher_pool.json', {'placement': c['teacher_placement'], 'keys': [list(k) for k in teachers.keys()],
                                      'load_seconds': teachers.load_seconds})
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
    with (out / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        while trainer.state['step'] < c['max_steps']:
            record, rollouts = trainer.train_step()
            step = trainer.state['step']
            if c['save_rollouts_every'] and step % c['save_rollouts_every'] == 0:
                save_rollout_texts(out, step, rollouts, decode, c['rollouts_saved_per_domain'])
            if step % c['save_steps'] == 0 or step == c['max_steps']:
                save_checkpoint(out / f'checkpoint-{step}', trainer, identity, record['phases'])
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps({k: record[k] for k in ('step', 'loss', 'gradient_norm', 'gradient_cosine_math_code',
                                                     'total_wall_seconds')}), flush=True)
    return trainer.state


def cli():
    parser = argparse.ArgumentParser(description='opd-v3 MOPD-style dual-teacher OPD (single GPU)')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', help='e.g. /content/drive/MyDrive/project2/runs')
    args = parser.parse_args()
    run(args.config, args.output_root)
