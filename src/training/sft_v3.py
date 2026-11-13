"""sft-v3: LoRA SFT on teacher-generated responses.

Two uses, one code path:
* the SeqKD-Mix / "Off-Policy Finetune" baseline of the ablation ladder;
* the off-policy warm start before OPD. It moves the student toward the
  external teachers' distribution first, the strong-to-weak recipe of Qwen3 and
  Thinking Machines; MOPD reports that non-same-origin teachers destabilise
  on-policy distillation. The resulting adapter feeds ``init_adapter`` in opd-v3.

Teacher outputs come from scripts/generate_eval_v3.py (``train_math`` /
``train_code`` suites with ``--save-token-ids``). They are rendered with the
student prompt contract, so prompt IDs are re-rendered here and must be
identical. The token budget, the support check and the LoRA config match opd-v3.
"""
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from src.teachers.opd_contract import load_contract
from src.training.opd_v3 import (LORA_TARGETS, audited_support, load_model, load_pools, lr_lambda, pack_by_tokens,
                                 render_prefixes, resolve_run_path, right_pad)
from src.training.runtime_v2 import PhaseRecorder, read, sha, write

SFT_DEFAULTS = {'filters': {'require_eos': True, 'math_correct_only': False},
                'examples_per_step': 64, 'epochs': 1, 'train_microbatch_tokens': 16384, 'row_chunk': 64,
                'lora': {'r': 16, 'alpha': 32, 'dropout': 0.0, 'target_modules': LORA_TARGETS},
                'learning_rate': 1e-4, 'warmup_steps': 5, 'lr_schedule': 'linear', 'max_grad_norm': 1.0,
                'save_steps': 50}
SFT_SOURCES = ['src/training/sft_v3.py', 'src/losses/sft_v3.py', 'src/losses/mopd_v3.py', 'src/training/opd_v3.py',
               'src/teachers/opd_contract.py', 'src/data/approved_training_inputs.py']


def load_config(path):
    c = read(path)
    if c.get('implementation_version') != 'sft-v3':
        raise ValueError('Use a versioned sft-v3 configuration')
    for key, value in SFT_DEFAULTS.items():
        c[key] = {**value, **c.get(key, {})} if isinstance(value, dict) else c.get(key, value)
    if set(c['teacher_outputs']) != set(c['data_files']):
        raise ValueError('teacher_outputs must cover exactly the data_files domains')
    return c


def load_examples(c, pools, prefixes, eos_id, support, output_root=None):
    """Join teacher outputs to approved rows; returns (examples, statistics)."""
    examples, stats = [], Counter()
    for domain, rows in pools.items():
        path = resolve_run_path(c['teacher_outputs'][domain], output_root)
        records = {}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                records[record['id']] = record
        for row in rows:
            record = records.get(row['sample_id'])
            if record is None:
                stats[domain + '_missing'] += 1
                continue
            if 'response_ids' not in record:
                raise ValueError(f'{path}: regenerate with --save-token-ids')
            if c['filters']['require_eos'] and record['finish_reason'] != 'stop':
                stats[domain + '_truncated'] += 1
                continue
            if (domain == 'math' and c['filters']['math_correct_only']
                    and not (record.get('grade') or {}).get('math_verify', {}).get('correct')):
                stats[domain + '_incorrect'] += 1
                continue
            response = list(record['response_ids'])
            if record['finish_reason'] == 'stop' and (not response or response[-1] != eos_id):
                response.append(eos_id)  # vLLM may omit the stop token from token_ids
            if not response or any(not 0 <= t < support for t in response):
                stats[domain + '_invalid_tokens'] += 1
                continue
            if record.get('prompt_tokens') not in (None, len(prefixes[row['sample_id']])):
                raise ValueError('Teacher outputs were rendered with a different prompt contract')
            examples.append({'sample_id': row['sample_id'], 'domain': domain,
                             'prompt_ids': prefixes[row['sample_id']], 'response_ids': response})
            stats[domain + '_kept'] += 1
    if not examples:
        raise ValueError('No usable teacher outputs')
    return examples, dict(stats)


class SFTTrainerV3:
    def __init__(self, c, student, examples, support, pad_id, device, optimizer, scheduler, recorder=None):
        self.c, self.student, self.examples = c, student, examples
        self.support, self.pad_id, self.device = support, pad_id, device
        self.optimizer, self.scheduler = optimizer, scheduler
        self.recorder = recorder or PhaseRecorder()
        self.params = [p for p in student.parameters() if p.requires_grad]
        self.step = 0
        self.order = []
        for epoch in range(c['epochs']):
            indices = list(range(len(examples)))
            random.Random(f'sft-v3/{c["seed"]}/{epoch}').shuffle(indices)
            self.order.extend(indices)
        self.total_steps = -(-len(self.order) // c['examples_per_step'])

    def train_step(self):
        import torch
        from src.losses.mopd_v3 import hidden_rows, prediction_rows, support_logits
        from src.losses.sft_v3 import row_cross_entropy
        from src.teachers.opd_contract import teacher_batch_from_student
        c, phase = self.c, self.recorder.phase
        self.recorder.records = []
        wall = time.monotonic()
        start = self.step * c['examples_per_step']
        batch = [self.examples[i] for i in self.order[start:start + c['examples_per_step']]]
        tokens = sum(len(x['response_ids']) for x in batch)
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_value, per_domain = 0., Counter()
        counts = Counter()
        for group in pack_by_tokens([len(x['prompt_ids']) + len(x['response_ids']) for x in batch],
                                    c['train_microbatch_tokens']):
            chosen = [batch[i] for i in group]
            ids, attention = right_pad([x['prompt_ids'] + x['response_ids'] for x in chosen], self.pad_id, self.device)
            _, response = teacher_batch_from_student(ids, attention, [len(x['prompt_ids']) for x in chosen],
                                                     [x['prompt_ids'] for x in chosen], self.support)
            b, positions = prediction_rows(response)
            with phase('forward'):
                hidden, head = hidden_rows(self.student, ids, attention, b, positions)
                logits = support_logits(hidden, head, self.support)
                weights = torch.full((len(b),), 1. / tokens, dtype=torch.float32, device=logits.device)
                loss, nll = row_cross_entropy(logits, ids[b, positions + 1], weights, c['row_chunk'])
            with phase('backward'):
                loss.backward()
            loss_value += float(loss.detach())
            domains = [chosen[i]['domain'] for i in b.tolist()]
            for d, value in zip(domains, nll.tolist()):
                per_domain[d] += value
                counts[d] += 1
            del loss, logits, hidden, ids, attention
        with phase('optimizer'):
            norm = torch.nn.utils.clip_grad_norm_(self.params, c['max_grad_norm'], error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
        self.step += 1
        return {'step': self.step, 'loss_token_mean_nll': loss_value, 'response_tokens': tokens,
                'examples': len(batch), 'domain_nll_token_mean': {d: per_domain[d] / counts[d] for d in counts},
                'gradient_norm': float(norm), 'learning_rate': self.scheduler.get_last_lr()[0],
                'total_wall_seconds': time.monotonic() - wall, 'phases': self.recorder.records}


def run(config_path, output_root=None, teacher_outputs=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, set_seed
    c = load_config(config_path)
    if teacher_outputs:
        c['teacher_outputs'].update(teacher_outputs)
    output_root = output_root or os.environ.get('OPD_OUTPUT_ROOT')
    out = resolve_run_path(c['output_dir'], output_root)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; use a new output_dir')
    out.mkdir(parents=True, exist_ok=True)
    pools = load_pools(c)
    support = audited_support({**c, 'teachers': {}})
    if not torch.cuda.is_available():
        raise RuntimeError('sft-v3 training needs a CUDA GPU; CPU tests use tiny models')
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['tokenizer_path'], local_files_only=True)
    prefixes = render_prefixes(pools, tok, load_contract())
    examples, stats = load_examples(c, pools, prefixes, tok.eos_token_id, support, output_root)
    base = load_model(c['student'], 'cuda')
    lora = c['lora']
    student = get_peft_model(base, LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                                              target_modules=lora['target_modules'], task_type='CAUSAL_LM'))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.enable_input_require_grads()
    student.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=c['learning_rate'])
    trainer = SFTTrainerV3(c, student, examples, support, tok.pad_token_id, 'cuda', optimizer, None)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda({**c, 'max_steps': trainer.total_steps}))
    write(out / 'sft_identity.json', {'schema': 'sft-v3', 'config': c, 'examples': stats,
                                      'manifest_sha': sha(c['data_manifest']),
                                      'teacher_outputs_sha': {d: sha(resolve_run_path(p, output_root))
                                                              for d, p in c['teacher_outputs'].items()},
                                      'source_files': {s: sha(s) for s in SFT_SOURCES}})
    with (out / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        while trainer.step < trainer.total_steps:
            record = trainer.train_step()
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps({k: record[k] for k in ('step', 'loss_token_mean_nll', 'gradient_norm')}), flush=True)
            if trainer.step % c['save_steps'] == 0:
                student.save_pretrained(out / f'adapter-step-{trainer.step}', safe_serialization=True)
    student.save_pretrained(out / 'final_adapter', safe_serialization=True)
    return trainer.step


def cli():
    parser = argparse.ArgumentParser(description='sft-v3: LoRA SFT on teacher outputs (SeqKD baseline / OPD warm start)')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--teacher-outputs', nargs='*', default=[], help='domain=path overrides, e.g. math=/drive/.../train_math.jsonl')
    args = parser.parse_args()
    run(args.config, args.output_root, dict(item.split('=', 1) for item in args.teacher_outputs))
