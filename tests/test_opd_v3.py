"""CPU contract tests for opd-v3 (tiny Qwen2 models; no real Qwen, no GPU claims)."""
import json
import math
from collections import Counter

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.losses.mopd_v3 import (STAT, hidden_rows, microbatch_loss, prediction_rows, row_divergence,
                                support_logits)
from src.losses.reference import aligned_kl
from src.teachers.opd_contract import teacher_batch_from_student
from src.training.opd_v3 import (OPDTrainerV3, lr_lambda, pack_by_tokens, pcgrad, restore_engine, right_pad,
                                 save_checkpoint, sequence_weights, step_plan, validate_config, with_defaults)
from src.teachers.opd_contract import load_contract
from src.training.rollout_v3 import GenerationCopy, HFRolloutEngine, split_generated
from src.training.runtime_v3 import SOURCES, build_identity, check_identity, validate_checkpoint
from src.training.sampler_v3 import StratifiedSampler
from src.training.sampling_v2 import generation_config, resolved_contract
from src.training.teachers_v3 import TeacherPool

SUPPORT, EOS, PAD = 32, 2, 0
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def tiny_qwen(vocab=40, tied=True, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         tie_word_embeddings=tied, rope_theta=10000.0, use_sliding_window=False,
                         # Production uses sdpa. HF eager + float64 + left padding yields NaN rows.
                         attn_implementation='sdpa')
    return Qwen2ForCausalLM(config).to(dtype).eval()


def peft_student(base, seed=0):
    from peft import LoraConfig, get_peft_model
    student = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                              task_type='CAUSAL_LM'))
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in student.named_parameters():
            if 'lora_' in name:
                p.normal_(0, .2)
    return student


# ---------------------------------------------------------------- losses

@pytest.mark.parametrize('chunk', [1, 3, 64])
def test_exact_rows_match_repository_reference(chunk):
    torch.manual_seed(1)
    s = torch.randn(7, 19, dtype=torch.double, requires_grad=True)
    t = torch.randn(7, 19, dtype=torch.double)
    y = torch.randint(0, 19, (7,))
    loss, stats = row_divergence(s, t, y, torch.full((7,), 1 / 7, dtype=torch.double), 'exact_rkl', chunk=chunk)
    pad = torch.zeros(1, 19, dtype=torch.double)
    reference = aligned_kl(torch.cat([s, pad])[None], torch.cat([t, pad])[None],
                           torch.ones(1, 8, dtype=torch.long), range(19), range(19))
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    expected = torch.autograd.grad(reference, s, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(s.grad, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(stats[:, STAT['kl_exact']].double().mean(), reference.detach(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode,alpha', [('exact_rkl', 1.), ('exact_rkl', .6), ('topk_rkl', 1.), ('topk_rkl', .6),
                                        ('pg', 1.), ('pg', .6)])
def test_all_modes_match_explicit_autograd_with_detached_targets(mode, alpha):
    torch.manual_seed(2)
    s = torch.randn(6, 11, dtype=torch.double, requires_grad=True)
    t = torch.randn(6, 11, dtype=torch.double)
    y = torch.randint(0, 11, (6,))
    w = torch.rand(6, dtype=torch.double)
    loss, _ = row_divergence(s, t, y, w, mode, topk=4, clip=.5, alpha=alpha, chunk=2)
    log_p, log_q = s.log_softmax(-1), t.log_softmax(-1)
    target = log_q if alpha == 1 else torch.logaddexp(log_q + math.log(alpha), log_p.detach() + math.log1p(-alpha))
    target = target.detach()
    if mode == 'exact_rkl':
        value = (log_p.exp() * (log_p - target)).sum(-1)
    elif mode == 'topk_rkl':
        index = log_q.topk(4, -1).indices
        lp, lt = log_p.gather(-1, index), target.gather(-1, index)
        value = (lp.exp() * (lp - lt) - lp.exp() + lt.exp()).sum(-1)
    else:
        lp_y = log_p.gather(-1, y[:, None]).squeeze(-1)
        advantage = (target.gather(-1, y[:, None]).squeeze(-1) - lp_y).clamp(-.5, .5).detach()
        value = -advantage * lp_y
    reference = (value * w).sum()
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    expected = torch.autograd.grad(reference, s)[0]
    loss.backward()
    torch.testing.assert_close(s.grad, expected, rtol=1e-10, atol=1e-12)


def test_divergences_vanish_at_teacher_and_gradcheck():
    torch.manual_seed(3)
    t = torch.randn(4, 9, dtype=torch.double)
    y = torch.randint(0, 9, (4,))
    w = torch.ones(4, dtype=torch.double)
    for mode in ['exact_rkl', 'topk_rkl', 'pg']:
        s = t.clone().requires_grad_(True)
        loss, _ = row_divergence(s, t, y, w, mode, topk=3)
        loss.backward()
        if mode != 'pg':
            assert abs(float(loss.detach())) < 1e-12
        assert s.grad.abs().max() < 1e-12  # zero advantage => zero PG gradient
        if mode != 'pg':
            # PG and alpha < 1 use stop-gradient targets by design, so only the
            # pure divergences are true gradients of their forward value.
            s = torch.randn(4, 9, dtype=torch.double, requires_grad=True)
            assert torch.autograd.gradcheck(lambda x: row_divergence(x, t, y, w, mode, topk=3, chunk=3)[0], (s,))


def test_invalid_options_fail_closed():
    s, t = torch.randn(3, 5), torch.randn(3, 5)
    y, w = torch.zeros(3, dtype=torch.long), torch.ones(3)
    with pytest.raises(ValueError):
        row_divergence(s, t, y, w, 'unknown')
    with pytest.raises(ValueError):
        row_divergence(s, t, y, w, 'topk_rkl', topk=9)
    with pytest.raises(ValueError):
        row_divergence(s, t, y, w, 'exact_rkl', alpha=0.)
    with pytest.raises(ValueError, match='Nonfinite'):
        row_divergence(torch.full((3, 5), float('nan')), t, y, w)


# ------------------------------------------------- response rows / batching

def test_response_row_logits_match_single_sequence_full_logits():
    model = tiny_qwen()
    sequences, prompts = [[1, 3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14]], [2, 1, 3]
    ids, attention = right_pad(sequences, PAD, 'cpu')
    _, response = teacher_batch_from_student(ids, attention, prompts, [s[:n] for s, n in zip(sequences, prompts)], SUPPORT)
    b, positions = prediction_rows(response)
    assert len(b) == sum(len(s) - n for s, n in zip(sequences, prompts))
    hidden, head = hidden_rows(model, ids, attention, b, positions)
    rows = support_logits(hidden, head, SUPPORT)
    with torch.no_grad():
        for k, (i, p) in enumerate(zip(b.tolist(), positions.tolist())):
            single = model(torch.tensor([sequences[i]])).logits[0, p, :SUPPORT]
            torch.testing.assert_close(rows[k], single, rtol=1e-10, atol=1e-10)


def test_generation_copy_merges_lora_out_of_place_and_idempotently():
    base = tiny_qwen()
    gen = GenerationCopy(base)
    student = peft_student(base)
    frozen = {n: p.detach().clone() for n, p in student.get_base_model().named_parameters() if 'lora_' not in n}
    x = torch.tensor([[1, 4, 6, 9, 3]])
    for _ in range(3):
        assert gen.sync_from(student) == 2 * 7
        with torch.no_grad():
            torch.testing.assert_close(gen.model(x).logits, student(x).logits, rtol=1e-6, atol=1e-6)
    for n, p in student.get_base_model().named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n


def test_split_generated_keeps_first_eos_and_unfinished_rows():
    sequences = torch.tensor([[0, 0, 5, 6, 7, EOS, PAD, PAD], [0, 5, 6, 7, 8, 8, 8, 8], [1, 1, 1, 1, EOS, PAD, PAD, PAD]])
    assert split_generated(sequences, 4, EOS) == [([7, EOS], True), ([8, 8, 8, 8], False), ([EOS], True)]


def engine_for(model, max_new=4, batch=2):
    config = generation_config(EOS, PAD, max_new)
    contract = resolved_contract(model, config, SUPPORT, torch.device('cpu'))
    return HFRolloutEngine(model, {'math': config, 'code': config}, {'math': contract, 'code': contract}, SUPPORT,
                           PAD, EOS, batch)


def test_batched_rollouts_keep_order_prefix_and_support():
    torch.manual_seed(5)
    engine = engine_for(tiny_qwen())
    prefixes = [[1, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12]]
    out = engine.generate(prefixes, 'math')
    assert [o['prompt_ids'] for o in out] == prefixes
    for o in out:
        assert 1 <= len(o['response_ids']) <= 4 and all(0 <= v < SUPPORT for v in o['response_ids'])
        assert o['ended_with_eos'] == (o['response_ids'][-1] == EOS)
        assert EOS not in o['response_ids'][:-1]


# ------------------------------------------------------------ scheduling

def rows_for(domain, n, key, values):
    return [{'sample_id': f'{domain}/{i:03d}', 'domain': domain, key: values[i % len(values)]} for i in range(n)]


def test_stratified_sampler_mixes_strata_is_deterministic_and_resumes():
    rows = rows_for('math', 60, 'source', ['MATH']) + [{'sample_id': f'g/{i}', 'domain': 'math', 'source': 'GSM8K'} for i in range(40)]
    a, b = StratifiedSampler(rows, 'math', 42, 'source'), StratifiedSampler(rows, 'math', 42, 'source')
    first = a.next(10)
    assert [r['sample_id'] for r in first] == [r['sample_id'] for r in b.next(10)]
    assert abs(Counter(r['source'] for r in first)['GSM8K'] - 4) <= 1
    epoch0 = a.epoch_ids()
    a.next(95)
    assert a.epoch == 1 and a.epoch_ids() != epoch0 and sorted(a.epoch_ids()) == sorted(epoch0)
    state = a.state()
    expected = [r['sample_id'] for r in a.next(7)]
    c = StratifiedSampler(rows, 'math', 42, 'source')
    c.load_state(state)
    assert [r['sample_id'] for r in c.next(7)] == expected
    with pytest.raises(ValueError):
        StratifiedSampler(rows[:-1], 'math', 42, 'source').load_state(state)


def test_sequence_weights_sum_to_one_and_follow_definitions():
    lengths = {'math': [2, 6], 'code': [4]}
    for mode in ['token_mean', 'sequence_mean', 'domain_token_mean']:
        w = sequence_weights(lengths, mode)
        assert sum(x * n for d in lengths for x, n in zip(w[d], lengths[d])) == pytest.approx(1)
    assert sequence_weights(lengths, 'sequence_mean')['math'] == pytest.approx([1 / 6, 1 / 18])
    w = sequence_weights(lengths, 'domain_token_mean', {'math': 3, 'code': 1})
    assert sum(x * n for x, n in zip(w['math'], lengths['math'])) == pytest.approx(.75)
    with pytest.raises(ValueError):
        sequence_weights({'math': [0]}, 'token_mean')


def test_pack_by_tokens_bounds_padded_size_and_covers_all():
    lengths = [5, 9, 3, 9, 1, 7, 20]
    groups = pack_by_tokens(lengths, 18)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        assert len(g) == 1 or len(g) * max(lengths[i] for i in g) <= 18


def test_step_plan_mixed_and_blocked():
    c = {'prompts_per_step': {'math': 3, 'code': 2}, 'schedule': {'type': 'mixed'}}
    assert step_plan(c, 7) == {'math': 3, 'code': 2}
    c['schedule'] = {'type': 'blocked', 'block_steps': 2}
    assert [step_plan(c, s) for s in range(5)] == [{'math': 5}, {'math': 5}, {'code': 5}, {'code': 5}, {'math': 5}]


def test_pcgrad_removes_conflict_only():
    a, b = torch.tensor([1., 0.]), torch.tensor([-1., 1.])
    combined, applied = pcgrad(a, b)
    assert applied and float(combined @ a) >= -1e-9 and float(combined @ b) >= -1e-9
    combined, applied = pcgrad(a, torch.tensor([1., 1.]))
    assert not applied and torch.equal(combined, torch.tensor([2., 1.]))


def test_teacher_pool_dedups_same_identity_and_counts_swaps():
    loads = []
    def loader(spec):
        loads.append(spec['model'])
        return tiny_qwen()
    same = {'model': 'general', 'revision': 'r'}
    pool = TeacherPool({'math': same, 'code': dict(same)}, loader, 'cpu')
    assert loads == ['general'] and pool.get('math') is pool.get('code')
    pool = TeacherPool({'math': {'model': 'm', 'revision': 'r'}, 'code': {'model': 'c', 'revision': 'r'}}, loader, 'cpu', 'cpu_swap')
    pool.get('math'), pool.get('code'), pool.get('code'), pool.get('math')
    assert pool.swaps == 3 and pool.get('math') is not pool.get('code')


# -------------------------------------------------------------- identity

def fake_root(tmp_path):
    for name in SOURCES + ['configs/project2_prompt_contract.json', 'data/manifest.json', 'reports/audit.json', 'data/tok/tokenizer.json']:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}', encoding='utf-8')
    return {'data_manifest': 'data/manifest.json', 'tokenizer_audit': 'reports/audit.json', 'tokenizer_path': 'data/tok', 'seed': 1}


def test_identity_soft_facts_warn_but_hard_fields_block(tmp_path):
    config = fake_root(tmp_path)
    saved = build_identity(config, {}, tmp_path)
    environment_changed = json.loads(json.dumps(saved))
    environment_changed['soft']['platform'] = 'another-colab-image'
    environment_changed['soft']['dependencies']['somepackage'] = '9.9'
    assert check_identity(saved, environment_changed) == ['dependencies', 'platform']
    (tmp_path / 'src/losses/mopd_v3.py').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match=r'hard fields\): source_files'):
        check_identity(saved, build_identity(config, {}, tmp_path))
    with pytest.raises(ValueError, match='config'):
        check_identity(saved, build_identity({**config, 'seed': 2}, {}, tmp_path))


# ------------------------------------------------------------ end to end

def trainer_config(**overrides):
    c = {'implementation_version': 'opd-v3', 'prompts_per_step': {'math': 3, 'code': 2},
         'data_files': {'math': 'm', 'code': 'c'}, 'teachers': {'math': {}, 'code': {}},
         'max_new_tokens': {'math': 4, 'code': 5}, 'max_steps': 4, 'rollout_batch_size': 2,
         'train_microbatch_tokens': 16, 'gradient_surgery': 'pcgrad', 'learning_rate': 5e-2}
    c.update(overrides)
    return validate_config(with_defaults(c), load_contract())


def build_trainer(c, base_state, seed=0):
    base = tiny_qwen(seed=99)
    base.load_state_dict(base_state)
    gen = GenerationCopy(base)
    student = peft_student(base, seed)
    teachers = TeacherPool({'math': {'model': 'math-teacher', 'revision': 'r'}, 'code': {'model': 'code-teacher', 'revision': 'r'}},
                           lambda spec: tiny_qwen(vocab=48, tied=False, seed=7 if spec['model'] == 'math-teacher' else 8), 'cpu')
    engine = engine_for(gen.model, max_new=5)
    engine.sync = gen.sync_from
    pools = {'math': rows_for('math', 7, 'source', ['MATH', 'GSM8K']), 'code': rows_for('code', 5, 'interface', ['function', 'stdin'])}
    prefixes = {r['sample_id']: [1 + (i % 9), 10 + (i % 7), 20 + (i % 5)][: 1 + i % 3]
                for d in pools for i, r in enumerate(pools[d])}
    samplers = {d: StratifiedSampler(rows, d, 42, c['stratify_keys'][d]) for d, rows in pools.items()}
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(c))
    return OPDTrainerV3(c, student, teachers, engine, samplers, prefixes, SUPPORT, PAD, 'cpu', optimizer, scheduler)


def test_mixed_step_routes_teachers_updates_only_lora_and_reports_telemetry():
    torch.manual_seed(11)
    c = trainer_config()
    trainer = build_trainer(c, tiny_qwen(seed=99).state_dict())
    lora_before = {n: p.detach().clone() for n, p in trainer.student.named_parameters() if p.requires_grad}
    base_before = {n: p.detach().clone() for n, p in trainer.student.named_parameters() if not p.requires_grad}
    record, rollouts = trainer.train_step()
    assert record['domains_in_step'] == {'math': 3, 'code': 2} and set(rollouts) == {'math', 'code'}
    assert record['gradient_cosine_math_code'] is not None and -1 <= record['gradient_cosine_math_code'] <= 1
    assert sum(1 for s in record['samples'] if s['domain'] == 'math') == 3
    for d in ('math', 'code'):
        stats = record['domains'][d]
        assert stats['response_rows'] == stats['response_tokens'] > 0 and stats['kl_exact_token_mean'] > 0
        assert len(stats['kl_by_relative_position_decile']) == 10 and stats['removed_mass_teacher']['max'] > 0
    assert any(not torch.equal(p, lora_before[n]) for n, p in trainer.student.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, base_before[n]) for n, p in trainer.student.named_parameters() if not p.requires_grad)
    for teacher in trainer.teachers.models.values():
        assert all(p.grad is None for p in teacher.parameters())
    blocked = build_trainer(trainer_config(schedule={'type': 'blocked', 'block_steps': 1}, gradient_surgery='none'),
                            tiny_qwen(seed=99).state_dict())
    first, _ = blocked.train_step()
    second, _ = blocked.train_step()
    assert first['domains_in_step'] == {'math': 5} and second['domains_in_step'] == {'code': 5}
    assert first['gradient_cosine_math_code'] is None


@pytest.mark.parametrize('mode', ['topk_rkl', 'pg'])
def test_other_mopd_modes_run_end_to_end(mode):
    torch.manual_seed(12)
    trainer = build_trainer(trainer_config(loss={'mode': mode, 'topk': 5, 'normalization': 'token_mean'}),
                            tiny_qwen(seed=99).state_dict())
    record, _ = trainer.train_step()
    assert record['loss_mode'] == mode and torch.isfinite(torch.tensor(record['loss']))


def test_checkpoint_resume_reproduces_next_step(tmp_path):
    from peft import PeftModel
    torch.manual_seed(13)
    c = trainer_config()
    base_state = tiny_qwen(seed=99).state_dict()
    trainer = build_trainer(c, base_state)
    trainer.train_step()
    identity = {'schema': 'opd-runtime-v3', 'hard': {'test': 1}, 'soft': {}}
    save_checkpoint(tmp_path / 'checkpoint-1', trainer, identity, [])
    expected, _ = trainer.train_step()
    assert validate_checkpoint(tmp_path / 'checkpoint-1', identity) == []
    restored = build_trainer(c, base_state, seed=1)
    base = restored.student.get_base_model()
    restored.student = PeftModel.from_pretrained(tiny_qwen(seed=99), tmp_path / 'checkpoint-1' / 'student', is_trainable=True)
    restored.student.get_base_model().load_state_dict(
        {k: v for k, v in trainer.student.get_base_model().state_dict().items() if 'lora_' not in k}, strict=False)
    restored.params = [p for p in restored.student.parameters() if p.requires_grad]
    restored.optimizer = torch.optim.AdamW(restored.params, lr=c['learning_rate'])
    restored.scheduler = torch.optim.lr_scheduler.LambdaLR(restored.optimizer, lr_lambda(c))
    del base
    restore_engine(restored, torch.load(tmp_path / 'checkpoint-1' / 'engine.pt', weights_only=False))
    actual, _ = restored.train_step()
    assert [s['sample_id'] for s in actual['samples']] == [s['sample_id'] for s in expected['samples']]
    assert [s['response_tokens'] for s in actual['samples']] == [s['response_tokens'] for s in expected['samples']]
    assert actual['loss'] == pytest.approx(expected['loss'], rel=1e-6, abs=1e-9)
    assert actual['step'] == expected['step'] == 2


# ---------------------------------------------------------- experiment protocol

def test_experiment_configs_validate_and_differ_only_in_declared_factors():
    from pathlib import Path
    from src.training.opd_v3 import load_config
    configs = {p.stem.replace('project2_opd_v3_', ''): load_config(p) for p in Path('configs').glob('project2_opd_v3_*.json')}
    reference = configs['d4_dual']
    declared = {'d4_blocked': {'schedule'}, 'd4_pcgrad': {'gradient_surgery'}, 'd4_pg': {'loss'}, 'd4_topk': {'loss'},
                'd3_general': {'teachers'}, 'd4_warmstart': {'init_adapter'},
                'd1_math': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens'},
                'd2_code': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens'}}
    assert set(configs) == set(declared) | {'d4_dual', 'smoke'}
    for name, factors in declared.items():
        differing = {k for k in set(reference) | set(configs[name])
                     if reference.get(k) != configs[name].get(k)} - {'output_dir'}
        assert differing == factors, (name, differing)
    steps = reference['max_steps']
    # Per-domain prompt exposure is identical across D1/D2/D4 and the blocked ablation.
    mixed = {d: n * steps for d, n in reference['prompts_per_step'].items()}
    assert configs['d1_math']['prompts_per_step']['math'] * steps == mixed['math']
    assert configs['d2_code']['prompts_per_step']['code'] * steps == mixed['code']
    blocked = configs['d4_blocked']
    exposure = Counter()
    for step in range(steps):
        exposure.update(step_plan(blocked, step))
    assert dict(exposure) == mixed
    assert len({tuple(sorted(t.items())) for t in configs['d3_general']['teachers'].values()}) == 1
    assert all(c['data_manifest'] == 'data/project2_release_v5_20261108/release/manifest.json' for c in configs.values())
