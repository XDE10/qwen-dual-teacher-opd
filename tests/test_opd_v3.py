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
        index = target.topk(4, -1).indices
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
    with pytest.raises(ValueError, match='response_mask'):
        prediction_rows(torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match='Empty response rows'):
        prediction_rows(torch.zeros((2, 3), dtype=torch.bool))
    assert with_defaults({'loss': None, 'lora': None})['loss']['mode'] == 'exact_rkl'


def test_removed_mass_avoids_catastrophic_cancellation():
    from src.losses.mopd_v3 import removed_mass
    model = tiny_qwen(vocab=40)
    _, head = model.model, model.get_output_embeddings()
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:SUPPORT, 0] = 10.0
        head.weight[SUPPORT:, 0] = -15.0  # tiny tail mass (~1e-11), below float32 1 - exp() precision
    hidden = torch.ones(2, 16, dtype=torch.float64)
    hidden[:, 1:] = 0.0
    s_lse = support_logits(hidden, head, SUPPORT).float().logsumexp(-1)
    mass = removed_mass(hidden, head, SUPPORT, s_lse)
    assert (mass > 0).all() and (mass < 1e-9).all()


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


def test_latent_rollout_recovery_reuses_completed_batches(tmp_path):
    import json

    from src.training.latent_progress import RunProgress

    class Engine:
        batch_size = 2

        def __init__(self, fail_at=None):
            self.calls = 0
            self.fail_at = fail_at

        def generate(self, prefixes, domain):
            self.calls += 1
            if self.calls == self.fail_at:
                raise RuntimeError('interrupted')
            return [{'prompt_ids': prefix, 'domain': domain, 'ended_with_eos': True,
                     'response_ids': [torch.randint(16, 32, ()).item()]} for prefix in prefixes]

    identity = {'hard': {'config': {'seed': 42}}}
    progress = RunProgress(tmp_path, identity, 'cpu')
    prefixes = [[3, 4], [5], [6, 7], [8]]
    with pytest.raises(RuntimeError, match='interrupted'):
        progress.rollouts(Engine(fail_at=2), prefixes, 1, 'math')
    engine = Engine()
    restored = progress.rollouts(engine, prefixes, 1, 'math')
    assert engine.calls == 1
    assert [row['prompt_ids'] for row in restored] == prefixes
    (tmp_path / 'rollout_spool/step-000001/batch-0001.json.gz').write_bytes(b'corrupt')
    assert progress.rollouts(engine, prefixes, 1, 'math') == restored
    assert engine.calls == 2
    events = [json.loads(line) for line in (tmp_path / 'phase_metrics.jsonl').read_text().splitlines()]
    assert any(row['phase'] == 'rollout_batch_done' and row['reused'] for row in events)


def test_latent_top16_scores_and_gradients_match_dense_appendix_c(monkeypatch):
    from src.losses import latent_opd

    monkeypatch.setattr(latent_opd, 'LOGIT_ROWS', 2)
    student, teacher = tiny_qwen(seed=1), tiny_qwen(seed=2)
    ids, attention = right_pad([[3, 4, 5, 6], [7, 8, 9]], PAD, 'cpu')
    response = latent_opd.response_batch(ids, attention, [[3], [7, 8]], SUPPORT)
    batch, rows, positions = response
    student_hidden = student.model(**batch).last_hidden_state[rows, positions]
    teacher_hidden = teacher.model(**batch).last_hidden_state[rows, positions]
    student_log_probs = support_logits(student_hidden, student.lm_head, SUPPORT).float().log_softmax(-1)
    teacher_log_probs = support_logits(teacher_hidden, teacher.lm_head, SUPPORT).float().log_softmax(-1)
    selected, expected_ids = student_log_probs.detach().topk(16, dim=-1)
    expected_rewards = selected.softmax(-1) * (teacher_log_probs.detach().gather(-1, expected_ids) - selected)
    expected_advantages = expected_rewards / expected_rewards.std(correction=1).clamp_min(1e-6)
    expected_loss = -(expected_advantages * student_log_probs.gather(-1, expected_ids)).sum(-1).mean()

    candidates, rewards = latent_opd.score_candidates(student, teacher, response, SUPPORT)
    assert torch.equal(candidates, expected_ids) and not rewards.requires_grad
    torch.testing.assert_close(rewards, expected_rewards, atol=1e-7, rtol=1e-5)
    scales = latent_opd.reward_scale([(candidates[:2], rewards[:2]), (candidates[2:], rewards[2:])])
    actual_loss = latent_opd.token_objective(student_hidden, student.lm_head, SUPPORT, candidates, rewards / scales)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_grad, = torch.autograd.grad(actual_loss, student_hidden, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected_loss, student_hidden)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize(('variant', 'start_step', 'weights'), [
    ('token', 0, [(1.0, 0.0), (1.0, 0.0)]),
    ('latent', 0, [(0.0, 1.0), (0.0, 1.0)]),
    ('latent', 10, [(0.5, 0.5), (0.5, 0.5)]),
    ('math20', 20, [(1.0, 0.5), (1.0, 0.0)]),
    ('math20', 40, [(1.0, 0.0), (1.0, 0.0)]),
])
def test_latent_opd_updates_only_student_with_per_teacher_weights(variant, start_step, weights):
    from src.training.latent_opd import train_step

    trainer = build_trainer(trainer_config(), tiny_qwen(seed=99).state_dict())
    trainer.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainer.student.enable_input_require_grads()
    trainer.state['step'] = start_step
    for domain, expected in zip(('math', 'code'), weights):
        before = [param.detach().clone() for param in trainer.params]
        record = train_step(trainer, variant, prompts=1, responses=2, microbatch_tokens=8)
        assert record['domain'] == domain and record['responses'] == 2
        assert (record['token_weight'], record['latent_weight']) == expected
        assert math.isfinite(record['loss']) and record['gradient_norm'] > 0
        assert any(not torch.equal(param, old) for param, old in zip(trainer.params, before))
        assert all(param.grad is None for teacher in trainer.teachers.models.values()
                   for param in teacher.parameters())


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
                'd4_dual_lr1e-5': {'learning_rate'}, 'd4_dual_lr1e-4': {'learning_rate'}, 'd4_prox07': {'loss'},
                'd1_math': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens'},
                'd2_code': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens'},
                'd3_general_lr1e-4': {'teachers', 'learning_rate'},
                'd1_math_lr1e-4': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens', 'learning_rate'},
                'd2_code_lr1e-4': {'data_files', 'teachers', 'prompts_per_step', 'max_new_tokens', 'learning_rate'},
                'd4_blocked_lr1e-4': {'schedule', 'learning_rate'},
                'd4_warmstart_lr1e-4': {'init_adapter', 'learning_rate'},
                'd4_prox07_lr1e-4': {'loss', 'learning_rate'},
                'd4_aligned_lr1e-4': {'teachers', 'learning_rate'},
                'd4_math14b_lr1e-4': {'teachers', 'tokenizer_audit', 'learning_rate'}}
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


def test_latent_mopd_three_schedule_and_worker_boundaries():
    from src.losses.latent_opd import loss_weights
    from src.training.latent_mopd_three import DOMAINS, build_schedule, microbatches

    pools = {d: list(range(672)) for d in DOMAINS}
    schedule = build_schedule(pools)
    assert [d for d, _ in schedule[:6]] == ['math', 'code', 'logic'] * 2
    assert [sum(domain == d for domain, _ in schedule) for d in DOMAINS] == [21, 21, 20]
    assert schedule[3][1] == list(range(32, 64))
    weights = [loss_weights('latent', d, i // 3) for i, (d, _) in enumerate(schedule)]
    assert weights[:3] == [(0, 1)] * 3
    assert weights[3:6] == [(.1, .9)] * 3
    assert weights[30:] == [(1, 0)] * 32
    groups = list(microbatches([100, 6000, 18000, 800] * 32))
    assert sorted(i for group in groups for i in group) == list(range(128))
    assert all(group[0] // 16 == i // 16 for group in groups for i in group)


def test_latent_mopd_three_qwen2_final_norm_and_one_update():
    from src.losses.latent_opd import response_batch
    from src.training.latent_mopd_three import state_rows, update

    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=48, hidden_size=24, intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, pad_token_id=0)
    student = Qwen2ForCausalLM(config)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    teacher = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    prefixes = [[1, 2], [3, 4]]
    rollouts = [{'prompt_ids': p, 'response_ids': r} for p, r in zip(prefixes, [[5, 6], [7]])]
    ids, mask = right_pad([[1, 2, 5, 6], [3, 4, 7]], 0, 'cpu')
    batch, rows, positions = response_batch(ids, mask, prefixes, 43)
    with torch.no_grad():
        states, _ = state_rows(teacher, (batch, rows, positions), True)
        decoder_output = teacher.model(**batch, use_cache=False, output_hidden_states=True)
    assert positions.tolist() == [1, 2, 1]
    assert len(states) == 3 and torch.equal(states[-1], decoder_output.last_hidden_state[rows, positions])
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    calls = []
    optimizer.register_step_post_hook(lambda *_: calls.append(1))
    param = student.model.layers[-1].mlp.down_proj.weight
    before = param.detach().clone()
    record = update(student, teacher, optimizer, prefixes, rollouts, .1, .9, 0, 'cpu', support=43)
    assert calls == [1] and record['optimizer_updates'] == 1
    assert record['raw_rep_loss'] > 0 and record['gradient_norm'] > 0 and math.isfinite(record['loss'])
    assert not torch.equal(before, param)
    assert all(p.grad is None for p in teacher.parameters())


def test_latent_mopd_three_objective_matches_independent_reference():
    from src.losses.latent_opd import response_batch, reward_scale
    from src.training.latent_mopd_three import objective, score
    from src.training.rollout_v3 import HFRolloutEngine
    from src.training.sampling_v2 import generation_config, resolved_contract

    torch.manual_seed(71)
    config = Qwen2Config(vocab_size=48, hidden_size=24, intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False,
                         eos_token_id=9, pad_token_id=0)
    student = Qwen2ForCausalLM(config).eval()
    teacher = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    support = 43
    prefixes = [[1, 2], [3, 4]]
    with torch.no_grad():
        student.lm_head.weight[support:].mul_(64)
        teacher.lm_head.weight[support:].mul_(96)
    generation = generation_config(config.eos_token_id, config.pad_token_id, 6)
    contract = resolved_contract(student, generation, support, 'cpu')
    engine = HFRolloutEngine(student, {'math': generation}, {'math': contract},
                             support, config.pad_token_id, config.eos_token_id, 2)
    generated = engine.generate(prefixes, 'math')
    assert all(row['response_ids'] and max(row['response_ids']) < support for row in generated)
    # Left-pad row 2 and right-pad row 1. The final response token in each row is EOS.
    ids = torch.tensor([[1, 2, 7, 8, 9, 0], [0, 0, 3, 4, 6, 9]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]])
    response = response_batch(ids, mask, prefixes, support)
    batch, batch_rows, positions = response
    assert batch['input_ids'] is ids and batch['attention_mask'] is mask
    expected_rows, expected_positions = [0, 0, 0, 1, 1], [1, 2, 3, 3, 4]
    assert batch_rows.tolist() == expected_rows
    assert positions.tolist() == expected_positions
    assert ids[batch_rows, positions + 1].tolist() == [7, 8, 9, 6, 9]

    def expected_states(model):
        intermediate = []
        blocks = model.model.layers[-3:-1]
        hooks = [block.register_forward_hook(lambda _module, _args, out: intermediate.append(out))
                 for block in blocks]
        try:
            out = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            return torch.stack([h[expected_rows, expected_positions] for h in (*intermediate, out.last_hidden_state)])
        finally:
            for hook in hooks:
                hook.remove()

    with torch.no_grad():
        student_layers, teacher_layers = expected_states(student), expected_states(teacher)
        student_logits = student.lm_head(student_layers[-1]).double()
        teacher_logits = teacher.lm_head(teacher_layers[-1]).double()
        assert student_logits.softmax(-1)[:, support:].sum(-1).max() > 0.5
        assert teacher_logits.softmax(-1)[:, support:].sum(-1).max() > 0.5
        selected = torch.tensor([sorted(range(support), key=lambda j: float(row[j]), reverse=True)[:16]
                                 for row in student_logits])
        p, q = student_logits[:, :support].softmax(-1), teacher_logits[:, :support].softmax(-1)
        p_selected, q_selected = p.gather(-1, selected), q.gather(-1, selected)
        rewards_expected = p_selected / p_selected.sum(-1, keepdim=True) * (q_selected / p_selected).log()
        flattened = rewards_expected.flatten().tolist()
        mean = sum(flattened) / len(flattened)
        sigma = math.sqrt(sum((value - mean) ** 2 for value in flattened) / (len(flattened) - 1))
        advantages = rewards_expected / sigma
        token_expected = -(advantages * p_selected.log()).sum() / len(expected_rows)
        s, t = [h.double() / h.double().norm(dim=-1, keepdim=True) for h in (student_layers, teacher_layers)]
        latent_expected = ((s - t) ** 2).sum() / (3 * len(expected_rows) * config.hidden_size)
        # d/d logits of the fixed-advantage cross entropy over the audited support.
        d_logits = p * advantages.sum(-1, keepdim=True)
        d_logits.scatter_add_(-1, selected, -advantages)
        d_head = d_logits.T @ student_layers[-1].double() / len(expected_rows)

    candidates, rewards = score(student, teacher, response, support)
    assert torch.equal(candidates, selected)
    torch.testing.assert_close(rewards.double(), rewards_expected, rtol=2e-5, atol=2e-7)
    scale = reward_scale([(candidates[:3], rewards[:3]), (candidates[3:], rewards[3:])])
    assert float(scale) == pytest.approx(sigma, rel=2e-5)
    combined, token, latent = objective(student, teacher, response, candidates, rewards / scale, .1, .9, support)
    assert float(token) == pytest.approx(float(token_expected), rel=2e-5, abs=2e-6)
    assert float(latent) == pytest.approx(float(latent_expected), rel=2e-6)
    assert float(combined.detach()) == pytest.approx(float(.1 * token_expected + .9 * latent_expected), rel=2e-5)
    combined.backward()
    head_grad = student.lm_head.weight.grad.double()
    torch.testing.assert_close(head_grad[:support], .1 * d_head, rtol=2e-4, atol=8e-7)
    assert torch.count_nonzero(head_grad[support:]) == 0
    chosen_body_weight = student.model.layers[-1].mlp.down_proj.weight
    ref_layers = expected_states(student)
    ref_logprobs = student.lm_head(ref_layers[-1])[:, :support].double().log_softmax(-1).gather(-1, selected)
    ref_s = ref_layers.double() / ref_layers.double().norm(dim=-1, keepdim=True)
    ref_total = .1 * (-(advantages * ref_logprobs).sum() / len(expected_rows)) + .9 * (ref_s - t).square().mean()
    expected_body_grad, = torch.autograd.grad(ref_total, chosen_body_weight)
    torch.testing.assert_close(chosen_body_weight.grad, expected_body_grad, rtol=2e-4, atol=8e-7)
    assert all(param.grad is None for param in teacher.parameters())
    print(f'token={float(token):.9f} latent={float(latent):.9f} std={float(scale):.9f} '
          f'max_head_grad_error={float((head_grad[:support] - .1 * d_head).abs().max()):.3g} '
          f'max_body_grad_error={float((chosen_body_weight.grad - expected_body_grad).abs().max()):.3g}')


def test_latent_mopd_three_recovers_from_bad_latest_checkpoint(tmp_path, capsys):
    import random

    import numpy as np

    from src.training.latent_mopd_three import latest_resumable_checkpoint, save_checkpoint as save_full
    from src.training.opd_v3 import restore_rng
    from src.training.runtime_v3 import VERSION, prepare_output

    def one_step(model, optimizer):
        offset = random.randrange(1, 8) + int(np.random.randint(1, 8))
        tokens = torch.randint(1, 15, (1, 4)) + offset
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens).logits.square().mean()
        loss.backward()
        optimizer.step()
        return tokens

    random.seed(11)
    np.random.seed(12)
    model = tiny_qwen(dtype=torch.float32)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    identity = {'schema': VERSION, 'hard': {'variant': 'latent'}, 'soft': {}}
    prepare_output(tmp_path, identity)
    one_step(model, optim)
    save_full(tmp_path, model, optim, identity, 1, keep=3)
    expected_tokens = one_step(model, optim)
    save_full(tmp_path, model, optim, identity, 2, keep=3)
    save_full(tmp_path, model, optim, {**identity, 'hard': {'variant': 'token'}}, 3, keep=3)
    (tmp_path / 'checkpoint-2/student/model.safetensors').write_bytes(b'corrupt')
    (tmp_path / 'metrics.jsonl').write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n')

    selected, saved = latest_resumable_checkpoint(tmp_path, identity)
    assert selected.name == 'checkpoint-1' and saved['step'] == 1
    messages = capsys.readouterr().out
    assert 'skipping checkpoint-3: Run identity mismatch' in messages
    assert 'skipping checkpoint-2: Checkpoint digest mismatch: student/model.safetensors' in messages
    resumed = Qwen2ForCausalLM.from_pretrained(selected / 'student', local_files_only=True)
    resumed_optim = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
    resumed_optim.load_state_dict(saved['optimizer'])
    prepare_output(tmp_path, identity, 1, 'archive_future_v2')
    archive, = tmp_path.glob('resume_archive_*')
    assert sorted(p.name for p in archive.iterdir()) == ['checkpoint-2', 'checkpoint-3', 'metrics.jsonl']
    assert (tmp_path / 'metrics.jsonl').read_text() == '{"step": 1}\n'
    restore_rng(saved)
    assert torch.equal(one_step(resumed, resumed_optim), expected_tokens)
    for expect, actual in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(expect, actual)
    for state1, state2 in zip(optim.state.values(), resumed_optim.state.values(), strict=True):
        assert state1.keys() == state2.keys()
        assert all(torch.equal(state1[key], state2[key]) for key in state1)
    (selected / 'checkpoint_manifest.json').unlink()
    with pytest.raises(ValueError, match='No valid checkpoint matches the run identity'):
        latest_resumable_checkpoint(tmp_path, identity)


def test_latent_mopd_three_dev_rejects_corrupt_shard_before_generation(tmp_path, monkeypatch):
    import hashlib
    import sys
    from functools import partial

    from scripts import evaluate_latent_mopd_three_dev as evaluation
    from src.training.latent_mopd_three import restore_checkpoint, save_checkpoint as save_full, verify_checkpoint_files
    from src.training.runtime_v3 import VERSION

    model = tiny_qwen(dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters())
    identity = {'schema': VERSION, 'hard': {'variant': 'latent'}, 'soft': {}}
    monkeypatch.setattr(model, 'save_pretrained', partial(model.save_pretrained, max_shard_size='2KB'))
    save_full(tmp_path, model, optimizer, identity, 1, keep=2)
    checkpoint = tmp_path / 'checkpoint-1'
    manifest = checkpoint / 'checkpoint_manifest.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_checkpoint_files(checkpoint) == digest

    # A damaged optimizer payload blocks training resume but does not change evaluation weights.
    (checkpoint / 'engine.pt').write_bytes(b'corrupt')
    assert verify_checkpoint_files(checkpoint, verify_optimizer=False) == digest
    with pytest.raises(ValueError, match='digest mismatch: engine.pt'):
        restore_checkpoint(checkpoint, identity)

    dev = tmp_path / 'code_dev.jsonl'
    dev.write_text('{"sample_id": "probe"}\n')
    monkeypatch.setattr(evaluation, 'load_dev', lambda _: {'code_dev': ('code', dev, [{'sample_id': 'probe'}])})
    monkeypatch.setattr(evaluation, 'render', lambda *_: [1, 2])
    monkeypatch.setattr(evaluation, 'tokenizer_audit', lambda _: (type('Tok', (), {'eos_token_id': 2})(), None, None))
    calls = []

    def fake_generate(args, *_):
        calls.append(args.model)
        return {'code_dev': [('thinking</think>pass', 4, 'stop', [1, 2, 3, 4])]}, {}

    monkeypatch.setattr(evaluation, 'hf_generate', fake_generate)
    argv = ['evaluate', '--manifest', str(tmp_path / 'release.json'), '--checkpoint', str(checkpoint), '--output-dir']
    monkeypatch.setattr(sys, 'argv', [*argv, str(tmp_path / 'verified_eval')])
    evaluation.main()
    assert calls == [str(checkpoint / 'student')]
    summary = json.loads((tmp_path / 'verified_eval/code_dev.summary.json').read_text())
    assert summary['checkpoint_manifest_sha256'] == digest

    index = json.loads((checkpoint / 'student/model.safetensors.index.json').read_text())
    shard = next(iter(index['weight_map'].values()))
    (checkpoint / 'student' / shard).write_bytes(b'corrupt')
    monkeypatch.setattr(sys, 'argv', [*argv, str(tmp_path / 'blocked_eval')])
    with pytest.raises(ValueError, match='digest mismatch: student/'):
        evaluation.main()
    assert len(calls) == 1 and not (tmp_path / 'blocked_eval').exists()
