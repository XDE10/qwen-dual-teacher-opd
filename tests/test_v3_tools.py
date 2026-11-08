"""CPU tests: SFT warm start, teacher-fit probe, run summary and init_adapter helpers (tiny models)."""
import json

import pytest
import torch
import torch.nn.functional as F

from src.losses.sft_v3 import row_cross_entropy
from src.training.opd_v3 import adapter_digests, check_adapter_matches, lr_lambda, resolve_run_path
from src.training.probe_v3 import probe, verdict
from src.training.sft_v3 import SFTTrainerV3, load_examples
from test_opd_v3 import EOS, PAD, SUPPORT, build_trainer, engine_for, peft_student, tiny_qwen, trainer_config


def test_row_cross_entropy_matches_torch_value_and_gradient():
    torch.manual_seed(0)
    logits = torch.randn(9, 13, dtype=torch.double, requires_grad=True)
    targets = torch.randint(0, 13, (9,))
    weights = torch.full((9,), 1 / 9, dtype=torch.double)
    for chunk in (1, 4, 64):
        loss, nll = row_cross_entropy(logits, targets, weights, chunk)
        reference = F.cross_entropy(logits, targets)
        torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
        expected = torch.autograd.grad(reference, logits)[0]
        actual = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(nll.double(), F.cross_entropy(logits, targets, reduction='none').detach(),
                                   rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match='outside'):
        row_cross_entropy(logits, torch.full((9,), 13), weights)


def write_outputs(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')


def test_load_examples_filters_and_restores_eos(tmp_path):
    pools = {'math': [{'sample_id': f'm{i}', 'domain': 'math'} for i in range(4)],
             'code': [{'sample_id': 'c0', 'domain': 'code'}]}
    prefixes = {'m0': [1, 3], 'm1': [1, 4], 'm2': [1, 5], 'm3': [1, 6], 'c0': [7, 8]}
    write_outputs(tmp_path / 'math.jsonl', [
        {'id': 'm0', 'prompt_tokens': 2, 'finish_reason': 'stop', 'response_ids': [9, 10], 'grade': {'math_verify': {'correct': True}}},
        {'id': 'm1', 'prompt_tokens': 2, 'finish_reason': 'length', 'response_ids': [9, 9]},
        {'id': 'm2', 'prompt_tokens': 2, 'finish_reason': 'stop', 'response_ids': [11, EOS], 'grade': {'math_verify': {'correct': False}}}])
    write_outputs(tmp_path / 'code.jsonl', [{'id': 'c0', 'prompt_tokens': 2, 'finish_reason': 'stop', 'response_ids': [12, EOS]}])
    c = {'teacher_outputs': {'math': str(tmp_path / 'math.jsonl'), 'code': str(tmp_path / 'code.jsonl')},
         'filters': {'require_eos': True, 'math_correct_only': False}}
    examples, stats = load_examples(c, pools, prefixes, EOS, SUPPORT)
    assert [e['sample_id'] for e in examples] == ['m0', 'm2', 'c0']
    assert examples[0]['response_ids'] == [9, 10, EOS] and examples[1]['response_ids'] == [11, EOS]
    assert stats == {'math_kept': 2, 'math_truncated': 1, 'math_missing': 1, 'code_kept': 1}
    c['filters']['math_correct_only'] = True
    examples, stats = load_examples(c, pools, prefixes, EOS, SUPPORT)
    assert [e['sample_id'] for e in examples] == ['m0', 'c0'] and stats['math_incorrect'] == 1
    prefixes['m0'] = [1, 3, 3]
    with pytest.raises(ValueError, match='prompt contract'):
        load_examples(c, pools, prefixes, EOS, SUPPORT)


SFT_EXAMPLES = [{'sample_id': f'x{i}', 'domain': 'math' if i % 2 else 'code', 'prompt_ids': [1 + i, 4],
                 'response_ids': [10 + i, 20, EOS]} for i in range(6)]


def sft_trainer(student, lr, epochs):
    c = {'seed': 1, 'epochs': epochs, 'examples_per_step': 6, 'train_microbatch_tokens': 12, 'row_chunk': 4,
         'max_grad_norm': 1e9, 'warmup_steps': 0, 'lr_schedule': 'constant'}
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr) if lr == 0 else torch.optim.AdamW(params, lr=lr)
    trainer = SFTTrainerV3(c, student, SFT_EXAMPLES, SUPPORT, PAD, 'cpu', optimizer, None)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda({**c, 'max_steps': trainer.total_steps}))
    return trainer


def test_sft_step_equals_full_logit_cross_entropy_on_response_tokens():
    import copy
    from src.training.opd_v3 import right_pad
    torch.manual_seed(3)
    student = peft_student(tiny_qwen(seed=5))
    reference = copy.deepcopy(student)
    record = sft_trainer(student, 0.0, 1).train_step()  # lr 0: gradients stay in .grad
    sequences = [e['prompt_ids'] + e['response_ids'] for e in SFT_EXAMPLES]
    ids, attention = right_pad(sequences, PAD, 'cpu')
    reference.train()
    logits = reference(input_ids=ids, attention_mask=attention).logits[..., :SUPPORT]
    total = sum(F.cross_entropy(logits[i, len(e['prompt_ids']) - 1:len(sequences[i]) - 1],
                                ids[i, len(e['prompt_ids']):len(sequences[i])], reduction='sum')
                for i, e in enumerate(SFT_EXAMPLES))
    loss = total / sum(len(e['response_ids']) for e in SFT_EXAMPLES)
    loss.backward()
    assert record['loss_token_mean_nll'] == pytest.approx(float(loss.detach()), rel=1e-7)
    assert set(record['domain_nll_token_mean']) == {'math', 'code'} and record['response_tokens'] == 18
    for (_, a), (_, b) in zip(student.named_parameters(), reference.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_sft_training_lowers_loss_and_keeps_base_frozen():
    torch.manual_seed(3)
    student = peft_student(tiny_qwen(seed=5))
    frozen = {n: p.detach().clone() for n, p in student.named_parameters() if not p.requires_grad}
    trainer = sft_trainer(student, 5e-2, 15)
    first = trainer.train_step()
    while trainer.step < trainer.total_steps:
        last = trainer.train_step()
    assert last['loss_token_mean_nll'] < first['loss_token_mean_nll']
    assert all(torch.equal(p, frozen[n]) for n, p in student.named_parameters() if not p.requires_grad)


def test_probe_reports_zero_kl_for_an_identical_teacher_and_positive_otherwise():
    torch.manual_seed(4)
    student = tiny_qwen(seed=21)
    engine = engine_for(student, max_new=5)
    rollouts = {'math': engine.generate([[1, 3], [4, 5, 6], [7]], 'math'), 'code': engine.generate([[9, 10], [11]], 'code')}
    twin = tiny_qwen(seed=21)
    results = probe(student, {'twin': lambda: twin, 'other': lambda: tiny_qwen(vocab=48, tied=False, seed=8)},
                    rollouts, SUPPORT, PAD, 'cpu', microbatch_tokens=16)
    for d in ('math', 'code'):
        assert abs(results['twin'][d]['kl_exact_token_mean']) < 1e-9
        assert results['other'][d]['kl_exact_token_mean'] > 1e-3
        assert 0 <= results['other'][d]['teacher_prob_below_1e-2'] <= 1
        assert results['twin'][d]['rollouts'] == len(rollouts[d])
    lines = verdict(results)
    assert len(lines) == 4 and 'close to MOPD same-origin' in lines[0]


def test_summary_reads_a_real_step_record(tmp_path):
    from scripts.summarize_run_v3 import summarize
    torch.manual_seed(14)
    trainer = build_trainer(trainer_config(), tiny_qwen(seed=99).state_dict())
    record, _ = trainer.train_step()
    (tmp_path / 'metrics.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    summary = summarize(tmp_path)
    assert summary['steps'] == 1 and summary['per_step'][0]['math']['kl'] > 0
    assert 'forward_math' in summary['phase_seconds_total'] and summary['per_step'][0]['cosine_math_code'] is not None


def test_run_paths_and_adapter_checks(tmp_path):
    assert resolve_run_path('runs/sft/final_adapter', '/drive/x').as_posix() == '/drive/x/sft/final_adapter'
    assert resolve_run_path('runs/sft', None).as_posix() == 'runs/sft'
    assert resolve_run_path('/abs/path', '/drive/x').as_posix() == '/abs/path'
    student = peft_student(tiny_qwen())
    student.save_pretrained(tmp_path / 'adapter')
    digests = adapter_digests(tmp_path / 'adapter')
    assert {'adapter_config.json', 'adapter_model.safetensors'} <= set(digests)
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    check_adapter_matches(tmp_path / 'adapter', {'r': 2, 'alpha': 4, 'target_modules': targets})
    with pytest.raises(ValueError, match='differs'):
        check_adapter_matches(tmp_path / 'adapter', {'r': 16, 'alpha': 32, 'target_modules': targets})
    with pytest.raises(ValueError, match='not a PEFT adapter'):
        adapter_digests(tmp_path)
