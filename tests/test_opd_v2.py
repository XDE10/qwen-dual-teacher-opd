import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.losses.reference import aligned_kl
from src.losses.chunked_opd_v2 import chunked_reverse_kl, gradient_normalizer, removed_mass_values, summarize_mass
from src.training.sampling_v2 import generation_config, resolved_contract, generate_exact
from src.training.runtime_v2 import (REQUIRED, build_identity, seal_checkpoint,
    validate_checkpoint, prepare_output, write, teacher_key, PhaseRecorder)


def tiny():
    return GPT2LMHeadModel(GPT2Config(vocab_size=17, n_positions=32, n_embd=12,
        n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()


def test_actual_sampling_chain_ignores_hostile_model_defaults():
    model = tiny()
    model.generation_config.top_k = 1
    model.generation_config.top_p = .1
    model.generation_config.repetition_penalty = 4.
    model.generation_config.suppress_tokens = [1, 3]
    config = generation_config(2, 0, 3)
    contract = resolved_contract(model, config, 11, torch.device('cpu'))
    with torch.no_grad():
        result = generate_exact(model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.long),
            config, 11, contract['logits_processors'], diagnostics=True)
    for scores in result.scores:
        p = scores.softmax(-1)
        assert (p[:, :11] > 0).all()  # no inherited top-k/top-p/suppression
        assert torch.count_nonzero(p[:, 11:]) == 0
    assert len(contract['logits_processors']) == 1
    assert contract['generation_config']['repetition_penalty'] == 1.
    assert model.generation_config.top_k == 1  # original remains unchanged


def test_peft_actual_chain_is_audited():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(tiny(), LoraConfig(r=2, target_modules=['c_attn'], task_type='CAUSAL_LM'))
    model.get_base_model().generation_config.suppress_tokens = [1, 3]
    config = generation_config(2, 0, 2)
    contract = resolved_contract(model.get_base_model(), config, 11, torch.device('cpu'))
    with torch.no_grad():
        out = generate_exact(model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.long),
                             config, 11, contract['logits_processors'], diagnostics=True)
    assert all(torch.count_nonzero(s.softmax(-1)[:, 11:]) == 0 for s in out.scores)


@pytest.mark.parametrize('row_chunk,sequence_chunk', [(1, 2), (3, 4), (16, 128)])
def test_exact_chunked_fp64_value_and_gradient(row_chunk, sequence_chunk):
    torch.manual_seed(9)
    s = torch.randn(2, 9, 19, dtype=torch.double, requires_grad=True)
    t = torch.randn(2, 9, 23, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    expected = aligned_kl(s, t, mask, range(17), range(17))
    eg = torch.autograd.grad(expected, s)[0]
    actual = chunked_reverse_kl(s, t, mask, 17, sequence_chunk, row_chunk)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(s.grad, eg, rtol=1e-12, atol=1e-12)
    assert t.grad is None
    assert torch.count_nonzero(s.grad[:, :, 17:]) == 0
    assert torch.count_nonzero(s.grad[:, :-1][~mask[:, 1:]]) == 0
    assert torch.count_nonzero(s.grad[:, -1]) == 0


def test_chunked_gradcheck_and_empty_response():
    s = torch.randn(1, 4, 5, dtype=torch.double, requires_grad=True)
    t = torch.randn(1, 4, 7, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x: chunked_reverse_kl(x, t, mask, 4, 2, 1), (s,))
    with pytest.raises(ValueError, match='Empty response'):
        chunked_reverse_kl(s, t, torch.zeros_like(mask), 4)


def test_long_short_accumulation_matches_merged_token_gradient():
    torch.manual_seed(4)
    w = torch.randn(4, 7, dtype=torch.double, requires_grad=True)
    features = torch.randn(2, 8, 4, dtype=torch.double)
    teacher = torch.randn(2, 8, 9, dtype=torch.double)
    mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    merged = aligned_kl(features @ w, teacher, mask, range(6), range(6))
    expected = torch.autograd.grad(merged, w)[0]
    for i in range(2):
        count = int(mask[i, 1:].sum())
        loss = chunked_reverse_kl(features[i:i+1] @ w, teacher[i:i+1], mask[i:i+1], 6, 2, 1)
        (loss * count).backward()
    w.grad.div_(gradient_normalizer('response_tokens_v2', int(mask[:, 1:].sum()), 2))
    torch.testing.assert_close(w.grad, expected, rtol=1e-12, atol=1e-12)
    assert gradient_normalizer('microbatch_mean_v1', 8, 2) == 2


def test_mass_uses_all_microbatches_token_weighted():
    a = torch.tensor([.1, .3])
    b = torch.tensor([.8])
    result = summarize_mass([a, b])
    assert result['tokens'] == 3
    assert result['mean'] == pytest.approx(.4)
    assert result['max'] == pytest.approx(.8)
    s = torch.randn(1, 5, 12)
    m = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    values = removed_mass_values(s, m, 9, 2, 1)
    expected = 1 - s[:, :-1][m[:, 1:]].softmax(-1)[:, :9].sum(-1)
    torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-6)


def test_v2_forward_disables_teacher_cache_and_updates_student():
    from src.training.opd_v2 import update_loss
    s, t = tiny(), tiny()
    t.requires_grad_(False)
    calls = []
    hook = t.register_forward_pre_hook(lambda mod, args, kwargs: calls.append(kwargs['use_cache']), with_kwargs=True)
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    loss, mass, count = update_loss(s, t, ids, torch.ones_like(ids), [1, 4], 17, 2, 1)
    loss.backward()
    hook.remove()
    assert calls == [False] and count == 3
    assert s.transformer.wte.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in t.parameters())
    assert len(mass['student']) == 3


def fake_root(tmp_path):
    files = ['src/training/opd_v2.py', 'src/training/sampling_v2.py', 'src/training/runtime_v2.py',
        'src/losses/chunked_opd_v2.py', 'src/losses/reference.py', 'src/teachers/opd_contract.py',
        'src/data/approved_training_inputs.py', 'scripts/train_project2_opd.py', 'src/training/opd.py',
        'configs/project2_prompt_contract.json', 'data/manifest.json',
        'reports/audit.json', 'data/tok/tokenizer.json']
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}', encoding='utf-8')
    return {'data_manifest': 'data/manifest.json', 'tokenizer_audit': 'reports/audit.json', 'tokenizer_path': 'data/tok'}


@pytest.mark.parametrize('name', ['configs/project2_prompt_contract.json',
    'data/tok/tokenizer.json', 'src/training/opd_v2.py', 'src/losses/chunked_opd_v2.py'])
def test_identity_rejects_contract_tokenizer_and_source_changes(tmp_path, name):
    config = fake_root(tmp_path)
    original = build_identity(config, {}, tmp_path)
    ckpt = tmp_path / 'ckpt'
    for n in REQUIRED:
        p = ckpt / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('dummy', encoding='utf-8')
    write(ckpt / 'run_identity.json', original)
    seal_checkpoint(ckpt)
    validate_checkpoint(ckpt, original)
    (tmp_path / name).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='identity'):
        validate_checkpoint(ckpt, build_identity(config, {}, tmp_path))


@pytest.mark.parametrize('attack', ['missing', 'empty', 'escape', 'extra', 'digest'])
def test_checkpoint_closed_set_rejection(tmp_path, attack):
    for n in REQUIRED:
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('x', encoding='utf-8')
    write(tmp_path / 'run_identity.json', {'x': 1})
    seal_checkpoint(tmp_path)
    p = tmp_path / 'checkpoint_manifest.json'
    manifest = json.loads(p.read_text())
    if attack == 'missing':
        manifest['files'].pop('engine.pt')
    elif attack == 'empty':
        manifest['files'] = {}
    elif attack == 'escape':
        manifest['files']['../escape'] = 'x'
    elif attack == 'extra':
        (tmp_path / 'unlisted').write_text('x')
    else:
        (tmp_path / 'engine.pt').write_text('altered')
    write(p, manifest)
    with pytest.raises(ValueError):
        validate_checkpoint(tmp_path, {'x': 1})


def test_resume_requires_explicit_future_archive_and_no_duplicate_steps(tmp_path):
    identity = {'v': 2}
    write(tmp_path / 'run_identity.json', identity)
    write(tmp_path / 'checkpoint-2.json', {})
    (tmp_path / 'checkpoint-2.json').unlink()
    (tmp_path / 'checkpoint-2').mkdir()
    (tmp_path / 'metrics.jsonl').write_text('{"step":1}\n{"step":2}\n')
    with pytest.raises(ValueError, match='Later'):
        prepare_output(tmp_path, identity, 1)
    prepare_output(tmp_path, identity, 1, 'archive_future_v2')
    assert not (tmp_path / 'checkpoint-2').exists()
    assert [json.loads(l)['step'] for l in (tmp_path / 'metrics.jsonl').read_text().splitlines()] == [1]
    assert len(list(tmp_path.glob('resume_archive_*'))) == 1


def test_same_teacher_identity_reused_across_domains():
    teacher = {'model': 'general', 'revision': 'fixed'}
    assert teacher_key(dict(teacher)) == teacher_key(teacher)
    assert teacher_key({**teacher, 'revision': 'different'}) != teacher_key(teacher)


def test_phase_recorder_cpu_reports_no_gpu_claim():
    recorder = PhaseRecorder()
    with recorder.phase('fixture'):
        x = torch.ones(3).sum()
    assert x == 3
    assert recorder.records[0]['cpu_rss_after'] > 0


def test_real_opd_adapter_optimizer_scheduler_rng_resume_cpu(tmp_path):
    import random
    import numpy as np
    from peft import LoraConfig, get_peft_model, PeftModel
    from src.training.opd_v2 import update_loss, save_checkpoint
    torch.manual_seed(123)
    base = tiny()
    original_base = copy.deepcopy(base.state_dict())
    student = get_peft_model(base, LoraConfig(r=2, target_modules=['c_attn'], lora_dropout=0, task_type='CAUSAL_LM'))
    teacher = tiny().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1 - k/4)
    def update(model, opt, sched):
        ids = torch.cat([torch.tensor([[1, 4]]), torch.randint(5, 16, (1, 3))], dim=1)
        opt.zero_grad()
        loss, _, _ = update_loss(model, teacher, ids, torch.ones_like(ids), [1, 4], 17, 2, 1)
        loss.backward()
        opt.step()
        sched.step()
        return ids, float(loss.detach())
    update(student, optimizer, scheduler)
    identity = {'test': 'CPU actual OPD state'}
    checkpoint = tmp_path / 'checkpoint-1'
    save_checkpoint(checkpoint, student, optimizer, scheduler,
                    {'step': 1, 'positions': {'math': 1, 'code': 0}}, identity, [])
    validate_checkpoint(checkpoint, identity)
    expected_ids, expected_loss = update(student, optimizer, scheduler)
    expected_adapter = {k: v.clone() for k, v in student.state_dict().items() if 'lora_' in k}
    restored_base = tiny()
    restored_base.load_state_dict(original_base)
    restored = PeftModel.from_pretrained(restored_base, checkpoint / 'student', is_trainable=True)
    opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1 - k/4)
    saved = torch.load(checkpoint / 'engine.pt', map_location='cpu', weights_only=False)
    opt.load_state_dict(saved['optimizer'])
    sched.load_state_dict(saved['scheduler'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    actual_ids, actual_loss = update(restored, opt, sched)
    assert torch.equal(expected_ids, actual_ids)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-8)
    for k, v in restored.state_dict().items():
        if k in expected_adapter:
            torch.testing.assert_close(v, expected_adapter[k], atol=1e-7, rtol=1e-6)
    assert sched.get_last_lr() == scheduler.get_last_lr()
    assert saved['state']['positions'] == {'math': 1, 'code': 0}
