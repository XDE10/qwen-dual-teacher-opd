import json
import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.training.opd_v3 import right_pad


def tiny_qwen(vocab=40, tied=True, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         tie_word_embeddings=tied, rope_theta=10000.0, use_sliding_window=False,
                         # Production uses sdpa. HF eager + float64 + left padding yields NaN rows.
                         attn_implementation='sdpa')
    return Qwen2ForCausalLM(config).to(dtype).eval()


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
