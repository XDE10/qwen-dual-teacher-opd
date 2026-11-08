import copy
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from src.evaluation.math_verifier import check
from src.training.opd import update_loss,response_batch


def tiny():
    return GPT2LMHeadModel(GPT2Config(vocab_size=24,n_positions=32,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))


def test_opd_exact_prefix_frozen_teacher_and_update():
    torch.manual_seed(7);s=tiny();t=tiny();before=s.transformer.wte.weight.detach().clone();teacher=copy.deepcopy(t.state_dict())
    ids,attention,n=response_batch(torch.tensor([[1,2,3,4,5,6]]),[1,2],5)
    assert ids.tolist()==[[1,2,3,4,5]]
    loss,mass=update_loss(s,t,ids,attention,[1,2],24);assert torch.isfinite(loss)
    optimizer=torch.optim.AdamW(s.parameters(),lr=.01);loss.backward();optimizer.step()
    assert not torch.equal(before,s.transformer.wte.weight)
    assert all(p.grad is None for p in t.parameters())
    assert all(torch.equal(v,t.state_dict()[k]) for k,v in teacher.items())
    assert abs(mass['teacher_removed_mass'])<1e-5


def test_bounded_real_math_reward():
    assert check(r'Here is the result: \\boxed{2}', '2')['correct']
    assert not check(r'\\boxed{3}', '2')['correct']
