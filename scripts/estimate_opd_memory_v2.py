"""Analytical bounds, NOT GPU measurements. Full model logits remain allocated."""
import json
from pathlib import Path
from src.training.runtime_v2 import write


def parameter_estimate(c):
    h, inter, n = c['hidden_size'], c['intermediate_size'], c['num_hidden_layers']
    kv = h // c['num_attention_heads'] * c['num_key_value_heads']
    # Qwen2-style projections, QKV bias, two per-layer norms plus final norm.
    return c['vocab_size'] * h * (1 if c.get('tie_word_embeddings') else 2) + n * (2*h*h + 2*h*kv + 3*h*inter + 2*h + h + 2*kv) + h


def main():
    path = Path('data/tokenizers')
    s = json.loads((path / 'Qwen2.5-1.5B-Instruct/config.json').read_text())
    t = json.loads((path / 'Qwen2.5-Coder-7B-Instruct/config.json').read_text())
    gib = 2**30
    h, kv = s['hidden_size'], s['hidden_size'] // s['num_attention_heads'] * s['num_key_value_heads']
    lora = s['num_hidden_layers'] * 16 * (4*h + 2*(h+kv) + 3*(h+s['intermediate_size']))
    rows = []
    for length in [2048, 4096, 2560+1536]:
        fixed = {'student_and_one_teacher_weights': (parameter_estimate(s) + parameter_estimate(t))*2/gib,
                 'lora_grad_adam_master_allowance': lora*16/gib,
                 'student_teacher_full_bf16_logits': length*(s['vocab_size']+t['vocab_size'])*2/gib,
                 'student_full_bf16_logit_gradient': length*s['vocab_size']*2/gib,
                 'rollout_student_kv': 2*s['num_hidden_layers']*length*kv*2/gib}
        # 10 FP32 support arrays plus a full-head row selection and temporaries.
        workspace = 16 * 151665 * 4 * 10 / gib + 16 * 152064 * 2 * 2 / gib
        lower = sum(fixed.values()) + workspace
        # Explicit assumptions, not a promised peak; activations/SDPA/loading unknown.
        rows.append({'L': length, 'components_gib': fixed, 'chunk16_workspace_allowance_gib': workspace,
            'sum_conservative_components_gib': lower,
            'planning_envelope_gib': [lower + 6, lower + 14],
            'unknowns': 'Activation/checkpoint/SDPA/kernel/loading/allocator/host peak 未验证; KV not necessarily concurrent with loss'})
    result = {'status': '未验证: analytical planning only', 'assumptions': 'B=1, BF16, LoRA r16, one 7B teacher, no teacher scoring KV, no top-k',
        'parameter_estimates': {'student': parameter_estimate(s), 'teacher': parameter_estimate(t), 'lora': lora},
        'length_limit': 'math P<=3072 +1024; code P<=2560 +1536; L<=4096',
        'rows': rows, 'device_plan': {
            'A100_40GB': {'response_row_chunk_size': 16, 'sequence_chunk_size': 128, 'status': 'provisional low-workspace starting point; overall capacity 未验证; require >=4GiB measured headroom'},
            'A100_80GB': {'response_row_chunk_size': 16, 'sequence_chunk_size': 128, 'status': 'same conservative starting point; overall capacity 未验证; only raise chunks after measured peak'}},
        'warning': 'No chunk size guarantees safety while full logits and model activations remain. CPU tests cannot certify A100 memory.'}
    write('reports/handover_20261024_v2/memory_estimate.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
