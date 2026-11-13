"""Measure student-teacher distance on the untrained student's rollouts (no training).

  python -m scripts.probe_teacher_fit_v3 --config configs/project2_opd_v3_d4_dual.json \
      --output "$OPD_OUTPUT_ROOT/probe/teacher_fit.json" --dump-tokens 6

By default it probes the config's math and code teachers plus the general
Qwen2.5-7B-Instruct teacher, on both domains. That gives a 3 x 2 matrix, the
cross-domain cells included. Teachers are loaded one at a time (about 14 GiB each).
Each cell lists the sampled tokens that carry the most KL (decoded). --dump-tokens K
also writes <output stem>_tokens.jsonl with per-token values for the first K rollouts
of each domain, for every teacher.
"""
import argparse
import json
import time
from pathlib import Path

GENERAL = {'model': 'Qwen/Qwen2.5-7B-Instruct', 'revision': 'a09a35458c702b33eeacc393d103063234e8bc28'}


def parse_teacher(text):
    name, spec = text.split('=', 1)
    model, revision = spec.split('@', 1)
    return name, {'model': model, 'revision': revision}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--teachers', nargs='*', help='name=model@revision; default: config teachers + general')
    p.add_argument('--prompts-per-domain', type=int, default=64)
    p.add_argument('--rollout-batch-size', type=int, default=64)
    p.add_argument('--microbatch-tokens', type=int, default=16384)
    p.add_argument('--dump-tokens', type=int, default=0, metavar='K',
                   help='write per-token values for the first K rollouts per domain to <output stem>_tokens.jsonl')
    a = p.parse_args()
    import torch
    from transformers import AutoTokenizer, set_seed
    from src.teachers.opd_contract import load_contract
    from src.training.opd_v3 import audited_support, frozen_load, load_config, load_model, load_pools, render_prefixes
    from src.training.probe_v3 import probe, verdict
    from src.training.rollout_v3 import HFRolloutEngine
    from src.training.sampler_v3 import StratifiedSampler
    from src.training.sampling_v2 import generation_config, resolved_contract
    c = load_config(a.config)
    teachers = dict(parse_teacher(t) for t in a.teachers) if a.teachers else {
        **{f'{d}_teacher': spec for d, spec in c['teachers'].items()}, 'general_teacher': GENERAL}
    audit = json.loads(Path(c['tokenizer_audit']).read_text(encoding='utf-8'))
    pinned = {r['model']: r['revision'] for r in audit['models']}
    for name, spec in teachers.items():
        if pinned.get(spec['model']) != spec['revision']:
            raise ValueError(f'{name}: {spec} is not in the tokenizer audit')
    support = audited_support(c)
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['tokenizer_path'], local_files_only=True)
    pools = load_pools(c)
    prefixes = render_prefixes(pools, tok, load_contract())
    rows = {d: StratifiedSampler(r, d, c['seed'], c['stratify_keys'].get(d)).next(a.prompts_per_domain)
            for d, r in pools.items()}
    seconds = {}
    start = time.monotonic()
    student = load_model(c['student'], 'cuda').eval().requires_grad_(False)
    configs = {d: generation_config(tok.eos_token_id, tok.pad_token_id, n) for d, n in c['max_new_tokens'].items()}
    contracts = {d: resolved_contract(student, cfg, support, student.device) for d, cfg in configs.items()}
    engine = HFRolloutEngine(student, configs, contracts, support, tok.pad_token_id, tok.eos_token_id, a.rollout_batch_size)
    rollouts = {}
    for d in rows:
        t0 = time.monotonic()
        rollouts[d] = engine.generate([prefixes[r['sample_id']] for r in rows[d]], d)
        seconds['rollout_' + d] = round(time.monotonic() - t0, 1)
    loaders = {}
    for name, spec in teachers.items():
        def load(spec=spec, name=name):
            t0 = time.monotonic()
            model = frozen_load(spec, 'cuda')
            seconds['load_' + name] = round(time.monotonic() - t0, 1)
            return model
        loaders[name] = load
    t0 = time.monotonic()
    results = probe(student, loaders, rollouts, support, tok.pad_token_id, 'cuda', a.microbatch_tokens, a.dump_tokens)
    seconds['score_all_teachers'] = round(time.monotonic() - t0, 1)
    seconds['total'] = round(time.monotonic() - start, 1)
    piece = lambda i: tok.decode([i], skip_special_tokens=False)
    dumps = []
    for name, cells in results.items():
        for d, cell in cells.items():
            cell['top_kl_tokens'] = [[piece(entry[0]), *entry] for entry in cell['top_kl_tokens']]
            for item in cell.pop('token_dump', []):
                index = item['rollout']
                dumps.append({'teacher': name, 'domain': d, 'rollout': index, 'sample_id': rows[d][index]['sample_id'],
                              'ended_with_eos': rollouts[d][index]['ended_with_eos'],
                              'fields': ['response_position', 'token', *item['fields'][2:]],
                              'tokens': [[t[0], piece(t[1]), *t[2:]] for t in item['tokens']]})
    report = {'config': a.config, 'student': c['student'], 'teachers': teachers, 'prompts_per_domain': a.prompts_per_domain,
              'sample_ids': {d: [r['sample_id'] for r in rows[d]] for d in rows}, 'gpu': torch.cuda.get_device_name(0),
              'seconds': seconds, 'results': results, 'reading': verdict(results),
              'top_kl_tokens_fields': ['token', 'token_id', 'count', 'share_of_kl', 'mean_kl',
                                       'mean_student_entropy', 'mean_teacher_entropy'],
              'reference_points': 'MOPD Sec. 4.4.2: same-origin initial KL ~0.04, external stronger teacher ~0.19'}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if dumps:
        tokens_path = output.with_name(output.stem + '_tokens.jsonl')
        tokens_path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in dumps), encoding='utf-8')
    print('\n'.join(report['reading']))
    print(json.dumps({'seconds': seconds}))


if __name__ == '__main__':
    main()
