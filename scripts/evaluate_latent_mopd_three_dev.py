"""Generate matched math/code/logic dev evaluations for three-teacher Latent-MOPD."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

from scripts.generate_eval_v3 import RELEASE, hf_generate
from scripts.prepare_latent_mopd_three import GYM_VERSION, json_rows, write_rows
from src.evaluation.eval_v3 import grade_math, render, suite_rows, summarize
from src.teachers.opd_contract import load_contract
from src.training.latent_mopd_three import STUDENT, SUPPORT, render_prefix, tokenizer_audit, verify_checkpoint_files
from src.training.runtime_v2 import read, sha, write


def final_response(text):
    _, close, answer = text.partition('</think>')
    return answer.strip() if close else ''


@lru_cache
def logic_generator(task, seed, size):
    import reasoning_gym

    if reasoning_gym.__version__ != GYM_VERSION:
        raise ValueError('Logic verifier requires reasoning-gym==' + GYM_VERSION)
    return reasoning_gym.create_dataset(task, seed=seed, size=size)


def grade_logic(row, text):
    from reasoning_gym.utils import extract_answer

    generator = logic_generator(row['task'], row['generator_seed'], row['generator_size'])
    # Some generators depend on Python's hash ordering. Score the sealed
    # question and metadata that the model actually saw.
    source = {'question': row['problem'], 'answer': row['answer'], 'metadata': row['metadata']}
    answer = extract_answer(text)
    return float(generator.score_answer(answer, source)) if answer is not None else 0.0


def load_dev(manifest):
    logic_name = 'logic_dev.jsonl'
    doc = read(manifest)
    if logic_name not in doc.get('approved_evaluation_files', []):
        raise ValueError('Logic dev is outside evaluation approval scope')
    logic_path = manifest.parent / logic_name
    if sha(logic_path) != doc['files'][logic_name]['sha256']:
        raise ValueError('Logic dev digest differs from release')
    legacy = read(RELEASE)
    suites = {name: suite_rows(name, RELEASE) for name in ('math_dev', 'code_dev')}
    for _, path, _ in suites.values():
        if path.name not in legacy.get('approved_evaluation_files', []):
            raise ValueError(path.name + ' is outside evaluation approval scope')
        meta = legacy['files'][path.name]
        if sha(path) != (meta['sha256'] if isinstance(meta, dict) else meta):
            raise ValueError(path.name + ' differs from v5 release')
    return {**suites, 'logic_dev': ('logic', logic_path, list(json_rows(logic_path)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Full checkpoint directory. Omit for Base R1.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('runs/latent_mopd_three_assets'))
    parser.add_argument('--max-new-tokens', type=int, default=16384)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    checkpoint_digest = (verify_checkpoint_files(args.checkpoint, verify_optimizer=False)
                         if args.checkpoint else None)
    loaded = load_dev(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    tokenizer, _, _ = tokenizer_audit(args.assets)
    contract = load_contract()
    prompts = {name: [render_prefix(tokenizer, row['prompt']) if kind == 'logic'
                      else render(tokenizer, kind, row, contract) for row in rows]
               for name, (kind, _, rows) in loaded.items()}
    args.model = str(args.checkpoint / 'student') if args.checkpoint else STUDENT['model']
    args.revision = None if args.checkpoint else STUDENT['revision']
    args.adapter = None
    budgets = dict.fromkeys(loaded, args.max_new_tokens)
    results, engine = hf_generate(args, prompts, budgets, SUPPORT, tokenizer.eos_token_id, tokenizer)
    identity = {'protocol': 'latent-mopd-three-dev-greedy', 'model': args.model, 'revision': args.revision,
                'checkpoint_manifest_sha256': checkpoint_digest,
                'engine': engine, 'max_new_tokens': args.max_new_tokens, 'after_think': True}
    args.output_dir.mkdir(parents=True)
    for name, (kind, path, rows) in loaded.items():
        records = []
        for row, (text, n, finish, token_ids) in zip(rows, results[name], strict=True):
            final = final_response(text)
            record = {'id': row['sample_id'], 'text': text, 'final': final, 'generated_tokens': n,
                      'finish_reason': finish, 'response_ids': token_ids}
            if kind == 'math':
                record['grade'] = grade_math(final, row['reference_answer'], finish == 'length')
            elif kind == 'logic':
                record['verifier_score'] = grade_logic(row, final)
            records.append(record)
        summary = {**identity, **summarize(kind, records), 'source_sha256': sha(path)}
        if kind == 'code':
            write_rows(args.output_dir / 'code_dev.candidates.jsonl',
                       ({'sample_id': r['id'], 'text': r['final']} for r in records))
        if kind == 'logic':
            scores = [r['verifier_score'] for r in records]
            summary.update(accuracy=sum(score == 1.0 for score in scores) / len(scores),
                           mean_verifier_score=sum(scores) / len(scores))
        write_rows(args.output_dir / f'{name}.jsonl', records)
        write(args.output_dir / f'{name}.summary.json', summary)
        print(name, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
