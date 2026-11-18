"""Build a reviewable, source-pinned three-domain Latent-MOPD prompt release."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from src.data.decontamination_v5 import ProtectedIndex, strip_code_boilerplate, words
from src.data.release_v5 import OFFICIAL_CODE
from src.training.latent_mopd_three import (
    DOMAINS, MAX_PROMPT, PROMPTS, STEPS, STUDENT, build_schedule, render_prefix,
)
from src.training.runtime_v2 import read, sha, write

ROOT = Path('runs/latent_mopd_three')
V5 = Path('data/project2_release_v5_20261108/release')
POOL = 5952
DATASETS = {
    'dapo': ('BytedTsinghua-SIA/DAPO-Math-17k', '65877096c24ffa7abc4e4fa5edb95cf3413a5674'),
    'ocr': ('nvidia/OpenCodeReasoning', '20a1ca19c0d050fe9057fc08339d6b370ec1c67a'),
    'aime24': ('math-ai/aime24', '83a7f387baaa524a8bda0022eac0541582297103'),
    'aime25': ('math-ai/aime25', '563bb8404243c5f09de6ec262f2db674fe5bce9b'),
    'minerva': ('math-ai/minervamath', 'ee46ddc498933b1977577953250ca5c66be64f96'),
    'lcb': ('livecodebench/code_generation_lite', '0fe84c3912ea0c4d4a78037083943e8f0c4dd505'),
}
# Independent task selection. The authors' frozen GYM registry was not released.
LOGIC_TASKS = (
    'acre', 'aiw', 'base_conversion', 'basic_arithmetic', 'binary_alternation',
    'binary_matrix', 'calendar_arithmetic', 'chain_sum', 'circuit_logic', 'count_bits',
    'count_primes', 'course_schedule', 'decimal_arithmetic', 'decimal_chain_sum', 'dice',
    'family_relationships', 'futoshiki', 'graph_color', 'group_anagrams', 'isomorphic_strings',
    'jugs', 'knights_knaves', 'largest_island', 'maze', 'mini_sudoku', 'number_filtering',
    'number_format', 'number_sorting', 'palindrome_generation', 'propositional_logic',
    'rectangle_count', 'rotate_matrix', 'shortest_path', 'spell_backward', 'spiral_matrix',
    'syllogism', 'time_intervals', 'zebra_puzzles',
)
GYM_VERSION = '0.1.19'
GYM_SEEDS = {'test': 4242, 'dev': 42430021, 'train': 42440021}


def parquet_rows(path, columns=None):
    from pyarrow.parquet import ParquetFile

    for batch in ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def json_rows(path):
    with path.open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, default=str), file=file)


def content_hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fetch(root, dataset, relative, local=None):
    repo, revision = DATASETS[dataset]
    path = root / (local or f'{dataset}-{Path(relative).name}')
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.part')
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}'
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '4',
                        '--retry-all-errors', '--connect-timeout', '20', '-o', str(partial), url], check=True)
        partial.rename(path)
    return path, {'repo': repo, 'revision': revision, 'path': relative, 'sha256': content_hash(path)}


def get_sources(root):
    from concurrent.futures import ThreadPoolExecutor

    root.mkdir(parents=True, exist_ok=True)
    files = [
        ('dapo', 'data/dapo-math-17k.parquet', 'dapo.parquet'),
        *[('ocr', f'split_0/train-{i:05d}-of-00030.parquet', f'train-{i:05d}-of-00030.parquet') for i in range(3)],
        ('aime24', 'test-00000-of-00001.parquet', None),
        ('aime25', 'test.jsonl', None),
        ('minerva', 'test.jsonl', None),
        *[('lcb', f'test{n}.jsonl', None) for n in ('', 2, 3, 4, 5, 6)],
    ]
    with ThreadPoolExecutor(max_workers=4) as workers:
        sources = list(workers.map(lambda args: fetch(root, *args), files))
    return {(path.name if kind in ('ocr', 'lcb') else kind): (path, meta)
            for (kind, _, _), (path, meta) in zip(files, sources, strict=True)}


def math_rows(path):
    for row in parquet_rows(path, ['prompt', 'extra_info']):
        messages = row['prompt']
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Unexpected DAPO chat structure')
        prompt = messages[0]['content']
        parts = prompt.strip().split('\n\n')
        if len(parts) < 3 or 'Answer:' not in parts[-1]:
            raise ValueError('Unexpected DAPO problem wrapper')
        yield {'source_id': row['extra_info']['index'], 'prompt': prompt,
               'problem': '\n\n'.join(parts[1:-1])}


def code_rows(paths):
    for path in paths:
        for row in parquet_rows(path, ['id', 'input', 'split', 'source', 'license']):
            if row['split'] != 'train':
                continue
            yield {'source_id': row['id'], 'prompt': row['input'], 'problem': row['input'],
                   'source_platform': row['source'], 'source_license': row['license'], 'source_file': path.name}


def logic_rows(seed, count):
    import reasoning_gym
    from reasoning_gym.utils import SYSTEM_PROMPTS

    if reasoning_gym.__version__ != GYM_VERSION:
        raise ValueError('Requires reasoning-gym==' + GYM_VERSION)
    datasets = {task: reasoning_gym.create_dataset(task, seed=seed, size=count) for task in LOGIC_TASKS}
    for index in range(count):
        for task, data in datasets.items():
            row = data[index]
            yield {'source_id': f'{task}/{seed}/{index}', 'problem': row['question'],
                   'prompt': SYSTEM_PROMPTS['default'].strip() + '\n\n' + row['question'],
                   'answer': row['answer'], 'metadata': row.get('metadata'), 'task': task,
                   'generator_seed': seed, 'generator_index': index, 'generator_size': count}


def annotate(tokenizer, domain, row, split):
    prefix = render_prefix(tokenizer, row['prompt'])
    return {**row, 'sample_id': f"{domain}/{row['source_id']}", 'domain': domain, 'split': split,
            'prompt_tokens': len(prefix), 'prefix_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}


def protected_rows(sources):
    manifest = read(V5 / 'manifest.json')
    protected = {}
    for domain in ('math', 'code'):
        filename = f'project2_{domain}_dev.jsonl'
        path = V5 / filename
        meta = manifest['files'][filename]
        if sha(path) != (meta['sha256'] if isinstance(meta, dict) else meta):
            raise ValueError('V5 dev digest mismatch')
        protected[domain] = [(r['sample_id'], 'v5_dev', r['problem']) for r in json_rows(path)]

    math500 = Path('data/math/math500_test.jsonl')
    protected['math'].extend((r['sample_id'], 'math500', r['problem']) for r in json_rows(math500))
    for name in ('aime24', 'aime25', 'minerva'):
        path = sources[name][0]
        stream = parquet_rows(path) if path.suffix == '.parquet' else json_rows(path)
        protected['math'].extend((f'{name}/{i}', name, r.get('problem') or r['question']) for i, r in enumerate(stream))

    for name, meta in OFFICIAL_CODE.items():
        path = Path(meta['path'])
        if sha(path) != meta['sha256']:
            raise ValueError(f'Code benchmark {name} digest differs from v5')
        protected['code'].extend((str(r['task_id']), name, r['prompt']) for r in json_rows(path))
    for name, (path, _) in sources.items():
        if name.startswith('lcb-'):
            protected['code'].extend((str(r['question_id']), 'lcb', r['question_content']) for r in json_rows(path))
    return protected


def select_prompts(tokenizer, domain, rows, protected, pool_size, seed):
    # The hash defines a fixed release order. The loader consumes it without shuffling.
    selected, seen, seen_ids, counts = [], set(), set(), Counter()
    normalizer = strip_code_boilerplate if domain == 'code' else str
    exact = {tuple(words(normalizer(text))) for _, _, text in protected}
    # Different LiveCodeBench versions can contain the same question ID; preserve
    # every distinct text without treating a repeated snapshot as a separate item.
    protected_texts = dict.fromkeys((name, normalizer(text)) for _, name, text in protected)
    index = (ProtectedIndex([{'id': str(i), 'set': name, 'text': text}
                             for i, (name, text) in enumerate(protected_texts)])
             if domain != 'logic' else None)
    for row in rows:
        normalized = normalizer(row['problem'])
        key = tuple(words(normalized))
        if not key:
            counts['empty'] += 1
            continue
        if row['source_id'] in seen_ids or key in seen:
            counts['duplicate'] += 1
            continue
        seen.add(key)
        seen_ids.add(row['source_id'])
        if key in exact or (index is not None and index.query(normalized)):
            counts['protected_overlap'] += 1
            continue
        candidate = annotate(tokenizer, domain, row, 'train')
        if candidate['prompt_tokens'] > MAX_PROMPT:
            counts['overlong'] += 1
            continue
        selected.append(candidate)
        if len(selected) == pool_size:
            break
    if len(selected) != pool_size:
        raise ValueError(f'{domain} kept {len(selected)}/{pool_size} unique prompts after filtering: {counts}')
    selected.sort(key=lambda row: hashlib.sha256(f"{seed}/{row['sample_id']}".encode()).digest())
    return selected, counts


def logic_holdout(tokenizer):
    seen = set()
    result = {}
    for split, count in [('test', 25), ('dev', 32)]:
        rows = []
        for row in logic_rows(GYM_SEEDS[split], count):
            key = tuple(words(row['problem']))
            if not key or key in seen:
                continue
            seen.add(key)
            candidate = annotate(tokenizer, 'logic', row, split)
            rows.append(candidate)
        if split == 'dev':
            rows.sort(key=lambda row: hashlib.sha256(row['sample_id'].encode()).digest())
            rows = rows[:200]
        result[split] = rows
    return result


def build(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    output = args.output
    if output.exists():
        raise FileExistsError(output)
    sources = get_sources(args.sources)
    tokenizer_path = snapshot_download(STUDENT['model'], revision=STUDENT['revision'],
                                       allow_patterns=['tokenizer*.json', 'special_tokens_map.json',
                                                       'chat_template.jinja', 'config.json'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    holdout = logic_holdout(tokenizer)
    protected = protected_rows(sources)
    protected['logic'] = [(r['sample_id'], f'logic_{split}', r['problem'])
                          for split, rows in holdout.items() for r in rows]
    streams = {
        'math': math_rows(sources['dapo'][0]),
        'code': code_rows(path for name, (path, _) in sources.items() if name.startswith('train-')),
        'logic': logic_rows(GYM_SEEDS['train'], 1024),
    }
    pools, stats = {}, {}
    for domain in DOMAINS:
        pools[domain], stats[domain] = select_prompts(
            tokenizer, domain, streams[domain], protected[domain], POOL, args.seed,
        )
        print(f'{domain}: {len(pools[domain])} candidate prompts, filtered {dict(stats[domain])}', flush=True)
    schedule = build_schedule(pools)
    output.mkdir(parents=True)
    for domain, rows in pools.items():
        write_rows(output / f'{domain}_train.jsonl',
                   ({**r, 'training_approved': False} for r in rows))
    for split, rows in holdout.items():
        write_rows(output / f'logic_{split}.jsonl', rows)
    write_rows(output / 'schedule.jsonl',
               ({'step': step, 'domain': domain, 'sample_ids': [r['sample_id'] for r in rows]}
                for step, (domain, rows) in enumerate(schedule, 1)))
    write(output / 'protected_ids.json', {'sample_ids': sorted({i for values in protected.values() for i, _, _ in values})})
    doc = {'schema': 'latent-mopd-three-candidate-1', 'student': STUDENT, 'seed': args.seed,
           'source_files': {name: meta for name, (_, meta) in sources.items()},
           'reasoning_gym': {'version': GYM_VERSION, 'tasks': LOGIC_TASKS, 'seeds': GYM_SEEDS},
           'protection_inputs': {path.as_posix(): sha(path) for path in [
               V5 / 'manifest.json', Path('data/math/math500_test.jsonl'),
               *(Path(meta['path']) for meta in OFFICIAL_CODE.values()),
           ]},
           'pool_rows': {d: len(pools[d]) for d in DOMAINS}, 'filtered': stats,
           'protected_sets': Counter(name for rows in protected.values() for _, name, _ in rows),
           'steps': STEPS, 'prompts_per_step': PROMPTS, 'approved_training_files': [],
           'approved_evaluation_files': [],
           'files': {p.name: {'sha256': sha(p)} for p in output.iterdir() if p.is_file()}}
    write(output / 'candidate_manifest.json', doc)
    print(f'{output}/candidate_manifest.json {sha(output / "candidate_manifest.json")}', flush=True)


def approve(args):
    """The project owner records explicit approval of one immutable candidate."""
    from datetime import datetime, timezone

    candidate = args.candidate
    path = candidate / 'candidate_manifest.json'
    if sha(path) != args.candidate_sha256:
        raise ValueError('Candidate manifest does not match the digest reviewed by the owner')
    doc = read(path)
    if doc['schema'] != 'latent-mopd-three-candidate-1' or doc['approved_training_files']:
        raise ValueError('Invalid candidate manifest')
    for name, meta in doc['files'].items():
        if Path(name).name != name or sha(candidate / name) != meta['sha256']:
            raise ValueError('Changed/invalid candidate file ' + name)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    training = [f'{d}_train.jsonl' for d in DOMAINS]
    evaluation = ['logic_dev.jsonl', 'logic_test.jsonl']
    for name in training:
        write_rows(args.output / name, ({**row, 'training_approved': True} for row in json_rows(candidate / name)))
    for name in [*evaluation, 'protected_ids.json', 'schedule.jsonl']:
        shutil.copyfile(candidate / name, args.output / name)
    release = {**doc, 'schema': 'latent-mopd-three-release-1', 'candidate_sha256': args.candidate_sha256,
               'protected_dev_ids_file': 'protected_ids.json', 'approved_training_files': training,
               'approved_evaluation_files': evaluation,
               'files': {p.name: {'sha256': sha(p)} for p in args.output.iterdir() if p.is_file()},
               'approval': {'approver': args.approver, 'note': args.note,
                            'timestamp_utc': datetime.now(timezone.utc).isoformat()}}
    write(args.output / 'manifest.json', release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--sources', type=Path, default=ROOT / 'sources')
    builder.add_argument('--output', type=Path, default=ROOT / 'candidate')
    builder.add_argument('--seed', type=int, default=42)
    approval = commands.add_parser('approve')
    approval.add_argument('--candidate', type=Path, default=ROOT / 'candidate')
    approval.add_argument('--candidate-sha256', required=True)
    approval.add_argument('--output', type=Path, default=ROOT / 'release')
    approval.add_argument('--approver', required=True)
    approval.add_argument('--note', required=True)
    args = parser.parse_args()
    (build if args.command == 'build' else approve)(args)


if __name__ == '__main__':
    main()
