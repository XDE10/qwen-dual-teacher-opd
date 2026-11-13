"""Download the pinned student/teacher snapshots of an opd-v3 config into the HF cache.

Run once per Colab session BEFORE training, so download time is not counted as
teacher-load time, and a missing file fails before any GPU work. Only weights,
configs and tokenizer files are fetched; remote code is never executed.

  python -m scripts.prefetch_models_v3 --config configs/project2_opd_v3_d4_dual.json
"""
import argparse
import json
import time

from huggingface_hub import snapshot_download

ALLOW = ['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'tokenizer*', '*.jinja']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, nargs='+')
    args = parser.parse_args()
    specs = {}
    for path in args.config:
        config = json.loads(open(path, encoding='utf-8').read())
        for spec in [config['student'], *config['teachers'].values()]:
            specs[(spec['model'], spec['revision'])] = spec
    for (model, revision) in sorted(specs):
        if revision == 'main' or len(revision) != 40:
            raise ValueError('Pin a full commit revision for ' + model)
        start = time.monotonic()
        path = snapshot_download(model, revision=revision, allow_patterns=ALLOW, max_workers=8)
        print(json.dumps({'model': model, 'revision': revision, 'path': path, 'seconds': round(time.monotonic() - start, 1)}),
              flush=True)


if __name__ == '__main__':
    main()
