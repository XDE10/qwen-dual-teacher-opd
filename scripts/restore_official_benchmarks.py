"""Download and verify the pinned EvalPlus files used in overlap checks."""
import argparse
import gzip
import hashlib
import time
import urllib.request
from pathlib import Path

from src.evaluation.benchmark_inputs import OFFICIAL_CODE

URLS = {'humanevalplus': 'https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz',
        'mbppplus': 'https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz'}
SIZES = {'humanevalplus': 7714666, 'mbppplus': 2592369}


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', help='directory with HumanEvalPlus/MbppPlus .jsonl or .jsonl.gz copies')
    args = parser.parse_args()
    for name, meta in OFFICIAL_CODE.items():
        target = Path(meta['path'])
        size, digest = SIZES[name], meta['sha256']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f'{target}: already present and verified')
            continue
        if args.source_dir:
            candidates = sorted(Path(args.source_dir).glob(target.stem.split('-')[0] + '*'))
            if not candidates:
                raise FileNotFoundError(f'No copy of {target.name} in {args.source_dir}')
            raw = candidates[0].read_bytes()
            origin = candidates[0].as_posix()
        else:
            raw, origin = fetch(URLS[name]), URLS[name]
        data = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
        actual = hashlib.sha256(data).hexdigest()
        if len(data) != size or actual != digest:
            raise ValueError(f'{origin}: got {len(data)} bytes / {actual}, expected {size} / {digest}; not written')
        rows = sum(1 for line in data.splitlines() if line.strip())
        if rows != meta['rows']:
            raise ValueError(f'{origin}: {rows} rows, expected {meta["rows"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f'{target}: restored from {origin} ({size} bytes, sha256 {digest}, {rows} rows)')


if __name__ == '__main__':
    main()
