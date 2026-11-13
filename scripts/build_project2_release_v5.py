"""Build the v5 CANDIDATE: audit v4, drop overlap/budget failures, refill math from sealed pools.

Does not approve anything. Run from the repository root after
``python -m scripts.restore_official_benchmarks``:

  python -m scripts.build_project2_release_v5                 # same size as v4
  python -m scripts.build_project2_release_v5 --math-target 4000   # optional growth

Then read reports/project2_release_v5_20261108/README.md and approve explicitly
with scripts/approve_project2_release_v5.py.
"""
import argparse
import json

from src.data.release_v5 import build_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--math-target', type=int, help='grow math train to this many rows (default: keep v4 size)')
    args = parser.parse_args()
    report = build_candidate('.', math_target=args.math_target)
    print(json.dumps({'counts': report['counts'], 'decontamination': report['decontamination'],
                      'refill': report['refill']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
