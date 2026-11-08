"""Owner approval of the v5 candidate (explicit, recorded, never overwrites).

AGENTS.md: a candidate is not approval and AI review is not human gold.
This command is the recorded owner decision. It binds to the candidate and
audit-report digests, keeps every retained v4 line byte-identical, and flips
only the added rows to training_approved=true.

  python -m scripts.approve_project2_release_v5 --approver "<your name>" \
      --note "Read reports/project2_release_v5_20261108/README.md and audit.json"
"""
import argparse
import json

from src.data.release_v5 import approve_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approver', required=True)
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    print(json.dumps(approve_candidate('.', args.approver, args.note), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
