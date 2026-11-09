#!/usr/bin/env python3
"""Score the code candidates of a GPU evaluation in the local Docker sandbox (WSL2), then optionally publish.

  python3 scripts/score_code_local_v3.py session2/20261109T104721Z/runs/eval/d0 \
      --evalplus-image sha256:... --publish --credentials ~/.config/opd/gh_credentials

Each argument names a folder of <suite>.candidates.jsonl files written by scripts/generate_eval_v3.py on the
GPU: a path on the results branch (read with git, after `git fetch origin results`), or a local folder.
  * code_dev: internal dev, 200 tasks, reviewed comparators (scripts/evaluate_code_v2.py, sandbox image from
    reports/code_v2_sandbox_environment.json unless --code-image is given);
  * humanevalplus / mbppplus: official EvalPlus v0.3.1 (scripts/evaluate_evalplus_v2.py); needs an image built
    from scripts/docker/evalplus-0.3.1.Dockerfile (--evalplus-image, $OPD_EVALPLUS_IMAGE, or the ID saved in
    ~/.config/opd/evalplus_image). Without one, these two suites are skipped and reported as such.
Candidate code runs only inside Docker (no network, read-only, unprivileged). A missing Docker daemon or image
stops the run; nothing falls back to executing on the host. Results go to <scores-root>/<task>-<stamp>/<model>/
(default ~/opd_scores), with scores.json summarizing every suite. --publish pushes that folder (text files) to
the results branch as code_scores/<task>-<stamp>-<model>/. Internal dev scores and official EvalPlus scores
are reported separately and are never merged. Run from any directory; Python 3.9+, stdlib only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DEV = 'data/project2_release_v5_20261108/release/project2_code_dev.jsonl'
REGISTRY = 'reports/handover_20261024_v3/reviewed_comparators.json'
OFFICIAL = {'humanevalplus': ('humaneval', 'data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl'),
            'mbppplus': ('mbpp', 'data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl')}
SUITES = ('code_dev', 'humanevalplus', 'mbppplus')
SAFE = re.compile(r'[^A-Za-z0-9_.-]')


class ScoreError(Exception):
    pass


def say(text):
    print(f'[score] {text}', flush=True)


def run(cmd, log=None, timeout=None, check=False):
    result = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=timeout)
    if log:
        Path(log).write_text(result.stdout, encoding='utf-8')
    if check and result.returncode:
        raise ScoreError(f'{" ".join(cmd[:4])} ... failed (exit {result.returncode}): {result.stdout.strip()[-800:]}')
    return result


def docker_ready(images):
    docker = shutil.which('docker')
    if not docker:
        raise ScoreError('docker is not on PATH (WSL2: install Docker Engine, or enable WSL integration in Docker Desktop)')
    info = run([docker, 'info', '--format', '{{.ServerVersion}}'], timeout=60)
    if info.returncode:
        raise ScoreError('the Docker daemon is not reachable: ' + info.stdout.strip()[-400:])
    for label, image in images.items():
        if image and run([docker, 'image', 'inspect', image], timeout=60).returncode:
            raise ScoreError(f'{label} image {image} is not present locally (images are never pulled implicitly); '
                             'see docs/handover/kiro_ide_onboarding.md')
    return info.stdout.strip()


def historical_code_image():
    return json.loads((ROOT / 'reports/code_v2_sandbox_environment.json').read_text(encoding='utf-8'))['image_id']


def image_from_file(name):
    """One-line image ID saved in ~/.config/opd/<name>, or None."""
    path = Path(os.path.expanduser('~/.config/opd')) / name
    if path.is_file():
        text = path.read_text(encoding='utf-8').strip()
        return text or None
    return None


def naming(source):
    """(group, model): '<task>-<stamp>' for a results-branch path such as session2/<stamp>/runs/eval/d0,
    'local' for a folder on disk."""
    path = Path(os.path.expanduser(source))
    parts = [p for p in path.parts if p not in ('', '/')]
    model = SAFE.sub('_', parts[-1])
    if path.is_dir() or len(parts) < 3:
        return 'local', model
    return SAFE.sub('_', '-'.join(parts[:2])), model


def fetch_candidates(source, ref, target):
    """Copy <suite>.candidates.jsonl from a local folder or from the results branch; returns the suites found."""
    target.mkdir(parents=True, exist_ok=True)
    local = Path(os.path.expanduser(source))
    found = []
    for suite in SUITES:
        name = f'{suite}.candidates.jsonl'
        if local.is_dir():
            if (local / name).is_file():
                shutil.copyfile(local / name, target / name)
                found.append(suite)
            continue
        blob = f'{ref}:{source.strip("/")}/{name}'
        if run(['git', 'cat-file', '-e', blob]).returncode == 0:
            data = subprocess.run(['git', 'show', blob], cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout
            (target / name).write_bytes(data)
            found.append(suite)
    for name in ('math_dev.summary.json', 'code_dev.summary.json'):  # generation identity, if present
        if not local.is_dir():
            blob = f'{ref}:{source.strip("/")}/{name}'
            if run(['git', 'cat-file', '-e', blob]).returncode == 0:
                (target / name).write_bytes(subprocess.run(['git', 'show', blob], cwd=ROOT, stdout=subprocess.PIPE,
                                                           check=True).stdout)
    return found


def score_code_dev(candidates, out, image):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'scripts.evaluate_code_v2', '--mode', 'student', '--data', CODE_DEV,
           '--candidates', str(candidates), '--registry', REGISTRY, '--output-dir', str(out)]
    if image:
        cmd += ['--image-id', image]
    result = run(cmd, log=out / 'evaluate.log', timeout=4 * 3600)
    report_path = out / 'student.json'
    if result.returncode or not report_path.is_file():
        return {'status': 'failed', 'exit_code': result.returncode, 'log': 'code_dev/evaluate.log'}
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if 'results' not in report:
        return {'status': report.get('status'), 'category': report.get('category'), 'error': report.get('error')}
    rows = {json.loads(l)['sample_id']: json.loads(l) for l in (ROOT / CODE_DEV).read_text(encoding='utf-8').splitlines()
            if l.strip()}
    statuses, by_interface = {}, {}
    for r in report['results']:
        statuses[r['status']] = statuses.get(r['status'], 0) + 1
        interface = rows[r['sample_id']]['interface']
        cell = by_interface.setdefault(interface, {'tasks': 0, 'passed': 0, 'blocked': 0})
        cell['tasks'] += 1
        cell['passed'] += r['status'] == 'passed'
        cell['blocked'] += r['status'] == 'blocked_after_review'
    total, passed = len(report['results']), statuses.get('passed', 0)
    blocked = statuses.get('blocked_after_review', 0)
    return {'status': 'scored', 'protocol': report.get('protocol'), 'tasks': total, 'passed': passed,
            'blocked_after_review': blocked, 'pass_rate': passed / total if total else None,
            'pass_rate_excluding_blocked': passed / (total - blocked) if total > blocked else None,
            'by_interface': by_interface, 'status_counts': statuses, 'image_id': report.get('image_id'),
            'note': 'internal dev with reviewed comparators; not official EvalPlus'}


def score_evalplus(suite, candidates, out, image):
    dataset, official = OFFICIAL[suite]
    if not (ROOT / official).is_file():
        return {'status': 'skipped', 'reason': f'{official} missing; run python3 -m scripts.restore_official_benchmarks'}
    if out.exists():
        shutil.rmtree(out)  # samples.jsonl is opened with mode "x"; every scoring starts from an empty folder
    out.mkdir(parents=True)
    out.chmod(0o777)  # the container runs as uid 65534 and writes its results into this bind-mounted folder
    cmd = [sys.executable, '-m', 'scripts.evaluate_evalplus_v2', '--dataset', dataset, '--predictions', str(candidates),
           '--output-dir', str(out), '--image-id', image, '--execute-docker']
    result = run(cmd, log=out / 'evaluate.log', timeout=4 * 3600)
    protocol = out / 'official_protocol.json'
    summary = {'status': 'failed' if result.returncode else 'executed', 'exit_code': result.returncode,
               'image_id': image, 'log': f'{suite}/evaluate.log'}
    if protocol.is_file():
        identity = json.loads(protocol.read_text(encoding='utf-8'))
        summary.update(status=identity.get('status'), evalplus_exit_code=identity.get('exit_code'),
                       dataset_sha256=identity.get('dataset_sha256'))
    found = sorted(out.glob('*eval_results.json'))
    if found:
        evaluated = json.loads(found[0].read_text(encoding='utf-8')).get('eval', {})
        tasks = len(evaluated)
        base = sum(1 for runs in evaluated.values() if runs and runs[0].get('base_status') == 'pass')
        plus = sum(1 for runs in evaluated.values()
                   if runs and runs[0].get('base_status') == 'pass' and runs[0].get('plus_status') == 'pass')
        summary.update(results_file=f'{suite}/{found[0].name}', tasks=tasks,
                       pass_at_1_base=base / tasks if tasks else None, pass_at_1_plus=plus / tasks if tasks else None)
    elif summary['status'] == 'executed':
        summary['status'] = 'executed_without_results_file'
    return summary


def publish(out, group, model, credentials):
    cmd = [sys.executable, str(ROOT / 'scripts/publish_results_v3.py'), '--name', 'code_scores',
           '--stamp', f'{group}-{model}', '--source', f'scores={out}']
    if credentials:
        cmd += ['--credentials', os.path.expanduser(credentials)]
    result = run(cmd, timeout=900)
    print(result.stdout.strip(), flush=True)
    return result.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('sources', nargs='+', metavar='RESULTS_PATH', help='results-branch folder or local folder')
    p.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    p.add_argument('--code-image', default=os.environ.get('OPD_CODE_IMAGE') or image_from_file('code_image'),
                   help='internal sandbox image ID; default: $OPD_CODE_IMAGE, ~/.config/opd/code_image, '
                        'else the recorded one')
    p.add_argument('--evalplus-image', default=os.environ.get('OPD_EVALPLUS_IMAGE') or image_from_file('evalplus_image'),
                   help='EvalPlus v0.3.1 image ID (sha256:...); default: $OPD_EVALPLUS_IMAGE or ~/.config/opd/evalplus_image')
    p.add_argument('--scores-root', default=os.environ.get('OPD_SCORES_ROOT', '~/opd_scores'))
    p.add_argument('--ref', default='origin/results', help='git ref of the results branch')
    p.add_argument('--no-fetch', action='store_true', help='do not run git fetch origin results first')
    p.add_argument('--force', action='store_true', help='score again even if scores.json exists')
    p.add_argument('--publish', action='store_true', help='push scores to the results branch (code_scores/...)')
    p.add_argument('--credentials', help='git credential-store file with a token that can push')
    a = p.parse_args(argv)
    for image in (a.code_image, a.evalplus_image):
        if image and not image.startswith('sha256:'):
            p.error(f'image IDs look like sha256:..., got {image!r}')
    wants_official = [s for s in a.suites if s in OFFICIAL]
    if wants_official and not a.evalplus_image:
        say('no --evalplus-image: humanevalplus and mbppplus are skipped (build scripts/docker/evalplus-0.3.1.Dockerfile)')
    images = {'code sandbox': a.code_image or historical_code_image() if 'code_dev' in a.suites else None,
              'EvalPlus': a.evalplus_image if wants_official else None}
    try:
        say(f'Docker server {docker_ready(images)}; images present')
        if not a.no_fetch and any(not Path(os.path.expanduser(s)).is_dir() for s in a.sources):
            run(['git', 'fetch', '-q', 'origin', 'results'], check=True, timeout=600)
    except (ScoreError, subprocess.TimeoutExpired) as error:
        say(f'FAILED: {error}')
        return 2
    root, status = Path(os.path.expanduser(a.scores_root)), 0
    for source in a.sources:
        group, model = naming(source)
        out = root / group / model
        if (out / 'scores.json').is_file() and not a.force:
            say(f'{source}: already scored in {out} (use --force to score again)')
            continue
        found = fetch_candidates(source, a.ref, out / 'candidates')
        if not found:
            say(f'{source}: no candidate files found (is the GPU evaluation finished and published?)')
            status = 1
            continue
        scores = {'source': source, 'ref': a.ref, 'scored_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'suites': {}}
        for suite in a.suites:
            candidates = out / 'candidates' / f'{suite}.candidates.jsonl'
            if suite not in found:
                scores['suites'][suite] = {'status': 'no candidates'}
            elif suite == 'code_dev':
                say(f'{model}: code_dev ...')
                scores['suites'][suite] = score_code_dev(candidates, out / 'code_dev', a.code_image)
            elif not a.evalplus_image:
                scores['suites'][suite] = {'status': 'skipped', 'reason': 'no EvalPlus image'}
            else:
                say(f'{model}: {suite} ...')
                scores['suites'][suite] = score_evalplus(suite, candidates, out / suite, a.evalplus_image)
            say(f'{model}: {suite}: {json.dumps(scores["suites"][suite], ensure_ascii=False)[:300]}')
            if scores['suites'][suite].get('status') == 'failed':
                status = 1
        (out / 'scores.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        say(f'{model}: wrote {out / "scores.json"}')
        if a.publish and not publish(out, group, model, a.credentials):
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
