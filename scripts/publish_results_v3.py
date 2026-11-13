#!/usr/bin/env python3
"""Push the small text results of a run (logs, JSON, JSONL) to the repository's results branch.

On the Colab VM, scripts/colab_job_v3.py --publish calls it when a job ends. It also runs
anywhere with git and push access, for example in WSL2 for results that were downloaded:

  python3 scripts/publish_results_v3.py --name first_session \
      --source report=/home/me/opd_first_session.tar.gz --credentials ~/.config/opd/gh_credentials

Each --source is a folder, a file or a .tar.gz archive, optionally written LABEL=PATH. Files go
to <name>/<stamp>/<label>/ on the results branch, next to PUBLISHED.json, which lists every
published file (bytes, sha256) and every file left out, with the reason. Only text files
with an allowed suffix and at most --max-file-mb are published. Checkpoint folders, weights,
archives and hidden files are skipped. Strings that look like GitHub, Hugging Face or Google
credentials, and passwords in URLs, are replaced with [REDACTED]. An archive with a single
top-level folder is unpacked straight into its label folder. Publishing again with the same
name and stamp replaces that folder; a running job updates its snapshot this way. Only
branches named results or results-* can be written.

Credentials: --credentials FILE (git credential-store format, https://USER:TOKEN@github.com),
else $OPD_GIT_CREDENTIALS, else /content/gh_credentials if it exists, else your normal git
setup. With a credentials file the push goes to https://github.com/OWNER/REPO.git of the
origin remote; --remote or $OPD_RESULTS_REMOTE sets another remote. --check only tests that
the remote accepts a push (git push --dry-run). Stdlib only, Python 3.8+.
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VM_CREDENTIALS = Path('/content/gh_credentials')
TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.tsv', '.yaml', '.yml', '.out', '.err'}
SKIP_FOLDER = re.compile(r'checkpoint-.*|adapter-step-.*|final_adapter|__pycache__')
SAFE_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')
RESULTS_BRANCH = re.compile(r'results(-[A-Za-z0-9_.-]+)?')
GITHUB_REPO = re.compile(r'github\.com[:/]+([^/\s:]+)/([^/\s]+?)(?:\.git)?/*$')
RETRYABLE_PUSH = re.compile(r'\(fetch first\)|\(non-fast-forward\)|failed to update ref|cannot lock ref')
ARCHIVE_SUFFIX = re.compile(r'(\.tar\.gz|\.tgz|\.tar)$')
SECRETS = [
    (re.compile(r'github_pat_[A-Za-z0-9_]{20,}'), 'github_pat_[REDACTED]'),
    (re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}'), 'gh_[REDACTED]'),
    (re.compile(r'\bhf_[A-Za-z0-9]{20,}'), 'hf_[REDACTED]'),
    (re.compile(r'\bya29\.[0-9A-Za-z_-]{20,}'), 'ya29.[REDACTED]'),
    (re.compile(r'\b1//[0-9A-Za-z_-]{20,}'), '1//[REDACTED]'),
    (re.compile(r'(\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)[^\s/@]+@'), r'\1[REDACTED]@'),
    (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S),
     '[REDACTED PRIVATE KEY]'),
]
MAX_SKIPPED_LISTED = 500
README = """# results 分支

这个分支只放运行结果里的小型文本文件（日志、JSON、JSONL），由 `scripts/publish_results_v3.py`
写入。代码在其他分支；不要把这个分支合并进任何代码分支。

目录是 `<任务名>/<UTC 时间>/`：

- `PUBLISHED.json`：发布时间；任务状态和退出码；产生结果的代码提交；GPU；已发布文件的字节数和 sha256；
  没发布的文件和原因。
- `job/`：`scripts/colab_job_v3.py` 的任务日志 `output.log` 和命令 `command.json`。
- 其他子目录：任务的 `--save` 内容，例如第一次上卡的 `report/`。

权重、checkpoint、压缩包和隐藏文件不会发布。看起来像 GitHub、Hugging Face、Google 凭据的字符串和
URL 里的密码会被替换成 `[REDACTED]`。完整产物在 Google Drive 的 `opd_runs/<任务名>/<UTC 时间>/`。
"""


class PublishError(Exception):
    pass


def log(text):
    print(f'[publish] {text}', flush=True)


def scrub(text):
    for pattern, replacement in SECRETS:
        text = pattern.sub(replacement, text)
    return text


def redact(data):
    """(bytes to publish, number of replaced strings). Unchanged bytes when nothing matched."""
    text = data.decode('utf-8', errors='replace')
    count = 0
    for pattern, replacement in SECRETS:
        text, n = pattern.subn(replacement, text)
        count += n
    return (text.encode('utf-8') if count else data), count


def parse_source(text):
    label, sep, rest = text.partition('=')
    if not (sep and SAFE_NAME.fullmatch(label) and rest):
        label, rest = None, text
    path = Path(os.path.expanduser(rest))
    return label, (path if path.is_absolute() else Path.cwd() / path)


def skip_reason(parts, size, max_file):
    """Why the file at relative path ``parts`` is not published, or None."""
    if any(p.startswith('.') for p in parts):
        return 'hidden'
    if any(SKIP_FOLDER.fullmatch(p) for p in parts[:-1]):
        return 'inside a checkpoint/adapter folder'
    if PurePosixPath(parts[-1]).suffix.lower() not in TEXT_SUFFIXES:
        return 'not a text result'
    if size > max_file:
        return f'larger than {max_file / 2 ** 20:g} MB'
    return None


def archive_parts(name):
    parts = [p for p in PurePosixPath(name).parts if p not in ('', '.')]
    if name.startswith('/') or '..' in parts:
        return None
    return tuple(parts)


def source_kind(path):
    if not path.exists():
        return 'missing'
    if path.is_dir():
        return 'folder'
    if path.is_file():
        return 'archive' if ARCHIVE_SUFFIX.search(path.name) else 'file'
    return 'other'


def candidates(label, path, max_file, skipped, archives):
    """Publishable files of one source: (destination parts, size, read function), plus source info.

    A folder or archive without a label is published under its own name; a single file
    without a label goes to the top of the task folder.
    """
    kind = source_kind(path)
    if label is None and kind in ('folder', 'archive'):
        label = re.sub(r'[^A-Za-z0-9_.-]', '_', ARCHIVE_SUFFIX.sub('', path.name)).lstrip('.-') or kind
    prefix = (label,) if label else ()
    info = {'label': label, 'path': str(path), 'kind': kind}
    found = []

    def consider(parts, size, reader):
        reason = skip_reason(parts, size, max_file)
        if reason:
            skipped.append({'path': '/'.join(prefix + parts), 'reason': reason})
        else:
            found.append((prefix + parts, size, reader))

    if kind == 'missing':
        skipped.append({'path': '/'.join(prefix) or path.name, 'reason': f'source not found: {path}'})
    elif kind == 'folder':
        for dirpath, dirnames, filenames in os.walk(path):
            here = Path(dirpath)
            rel = here.relative_to(path).parts
            keep = []
            for d in sorted(dirnames):
                if (here / d).is_symlink() or d.startswith('.') or SKIP_FOLDER.fullmatch(d):
                    skipped.append({'path': '/'.join(prefix + rel + (d,)) + '/', 'reason': 'folder not published'})
                else:
                    keep.append(d)
            dirnames[:] = keep
            for f in sorted(filenames):
                full = here / f
                if full.is_symlink() or not full.is_file():
                    skipped.append({'path': '/'.join(prefix + rel + (f,)), 'reason': 'not a regular file'})
                    continue
                consider(rel + (f,), full.stat().st_size, full.read_bytes)
    elif kind == 'archive':
        try:
            tar = archives.enter_context(tarfile.open(path, 'r:*'))
        except (tarfile.TarError, OSError) as error:
            skipped.append({'path': '/'.join(prefix), 'reason': f'cannot read archive: {error}'})
            return found, info
        members = [(m, archive_parts(m.name)) for m in tar.getmembers()]
        file_parts = [parts for m, parts in members if parts and m.isfile()]
        tops = {parts[0] for m, parts in members if parts}
        strip = len(tops) == 1 and all(len(parts) > 1 for parts in file_parts)
        if strip:
            info['archive_root'] = next(iter(tops))
        for member, parts in members:
            if parts is None:
                skipped.append({'path': member.name, 'reason': 'unsafe path in archive'})
            elif member.isdir() or not parts:
                continue
            elif not member.isfile():
                skipped.append({'path': '/'.join(prefix + parts), 'reason': 'not a regular file'})
            else:
                consider(parts[1:] if strip else parts, member.size,
                         lambda member=member: tar.extractfile(member).read())
    elif kind == 'file':
        consider((path.name,), path.stat().st_size, path.read_bytes)
    else:
        skipped.append({'path': '/'.join(prefix) or path.name, 'reason': 'not a file, folder or archive'})
    return found, info


def stage(sources, out, max_file, max_total):
    """Write the publishable files under ``out``; returns (files, skipped, source infos, redactions)."""
    skipped, infos, found, seen = [], [], [], set()
    with contextlib.ExitStack() as archives:
        for label, path in sources:
            entries, info = candidates(label, path, max_file, skipped, archives)
            infos.append(info)
            found.extend(entries)
        files, total, redactions = [], 0, 0
        for parts, size, reader in sorted(found, key=lambda e: (e[1], e[0])):
            rel = '/'.join(parts)
            if rel in seen or rel == 'PUBLISHED.json':
                skipped.append({'path': rel, 'reason': 'duplicate destination'})
                continue
            if total + size > max_total:
                skipped.append({'path': rel, 'reason': f'over the {max_total / 2 ** 20:g} MB total'})
                continue
            data = reader()
            if b'\x00' in data[:8192]:
                skipped.append({'path': rel, 'reason': 'binary content'})
                continue
            data, count = redact(data)
            target = out.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            seen.add(rel)
            total += len(data)
            redactions += count
            files.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                          **({'redacted_strings': count} if count else {})})
    return sorted(files, key=lambda f: f['path']), skipped, infos, redactions


class Git:
    """git with a fixed credential helper and identity, no prompts, and timeouts."""

    def __init__(self, credentials, name, email):
        self.config = []
        if credentials:
            self.config += ['-c', 'credential.helper=',
                            '-c', 'credential.helper=store --file=' + shlex.quote(str(credentials))]
        self.config += ['-c', 'user.name=' + name, '-c', 'user.email=' + email]

    def __call__(self, *args, cwd=None, check=True, timeout=600):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
        try:
            result = subprocess.run(['git', *self.config, *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                    timeout=timeout)
        except subprocess.TimeoutExpired:
            raise PublishError(f'git {args[0]} timed out after {timeout} s')
        if check and result.returncode:
            raise PublishError(f'git {args[0]} failed: ' + scrub((result.stderr or result.stdout).strip()[-1500:]))
        return result


def git_output(*args):
    try:
        result = subprocess.run(['git', *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                universal_newlines=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def identity():
    return (os.environ.get('OPD_GIT_NAME') or git_output('config', '--get', 'user.name') or 'opd-results',
            os.environ.get('OPD_GIT_EMAIL') or git_output('config', '--get', 'user.email')
            or 'opd-results@users.noreply.github.com')


def resolve_credentials(explicit):
    value = explicit or os.environ.get('OPD_GIT_CREDENTIALS')
    if value:
        path = Path(os.path.expanduser(value))
        if not path.is_file():
            raise PublishError(f'credentials file not found: {path}')
        return path
    return VM_CREDENTIALS if VM_CREDENTIALS.is_file() else None


def resolve_remote(explicit, credentials):
    if explicit:
        return explicit
    origin = git_output('-C', str(ROOT), 'remote', 'get-url', 'origin')
    if not origin:
        raise PublishError(f'no origin remote in {ROOT}; pass --remote')
    match = GITHUB_REPO.search(origin)
    if credentials and match:
        return f'https://github.com/{match.group(1)}/{match.group(2)}.git'
    return origin


def public(url):
    return re.sub(r'//[^/@\s]+@', '//', url)


def web_url(url, branch, folder):
    match = GITHUB_REPO.search(url)
    return f'https://github.com/{match.group(1)}/{match.group(2)}/tree/{branch}/{folder}' if match else None


def check_write(git, url):
    """Read the remote, then dry-run a push of a throwaway commit to a branch that is never created."""
    result = git('ls-remote', '--heads', url, check=False, timeout=120)
    if result.returncode:
        raise PublishError(f'cannot read {public(url)}: ' + scrub(result.stderr.strip()[-800:]))
    with tempfile.TemporaryDirectory(prefix='opd-publish-check-') as tmp:
        git('init', '--quiet', tmp)
        git('-c', 'commit.gpgsign=false', 'commit', '--quiet', '--allow-empty', '-m', 'write check', cwd=tmp)
        result = git('push', '--dry-run', url, 'HEAD:refs/heads/results-write-check', cwd=tmp, check=False,
                     timeout=120)
    if result.returncode:
        raise PublishError(f'{public(url)} does not accept pushes with these credentials: '
                           + scrub(result.stderr.strip()[-800:]))


def only_publish_time_changed(git, worktree, folder):
    """True when the staged change is nothing, or only the publish time in PUBLISHED.json."""
    changed = git('diff', '--cached', '--name-only', cwd=worktree).stdout.split()
    meta = f'{folder}/PUBLISHED.json'
    if not changed:
        return True
    if changed != [meta]:
        return False
    old = git('show', f'HEAD:{meta}', cwd=worktree, check=False)
    if old.returncode:
        return False
    try:
        before, after = json.loads(old.stdout), json.loads((worktree / meta).read_text(encoding='utf-8'))
    except ValueError:
        return False
    before.pop('published_utc', None)
    after.pop('published_utc', None)
    return before == after


def push(git, url, branch, staged, folder, message, attempts=5):
    """Commit ``staged/folder`` as ``folder`` on ``branch``; on a concurrent update, redo it on the new tip."""
    with tempfile.TemporaryDirectory(prefix='opd-publish-') as tmp:
        worktree = Path(tmp) / 'results'
        heads = git('ls-remote', '--heads', url, branch, check=False, timeout=120)
        if heads.returncode:
            raise PublishError(f'cannot read {public(url)}: ' + scrub(heads.stderr.strip()[-800:]))
        if heads.stdout.strip():
            git('clone', '--quiet', '--depth', '1', '--single-branch', '--branch', branch, url, str(worktree))
        else:
            log(f'branch {branch} does not exist yet; creating it')
            git('init', '--quiet', str(worktree))
            git('symbolic-ref', 'HEAD', 'refs/heads/' + branch, cwd=worktree)
            git('remote', 'add', 'origin', url, cwd=worktree)
            (worktree / 'README.md').write_text(README, encoding='utf-8')
            git('add', '-f', 'README.md', cwd=worktree)
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                git('fetch', '--quiet', '--depth', '1', 'origin', branch, cwd=worktree)
                git('reset', '--quiet', '--hard', 'FETCH_HEAD', cwd=worktree)
            target = worktree / folder
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(str(staged / folder), str(target))
            git('add', '-A', '-f', '--', folder, cwd=worktree)
            if only_publish_time_changed(git, worktree, folder):
                return 'unchanged'
            git('commit', '--quiet', '-m', message, cwd=worktree)
            result = git('push', '--quiet', 'origin', 'HEAD:refs/heads/' + branch, cwd=worktree, check=False)
            if result.returncode == 0:
                return 'pushed'
            if attempt < attempts and RETRYABLE_PUSH.search(result.stderr):
                log(f'{branch} moved while publishing (attempt {attempt}/{attempts}); redoing it on the new tip')
                time.sleep(2 * attempt)
                continue
            raise PublishError('git push failed: ' + scrub(result.stderr.strip()[-1500:]))
    raise PublishError('git push kept being rejected')


def gpu_names():
    if not shutil.which('nvidia-smi'):
        return None
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] or None


def code_state():
    status = git_output('-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no')
    return {'commit': git_output('-C', str(ROOT), 'rev-parse', 'HEAD'),
            'branch': git_output('-C', str(ROOT), 'branch', '--show-current') or None,
            'modified_tracked_files': None if status is None else bool(status)}


def parse_args(argv):
    p = argparse.ArgumentParser(description='Push small text results of a run to the results branch.')
    p.add_argument('--name', help='task name, e.g. first_session or d4_dual')
    p.add_argument('--source', action='append', default=[], metavar='[LABEL=]PATH',
                   help='folder, file or .tar.gz to publish (repeatable)')
    p.add_argument('--stamp', help='folder under <name>/ (default: current UTC time)')
    state = p.add_mutually_exclusive_group()
    state.add_argument('--status', type=int, help='exit status of the finished job')
    state.add_argument('--running', action='store_true', help='snapshot of a job that is still running')
    p.add_argument('--branch', default='results')
    p.add_argument('--credentials', help='git credential-store file with a token that can push')
    p.add_argument('--remote', default=os.environ.get('OPD_RESULTS_REMOTE'), help='default: origin (see above)')
    p.add_argument('--max-file-mb', type=float, default=20)
    p.add_argument('--max-total-mb', type=float, default=50)
    p.add_argument('--message', help='commit message')
    p.add_argument('--dry-run', action='store_true', help='show what would be published; no git')
    p.add_argument('--check', action='store_true', help='only test that the remote accepts a push')
    a = p.parse_args(argv)
    if not RESULTS_BRANCH.fullmatch(a.branch):
        p.error('--branch must be results or results-<suffix>')
    if not a.check:
        if not a.name or not SAFE_NAME.fullmatch(a.name):
            p.error('--name is required: letters, digits, "_", "-" and "."')
        if a.stamp is not None and not SAFE_NAME.fullmatch(a.stamp):
            p.error('--stamp: letters, digits, "_", "-" and "."')
        if not a.source:
            p.error('give at least one --source')
    return a


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        credentials = resolve_credentials(a.credentials)
        url = None if a.dry_run else resolve_remote(a.remote, credentials)
        git = Git(credentials, *identity())
        if a.check:
            check_write(git, url)
            log(f'{public(url)} accepts pushes')
            return 0
        stamp = a.stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        folder = f'{a.name}/{stamp}'
        with tempfile.TemporaryDirectory(prefix='opd-publish-stage-') as tmp:
            staged = Path(tmp)
            sources = [parse_source(s) for s in a.source]
            files, skipped, infos, redactions = stage(sources, staged / folder, a.max_file_mb * 2 ** 20,
                                                      a.max_total_mb * 2 ** 20)
            job = ({'state': 'finished', 'exit_status': a.status} if a.status is not None else
                   {'state': 'running'} if a.running else {'state': 'not recorded'})
            meta = {'schema': 'opd-results-v1', 'name': a.name, 'stamp': stamp,
                    'published_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'job': job,
                    'publisher': {'host': socket.gethostname(), 'gpu': gpu_names(), 'code': code_state()},
                    'sources': infos, 'files': files, 'redacted_strings': redactions,
                    'skipped': skipped[:MAX_SKIPPED_LISTED], 'skipped_count': len(skipped),
                    'limits_mb': {'per_file': a.max_file_mb, 'total': a.max_total_mb}}
            (staged / folder).mkdir(parents=True, exist_ok=True)
            (staged / folder / 'PUBLISHED.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n',
                                                            encoding='utf-8')
            size = sum(f['bytes'] for f in files)
            log(f'{len(files)} files ({size / 2 ** 20:.2f} MB), {len(skipped)} left out, '
                f'{redactions} strings redacted -> {a.branch}:{folder}')
            if a.dry_run:
                for f in files:
                    print(f"  + {f['path']} ({f['bytes']} bytes)")
                for s in skipped:
                    print(f"  - {s['path']}: {s['reason']}")
                return 0
            state = job['state'] + (f" (exit {a.status})" if a.status is not None else '')
            result = push(git, url, a.branch, staged, folder, a.message or f'results({a.name}): {stamp} {state}')
        link = web_url(url, a.branch, folder)
        log(('nothing changed since the last publish' if result == 'unchanged' else 'pushed')
            + f'; {link or public(url) + " " + a.branch + ":" + folder}')
        return 0
    except PublishError as error:
        log(f'FAILED: {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
