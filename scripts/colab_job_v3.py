"""Run a long command on a Colab VM so that it outlives a dropped colab-cli connection, and keep
its results when it ends.

google-colab-cli (`colab`, 0.7.5) runs code in the VM's Jupyter kernel, and Colab keeps the VM
while that kernel is busy. This runner starts the command under a detached supervisor process
(its own session), so a closed websocket or a kernel interrupt does not reach it. All output
goes to <jobs-dir>/<name>/output.log, and the runner keeps the kernel busy by printing that log
until the job ends. When the command exits, with any status, the supervisor
  * copies the job folder and every --save path to Google Drive with `rclone copy`
    (--drive REMOTE:PATH, into REMOTE:PATH/<name>/<stamp>/<label>/), and
  * pushes the small text files among them to the results branch with
    scripts/publish_results_v3.py (--publish, into <name>/<stamp>/<label>/),
and only then records the exit status. --sync-every MIN also does both while the job runs.
Before starting, the runner checks that the rclone remote answers and that GitHub accepts a push.

  colab exec -s opd --timeout 86400 <<'EOF'
  %run /content/project2/scripts/colab_job_v3.py first_session --publish --drive mydrive:opd_runs --save report=/content/opd_first_session.tar.gz -- bash scripts/colab_first_session_v3.sh
  EOF

Running it again with the same name follows the job if it is still running, or reports how it
ended. --again starts a new run and renames the old job folder. The command runs from the
repository root. Stdlib only.
"""
import argparse
import codecs
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / 'scripts' / 'publish_results_v3.py'
DEFAULT_JOBS_DIR = os.environ.get('OPD_JOBS_DIR', '/content/jobs')
POLL_SECONDS = 2
HEARTBEAT_SECONDS = 600
ATTACH_TAIL_BYTES = 4096
SAFE_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')
# --local-no-check-updated: output.log and metrics.jsonl may still grow while they are copied.
RCLONE_FLAGS = ['--stats', '1m', '--stats-one-line', '--stats-log-level', 'NOTICE', '--local-no-check-updated']
FINAL_COPY_TIMEOUT = 3 * 3600
PERIODIC_COPY_TIMEOUT = 3600
PUBLISH_TIMEOUT = 900
USAGE = ('%(prog)s NAME [--again] [--save [LABEL=]PATH ...] [--drive REMOTE:PATH] [--publish] '
         '[--sync-every MIN] [--jobs-dir DIR] [-- COMMAND ...]')


def say(text):
    print(f'[colab_job] {text}', flush=True)


def utc_now(compact=False):
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ' if compact else '%Y-%m-%dT%H:%M:%SZ')


def pid_running(pid):
    """True while ``pid`` exists and is not a zombie (finished but not yet reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


def read_int(path):
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if re.fullmatch(r'-?\d+', text) else None


def read_spec(job):
    try:
        return json.loads((job / 'command.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def job_state(job):
    """('absent', None), ('running', pid) or ('finished', exit status or None if unknown)."""
    if not job.is_dir():
        return 'absent', None
    if (job / 'exit_code').exists():
        return 'finished', read_int(job / 'exit_code')
    for name in ('pid', 'child_pid'):
        pid = read_int(job / name)
        if pid and pid_running(pid):
            return 'running', pid
    return 'finished', None  # ended without recording a status (killed, or never started)


def destinations(spec):
    places = []
    if spec.get('drive'):
        places.append(f"Drive {spec['drive'].rstrip('/')}/{spec['name']}/{spec['stamp']}/")
    if spec.get('publish'):
        places.append(f"GitHub results branch {spec['name']}/{spec['stamp']}/")
    return '; '.join(places)


def parse_save(text):
    label, sep, rest = text.partition('=')
    if not (sep and SAFE_NAME.fullmatch(label) and rest):
        label, rest = None, text
    path = Path(os.path.expanduser(rest))
    if not path.is_absolute():
        path = ROOT / path
    if label is None:
        label = re.sub(r'[^A-Za-z0-9_.-]', '_', re.sub(r'(\.tar\.gz|\.tgz|\.tar)$', '', path.name)).lstrip('.-')
    return [label or 'save', str(path)]


def run_capture(cmd, timeout):
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(cmd, 124, str(error))


def run_logged(cmd, timeout):
    """Run with output going to our stdout (the job log); returns the exit status."""
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        say(f'{cmd[0]} timed out after {timeout} s')
        return 124
    except OSError as error:
        say(f'cannot run {cmd[0]}: {error}')
        return 127


def check_drive(remote):
    if ':' not in remote:
        return f'--drive must look like REMOTE:PATH, got {remote!r}'
    if not shutil.which('rclone'):
        return 'rclone is not installed on this VM (docs/colab_cli_wsl2.md, step 2)'
    name = remote.split(':', 1)[0] + ':'
    result = run_capture(['rclone', 'listremotes'], 60)
    if result.returncode:
        return 'rclone listremotes failed: ' + result.stdout.strip()[-500:]
    if name not in result.stdout.split():
        return f'rclone has no remote {name}; is rclone.conf in ~/.config/rclone/?'
    result = run_capture(['rclone', 'lsd', name], 120)
    if result.returncode:
        return f'rclone cannot list {name}: ' + result.stdout.strip()[-500:]
    return None


def check_publish():
    result = run_capture([sys.executable, str(PUBLISHER), '--check'], 180)
    if result.returncode:
        return 'GitHub does not accept pushes (needed by --publish):\n' + result.stdout.strip()[-1000:]
    return None


def save(job, spec, status):
    """Copy to Drive and publish to GitHub; ``status`` None means a periodic save of a running job."""
    final = status is not None
    sources = [['job', str(job)]] + spec.get('save', [])
    if spec.get('drive'):
        base = f"{spec['drive'].rstrip('/')}/{spec['name']}/{spec['stamp']}"
        failed = []
        for label, path in sources:
            if not os.path.exists(path):
                if final:
                    say(f'{label}: {path} does not exist; nothing to copy')
                continue
            code = run_logged(['rclone', 'copy', path, f'{base}/{label}', *RCLONE_FLAGS],
                              FINAL_COPY_TIMEOUT if final else PERIODIC_COPY_TIMEOUT)
            if code:
                failed.append(f'{label} (rclone exit {code})')
        say(f'Drive copy incomplete: {", ".join(failed)}; destination {base}' if failed else f'saved to Drive: {base}')
    if spec.get('publish'):
        cmd = [sys.executable, str(PUBLISHER), '--name', spec['name'], '--stamp', spec['stamp']]
        for label, path in sources:
            if os.path.exists(path):
                cmd += ['--source', f'{label}={path}']
        cmd += ['--status', str(status)] if final else ['--running']
        code = run_logged(cmd, PUBLISH_TIMEOUT)
        if code:
            say(f'publishing to GitHub failed (exit {code}); the job folder {job} is unaffected')


def wait_for(child, job, spec):
    interval = float(spec.get('sync_every_min') or 0) * 60
    while True:
        try:
            code = child.wait(timeout=interval or None)
            break
        except subprocess.TimeoutExpired:
            say(f'periodic save (every {spec["sync_every_min"]:g} min)')
            try:
                save(job, spec, None)
            except Exception as error:  # a failed periodic save must not stop the job
                say(f'periodic save failed: {error!r}')
    return 128 - code if code < 0 else code  # killed by signal N -> 128 + N, as in a shell


def supervise(job):
    spec = read_spec(job)
    (job / 'pid').write_text(f'{os.getpid()}\n')
    say(f'running: {shlex.join(spec["command"])}')
    try:
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], stdin=subprocess.DEVNULL)
    except OSError as error:
        say(f'cannot start the command: {error}')
        status = 127
    else:
        (job / 'child_pid').write_text(f'{child.pid}\n')
        signal.signal(signal.SIGTERM, lambda signum, frame: child.send_signal(signal.SIGTERM))
        status = wait_for(child, job, spec)
    try:
        if spec.get('drive') or spec.get('publish'):
            say(f'command exited with status {status}; saving results')
            save(job, spec, status)
    except Exception as error:  # never lose the exit status
        say(f'saving results failed: {error!r}')
    finally:
        (job / 'exit_code').write_text(f'{status}\n')
    return status


def start(job, spec):
    job.mkdir(parents=True)
    (job / 'command.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    with open(job / 'output.log', 'ab') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervise', str(job)],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(ROOT), start_new_session=True)
    (job / 'pid').write_text(f'{proc.pid}\n')
    return proc


def follow(job, finished, offset=0):
    """Print output.log from ``offset`` until ``finished()``; print a heartbeat while it is silent."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    started = last_output = time.monotonic()
    at_line_start = True

    def show(text):
        nonlocal at_line_start
        if text:
            print(text, end='', flush=True)
            at_line_start = text.endswith(('\n', '\r'))

    with open(job / 'output.log', 'rb') as log:
        log.seek(offset)
        while True:
            done = finished()  # checked before reading, so bytes written before the exit are still printed
            chunk = log.read()
            if chunk:
                show(decoder.decode(chunk))
                last_output = time.monotonic()
            elif done:
                break
            elif time.monotonic() - last_output >= HEARTBEAT_SECONDS:
                show('' if at_line_start else '\n')
                say(f'{job.name} still running after {(time.monotonic() - started) / 60:.0f} min '
                    f'(no new output for {HEARTBEAT_SECONDS // 60} min)')
                last_output = time.monotonic()
            else:
                time.sleep(POLL_SECONDS)
    show(decoder.decode(b'', final=True))
    show('' if at_line_start else '\n')


def tail(path, lines=40):
    with open(path, 'rb') as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - 16384))
        return b'\n'.join(f.read().splitlines()[-lines:]).decode('utf-8', 'replace')


def finish(name, code):
    say(f'{name} finished with exit status {code if code is not None else "unknown (killed?)"}')
    if code:
        raise SystemExit(code)


def detached(name, pid):
    say(f'stopped following {name}; the job keeps running (pid {pid}). '
        'Run the same command again to follow it.')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--supervise'] and len(argv) == 2:
        raise SystemExit(supervise(Path(argv[1])))
    command = []
    if '--' in argv:
        split = argv.index('--')
        argv, command = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(prog='colab_job_v3.py', usage=USAGE)
    parser.add_argument('name', help='job name: letters, digits, "_", "-" and "."')
    parser.add_argument('--again', action='store_true', help='start a new run of a job that has finished')
    parser.add_argument('--save', action='append', default=[], metavar='[LABEL=]PATH',
                        help='file, folder or .tar.gz to keep when the job ends (repeatable)')
    parser.add_argument('--drive', metavar='REMOTE:PATH', help='rclone destination, e.g. mydrive:opd_runs')
    parser.add_argument('--publish', action='store_true', help='push small text results to the results branch')
    parser.add_argument('--sync-every', type=float, default=0, metavar='MIN',
                        help='also save every MIN minutes while the job runs (0: only at the end)')
    parser.add_argument('--jobs-dir', default=DEFAULT_JOBS_DIR, help='default: $OPD_JOBS_DIR or /content/jobs')
    a = parser.parse_args(argv)
    if not SAFE_NAME.fullmatch(a.name):
        parser.error(f'invalid job name {a.name!r}')
    job = Path(a.jobs_dir) / a.name
    state, value = job_state(job)

    if state == 'running':
        say(f'{a.name} is already running (pid {value}); following its log')
        if command:
            say('the command and options given now are ignored')
        size = (job / 'output.log').stat().st_size
        try:
            follow(job, lambda: job_state(job)[0] != 'running', max(0, size - ATTACH_TAIL_BYTES))
        except KeyboardInterrupt:
            return detached(a.name, value)
        return finish(a.name, job_state(job)[1])

    if state == 'finished' and not a.again:
        if (job / 'output.log').exists():
            print(tail(job / 'output.log'), flush=True)
        where = destinations(read_spec(job))
        say(f'{a.name} already finished with exit status {value if value is not None else "unknown"}; '
            f'log {job / "output.log"}' + (f'; results in {where}' if where else '') + '. Add --again to start a new run.')
        return None

    if not command:
        parser.error('give the command after --')
    saves = [parse_save(s) for s in a.save]
    labels = [label for label, _ in saves]
    if 'job' in labels or len(set(labels)) != len(labels):
        parser.error('--save labels must be unique and must not be "job"')
    if a.sync_every < 0 or (a.sync_every and not (a.drive or a.publish)):
        parser.error('--sync-every needs a positive number of minutes and --drive or --publish')
    problems = [p for p in (check_drive(a.drive) if a.drive else None, check_publish() if a.publish else None) if p]
    if problems:
        for problem in problems:
            say(problem)
        say('nothing was started')
        raise SystemExit(2)
    if job.exists():
        old = job.with_name(f"{job.name}.{read_spec(job).get('stamp') or utc_now(compact=True)}")
        job.rename(old)
        say(f'previous run moved to {old}')
    spec = {'name': a.name, 'command': command, 'cwd': str(ROOT), 'started_utc': utc_now(),
            'stamp': utc_now(compact=True), 'save': saves, 'drive': a.drive, 'publish': a.publish,
            'sync_every_min': a.sync_every}
    proc = start(job, spec)
    say(f'started {a.name} (pid {proc.pid}) in {ROOT}; log {job / "output.log"}')
    if destinations(spec):
        say(f'results will be saved to {destinations(spec)}')
    say('closing the terminal does not stop it; do not run `colab restart-kernel` or `colab stop` until it ends')
    try:
        follow(job, lambda: proc.poll() is not None)
    except KeyboardInterrupt:
        return detached(a.name, proc.pid)
    return finish(a.name, job_state(job)[1])


if __name__ == '__main__':
    main()
