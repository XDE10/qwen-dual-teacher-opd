"""Run a long command on a Colab VM so that it outlives a dropped colab-cli connection.

google-colab-cli (`colab`, 0.7.5) sends code to the VM's Jupyter kernel, and Colab keeps
the VM while that kernel is busy. This runner starts the command in its own session, so
a closed websocket or a kernel interrupt does not reach it, and writes all of its output
to <jobs-dir>/<name>/output.log. It keeps the kernel busy by printing that log until the
command exits. It uses no google.colab API, so no notebook page is needed.

  colab exec -s opd --timeout 86400 <<'EOF'
  %run /content/project2/scripts/colab_job_v3.py first_session -- bash scripts/colab_first_session_v3.sh
  EOF

Calling it again with the same name follows the job if it is still running, or reports
how it ended. --again starts a new run and renames the old job folder. The command runs
from the repository root. Stdlib only; `python scripts/colab_job_v3.py ...` works too.
"""
import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_DIR = os.environ.get('OPD_JOBS_DIR', '/content/jobs')
POLL_SECONDS = 2
HEARTBEAT_SECONDS = 600
ATTACH_TAIL_BYTES = 4096
# Runs the command from the repository root and records its exit status, also when the
# runner (or the kernel it runs in) is gone by the time the command finishes.
WRAPPER = 'cd "$1" && shift && "$@"; status=$?; echo "$status" > "$OPD_JOB_EXIT_FILE"; exit "$status"'


def say(text):
    print(f'[colab_job] {text}', flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


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


def job_state(job):
    """('absent', None), ('running', pid) or ('finished', exit status or None if unknown)."""
    if not job.is_dir():
        return 'absent', None
    exit_file = job / 'exit_code'
    if exit_file.exists():
        text = exit_file.read_text().strip()
        return 'finished', int(text) if re.fullmatch(r'-?\d+', text) else None
    pid_file = job / 'pid'
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    if pid and pid_running(pid):
        return 'running', pid
    return 'finished', None  # ended without recording a status (killed, or never started)


def start(job, command):
    job.mkdir(parents=True)
    (job / 'command.json').write_text(json.dumps({'command': command, 'cwd': str(ROOT), 'started_utc': utc_now()},
                                                 indent=2) + '\n')
    env = {**os.environ, 'OPD_JOB_EXIT_FILE': str(job / 'exit_code'), 'PYTHONUNBUFFERED': '1'}
    with open(job / 'output.log', 'ab') as log:
        proc = subprocess.Popen(['bash', '-c', WRAPPER, 'colab_job', str(ROOT), *command], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
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
    command = []
    if '--' in argv:
        split = argv.index('--')
        argv, command = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(prog='colab_job_v3.py', usage='%(prog)s NAME [--again] [--jobs-dir DIR] [-- COMMAND ...]')
    parser.add_argument('name', help='job name: letters, digits, "_", "-" and "."')
    parser.add_argument('--again', action='store_true', help='start a new run of a job that has finished')
    parser.add_argument('--jobs-dir', default=DEFAULT_JOBS_DIR, help='default: $OPD_JOBS_DIR or /content/jobs')
    a = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', a.name):
        parser.error(f'invalid job name {a.name!r}')
    job = Path(a.jobs_dir) / a.name
    state, value = job_state(job)

    if state == 'running':
        say(f'{a.name} is already running (pid {value}); following its log')
        if command:
            say('the command given now is ignored')
        size = (job / 'output.log').stat().st_size
        try:
            follow(job, lambda: job_state(job)[0] != 'running', max(0, size - ATTACH_TAIL_BYTES))
        except KeyboardInterrupt:
            return detached(a.name, value)
        return finish(a.name, job_state(job)[1])

    if state == 'finished' and not a.again:
        if (job / 'output.log').exists():
            print(tail(job / 'output.log'), flush=True)
        say(f'{a.name} already finished with exit status {value if value is not None else "unknown"}; '
            f'log {job / "output.log"}. Add --again to start a new run.')
        return None

    if not command:
        parser.error('give the command after --')
    if job.exists():
        old = job.with_name(f'{job.name}.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}')
        job.rename(old)
        say(f'previous run moved to {old}')
    proc = start(job, command)
    say(f'started {a.name} (pid {proc.pid}) in {ROOT}; log {job / "output.log"}')
    say('closing the terminal does not stop it; do not run `colab restart-kernel` or `colab stop` until it ends')
    try:
        follow(job, lambda: proc.poll() is not None)
    except KeyboardInterrupt:
        return detached(a.name, proc.pid)
    return finish(a.name, proc.returncode)


if __name__ == '__main__':
    main()
