#!/usr/bin/env python3
"""Run a detached queue and atomically publish its terminal status."""
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys

state = pathlib.Path(sys.argv[1])
command = sys.argv[2:]
record = json.loads(state.read_text())
child = None


def update(status, **fields):
    record.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    temporary = state.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(state)


def interrupted(signum, frame):
    if child is not None:
        os.killpg(child.pid, signum)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    raise SystemExit(128 + signum)


for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(signum, interrupted)

code = 1
try:
    update('running', pid=os.getpid(), command=command)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('ET_VK', 'ETVK', 'XE2_'))}
    environment['XE2_TOP'] = str(os.getpid())
    environment['XE2_JOB_TAG'] = record['job_id']
    child = subprocess.Popen(command, env=environment, start_new_session=True)
    code = child.wait()
except BaseException as error:
    record['error'] = str(error)
    if isinstance(error, SystemExit):
        code = int(error.code)
finally:
    update('completed' if code == 0 else 'failed', exit_code=code)
sys.exit(code)
