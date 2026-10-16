#!/usr/bin/env python3
"""Run one detached command and maintain the controller's JSON status file."""

import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


state_path = Path(sys.argv[1])
command = sys.argv[2:]
state = {
    "status": "queued",
    "command": command,
    "pid": os.getpid(),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}


def save():
    state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(state_path)


def interrupted(signum, _frame):
    raise SystemExit(f"received signal {signum}")


child = None
save()
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGHUP, interrupted)
try:
    child = subprocess.Popen(command, start_new_session=True)
    state.update(status="running", child_pid=child.pid)
    save()
    returncode = child.wait()
    state.update(
        status="completed" if returncode == 0 else "failed", returncode=returncode
    )
except BaseException as error:
    state.update(status="failed", error=str(error))
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
finally:
    save()
sys.exit(0 if state["status"] == "completed" else 1)
