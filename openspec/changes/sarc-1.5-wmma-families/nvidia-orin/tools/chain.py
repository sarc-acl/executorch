#!/usr/bin/env python3
"""Detached chain: run queue tools one after another, each synchronously on its own config, and stop at the first
failure. A tool is `<script> <config>`, run in its own job directory as with --launch, but in the foreground.

  chain.py <job directory> --launch <script> <config> [<script> <config> ...]

<job directory>/state.json is queued/running/completed/failed and names the step that runs or failed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from device_snapshot import state


def main():
    job = Path(sys.argv[1])
    status = job / 'state.json'
    if sys.argv[2] == '--launch':
        job.mkdir(parents=True, exist_ok=False)
        state(status, status='queued', phase='starting')
        with (job / 'chain.log').open('x') as output:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(job), '--run'] + sys.argv[3:],
                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(status)
        return
    pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
    done = []
    try:
        for script, config in pairs:
            directory = Path(json.loads(Path(config).read_text())['job_directory'])
            directory.mkdir(parents=True, exist_ok=False)
            state(directory / 'state.json', status='queued', phase='starting', chain=str(job))
            state(status, status='running', phase=f'{Path(script).name} {directory.name}', done=done, pid=os.getpid(),
                  job_state=str(directory / 'state.json'))
            with (directory / 'queue.log').open('x') as output:
                rc = subprocess.run([sys.executable, script, config], stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT).returncode
            result = json.loads((directory / 'state.json').read_text())
            if rc or result.get('status') != 'completed':
                raise RuntimeError(f'{directory.name}: rc {rc}, status {result.get("status")}: {result.get("error")}')
            done.append(str(directory / 'state.json'))
        state(status, status='completed', phase='done', done=done)
    except BaseException as error:
        traceback.print_exc()
        state(status, status='failed', error=str(error), done=done)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
