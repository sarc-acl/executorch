#!/usr/bin/env python3
"""Serialize the CPU input-preparation job with workstation timing/builds."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_rope_inputs import write_json


def main():
    source, artifacts, output, hashes, helper, helper_hash = map(str, sys.argv[1:])
    output = Path(output)
    state_path = output / 'job-state.json'
    state = {'status': 'queued', 'pid': os.getpid(),
             'purpose': 'CPU-only partial RoPE input preparation',
             'waiting_for': 'desktop-build lock / coordinator HOLD'}
    write_json(state_path, state)
    try:
        command = [sys.executable, str(Path(source) / 'prepare_rope_inputs.py'),
                   '--hash-record', hashes, '--output', str(output),
                   '--helper', helper, '--helper-sha256', helper_hash]
        write_json(output / 'command.json', command)
        env = dict(os.environ, SARC_HOLD_DIR=artifacts,
                   TMPDIR=str(Path(artifacts) / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
        with (Path.home() / '.cache/gpu-lab/lock-desktop-build').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(['bash', str(Path(source) / 'tools/hold.sh'), 'run',
                            'CPU partial RoPE input preparation', *command], env=env, check=True)
        assert json.loads(state_path.read_text())['status'] == 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
