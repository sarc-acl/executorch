import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def save(path, state):
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(state, indent=2) + '\n')
    pending.replace(path)


def run(spec_path, state_path):
    state = json.loads(state_path.read_text())
    child = None

    def interrupted(signum, frame):
        raise RuntimeError(f'queue interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        spec = json.loads(spec_path.read_text())
        state.update(status='running', pid=os.getpid(), steps=[])
        save(state_path, state)
        env = os.environ.copy()
        env.update(spec.get('env', {}))
        env['WMMA_RUN_TAG'] = spec['tag']
        hold = Path(__file__).resolve().with_name('hold.sh')
        for index, step in enumerate(spec['steps']):
            log = state_path.parent / f'{index:02d}-{step["name"]}.log'
            entry = dict(name=step['name'], status='running', log=str(log),
                         argv=step['argv'])
            state['steps'].append(entry)
            save(state_path, state)
            with log.open('x') as output:
                child = subprocess.Popen(
                    ['bash', str(hold), 'run', step['name'], *step['argv']],
                    cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = child.pid
                save(state_path, state)
                rc = child.wait(timeout=step['timeout_s'])
            child = None
            entry.update(status='completed' if rc == 0 else 'failed', rc=rc)
            save(state_path, state)
            if rc:
                raise RuntimeError(f'{step["name"]} exited {rc}')
        state['status'] = 'completed'
        return 0
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        for entry in state.get('steps', []):
            if entry['status'] == 'running':
                entry.update(status='failed', error=state['error'])
        return 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save(state_path, state)


parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['start', 'run'])
parser.add_argument('spec', type=Path)
parser.add_argument('state', type=Path)
args = parser.parse_args()
args.spec = args.spec.resolve()
args.state = args.state.resolve()
if args.action == 'run':
    sys.exit(run(args.spec, args.state))

spec = json.loads(args.spec.read_text())
if not spec['steps'] or any(step['timeout_s'] <= 0 for step in spec['steps']):
    raise SystemExit('queue needs nonempty steps and positive timeouts')
args.state.parent.mkdir(parents=True, exist_ok=True)
state = dict(status='queued', tag=spec['tag'], spec=str(args.spec),
             monitoring_visibility=spec['monitoring_visibility'],
             monitoring_owner_decision=spec['monitoring_owner_decision'])
with args.state.open('x') as output:
    output.write(json.dumps(state, indent=2) + '\n')
save(args.state, state)
try:
    with (args.state.parent / 'queue.log').open('x') as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'run',
             str(args.spec), str(args.state)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
except BaseException as exc:
    state.update(status='failed', error=f'launch failed: {exc}')
    save(args.state, state)
    raise
print(json.dumps(dict(pid=process.pid, job_state_file=str(args.state))))
