#!/usr/bin/env python3
"""Device side of the Orin WMMA matrix (owner decision 2026-10-15 23:30): control verify.sh, a six-cell A/A,
the clock floor, then per candidate verify.sh -> screen.py -> e2e5.sh session against the control.

All steps run on the one build in bin/. Every GPU step takes the device lock (verify.sh, e2e5.sh), every step
boundary waits for HOLD, and a device-loss, foreign-workload or memory abort stops the queue. state.json is
queued/running/completed/failed and lists every step with its outcome."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from sampler import foreign_processes

PROMPTS = ('prompt_2048.txt', 'prompt_real_2048.txt', 'prompt_check.txt', 'r1304.txt')
STOP_CODES = {70: 'device loss', 75: 'device lock busy', 76: 'foreign GPU workload', 77: 'missing input or memory'}


class Stop(Exception):
    pass


def main():
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())
    stage = config_path.parent
    root = Path(config['deployment_root'])
    status = stage / 'state.json'
    steps = []

    def state(**values):
        values.update(steps=steps, updated_epoch_us=time.time_ns() // 1000, pid=os.getpid())
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(values, indent=1) + '\n')
        tmp.replace(status)

    def boundary(phase):
        while (root / 'HOLD').exists():
            state(status='queued', phase='coordinator_hold before ' + phase)
            time.sleep(30)
        if (root / 'GPU_GONE').exists():
            raise Stop('GPU_GONE marker present')
        foreign = foreign_processes(os.getpid(), config['tag'])
        if foreign:
            raise Stop('foreign workload before ' + phase + ': ' + json.dumps(foreign))
        state(status='running', phase=phase)

    def leftovers():
        mine = {os.getpid()}
        found = []
        for process in Path('/proc').iterdir():
            if not process.name.isdigit() or int(process.name) in mine:
                continue
            try:
                if ('SARC_CAMPAIGN_TAG=' + config['tag']).encode() in (process / 'environ').read_bytes().split(b'\0'):
                    found.append(int(process.name))
            except OSError:
                continue
        return found

    def environment(extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('ET_VK', 'ETVK'))}
        env.update(extra, SARC_CAMPAIGN_TAG=config['tag'])
        return env

    def finish_step(step, rc):
        step['rc'] = rc
        step['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        rest = leftovers()
        if rest:
            step['leftover_processes'] = rest
            raise Stop(f'processes of this queue left after {step["name"]}: {rest}')
        if rc in STOP_CODES:
            raise Stop(f'{step["name"]}: rc {rc} ({STOP_CODES[rc]})')

    def verify(name, candidate, env):
        directory = stage / 'steps' / f'verify-{name}'
        step = {'name': f'verify-{name}', 'kind': 'verify', 'candidate': candidate, 'dir': str(directory),
                'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        steps.append(step)
        boundary(step['name'])
        directory.mkdir(parents=True)
        (directory / 'invocations').mkdir()
        (directory / 'bin').symlink_to(stage / 'bin')
        (directory / 'libllama_runner.so').symlink_to(stage / 'bin' / 'libllama_runner.so')
        for prompt in PROMPTS:
            shutil.copy(stage / prompt, directory / prompt)
        shutil.copy(stage / 'sensor-paths.json', directory / 'sensor-paths.json')
        paths = json.loads((stage / 'sensor-paths.json').read_text())
        local = dict(config['device'], stage=str(directory), candidate=candidate, environment=env,
                     session=f'{config["tag"]}-verify-{name}', tag=config['tag'], owner_pid=os.getpid(),
                     idle_temperature_milli=int(Path(paths['temperature']).read_text()))
        (directory / 'config.json').write_text(json.dumps(local, indent=1) + '\n')
        for program in ('llama_main', 'test_llama_microbench'):
            wrapper = directory / program
            wrapper.write_text(f'#!/bin/bash\nexec python3 {stage}/tools/device_snapshot.py "$WMMA_CONFIG" --invoke {program} "$@"\n')
            wrapper.chmod(0o755)
        command = ['bash', str(stage / 'verify.sh'), '--dir', str(directory), '--lock', config['lock_uuid'],
                   '--models', '1b,3b,8b', '--schemes', '4w,8da4w', '--pdiff', '--flat-models', config['models']]
        run_env = environment(dict(env, WMMA_CONFIG=str(directory / 'config.json')))
        with (directory / 'verify.out').open('x') as output:
            rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=run_env, timeout=4 * 3600).returncode
            output.write(f'VERIFY_DONE rc={rc}\n')
        if (directory / 'FOREIGN').exists():
            raise Stop(f'{step["name"]}: foreign workload recorded during verify.sh')
        records = [json.loads(l) for l in (directory / 'invocations.jsonl').read_text().splitlines()]
        step['invocations'] = len(records)
        step['instrumentation_ok'] = len(records) == 37 and all(
            not r.get('instrumentation_error') and not r['sensor_error'] and r.get('sampler_children') == [] and r['sampler_rc'] == 0
            for r in records)
        control_dir = stage / 'steps' / 'verify-control'
        with (directory / 'content-check.txt').open('x') as output:
            step['content_check_rc'] = subprocess.run(
                [sys.executable, str(stage / 'tools' / 'parent_gate_check.py'), 'verify', str(directory), str(control_dir)],
                stdout=output, stderr=subprocess.STDOUT, env=dict(run_env, WMMA_CONFIG=str(directory / 'config.json'))).returncode
        finish_step(step, rc)
        if not step['instrumentation_ok']:
            raise Stop(f'{step["name"]}: instrumentation incomplete ({len(records)} records)')
        return directory, step

    def session(name, candidate, env, calibrate):
        directory = stage / 'steps' / f'session-{name}'
        step = {'name': f'session-{name}', 'kind': 'session', 'candidate': candidate, 'dir': str(directory),
                'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        steps.append(step)
        boundary(step['name'])
        for arm, arm_env in (('parent', config['control_env']), ('cand', env)):
            (directory / arm).mkdir(parents=True)
            for binary in ('llama_main', 'libllama_runner.so'):
                (directory / arm / binary).symlink_to(stage / 'bin' / binary)
            (directory / arm / 'env').write_text(''.join(f'{k}={v}\n' for k, v in arm_env.items()))
            (directory / arm / 'COMMIT').write_text(config['device']['source_commit'] + '\n')
        for prompt in PROMPTS:
            shutil.copy(stage / prompt, directory / prompt)
        (directory / 'STAGE.md').write_text(
            f'session {step["name"]} of {config["tag"]}: parent = control {config["control_env"]}, cand = {candidate} {env}\n'
            f'build {config["device"]["build"]} source {config["device"]["source_commit"]} '
            f'manifest {config["device"]["build_manifest_sha256"]} binaries {json.dumps(config["device"]["binary_sha256"])}\n')
        run_env = environment({})
        tools = stage / 'tools' / 'e2e'
        subprocess.run(['bash', '-c', f'source {tools}/common.sh; cool_start 300'], env=run_env, check=True)
        command = ['bash', str(tools / 'e2e5.sh'), '--stage', str(directory), '--out', 'raw', '--lock', config['lock_uuid'],
                   '--reps', str(config['reps']), '--extra', str(config['extra'])]
        command += ['--calibrate'] if calibrate else ['--clkmin-file', str(stage / 'floor.json')]
        with (directory / 'e2e5.out').open('x') as output:
            rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=run_env, timeout=6 * 3600).returncode
        finish_step(step, rc)
        step['complete'] = rc == 0
        return step

    def sdpa(name, candidate, env):
        # Finalists only (coordinator decision 2026-10-16 08:20): the SDPA tiers of R7 under the device lock.
        directory = stage / 'steps' / f'sdpa-{name}'
        step = {'name': f'sdpa-{name}', 'kind': 'sdpa', 'candidate': candidate, 'dir': str(directory),
                'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        steps.append(step)
        boundary(step['name'])
        directory.mkdir(parents=True)
        (directory / 'invocations').mkdir()
        (directory / 'bin').symlink_to(stage / 'bin')
        shutil.copy(stage / 'sensor-paths.json', directory / 'sensor-paths.json')
        paths = json.loads((stage / 'sensor-paths.json').read_text())
        local = dict(config['device'], stage=str(directory), candidate=candidate, environment=env,
                     session=f'{config["tag"]}-sdpa-{name}', tag=config['tag'], owner_pid=os.getpid(),
                     idle_temperature_milli=int(Path(paths['temperature']).read_text()))
        (directory / 'config.json').write_text(json.dumps(local, indent=1) + '\n')
        (directory / 'env').write_text(''.join(f'{k}={v}\n' for k, v in env.items()))
        run_env = environment(dict(env, LD_LIBRARY_PATH=str(stage / 'bin')))
        rc = 0
        with open(Path.home() / '.cache/gpu-lab' / f'lock-{config["lock_uuid"]}', 'a') as lock:
            deadline = time.monotonic() + 900
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        finish_step(step, 75)
                    time.sleep(5)
            for tier, passes in config['sdpa_tiers']:
                for i in range(1, passes + 1):
                    with (directory / f'cand-{tier}-r{i}.log').open('x') as output:
                        rc = subprocess.run([sys.executable, str(stage / 'tools' / 'device_snapshot.py'), str(directory / 'config.json'),
                                             '--invoke', 'test_llama_microbench', '--sdpa-correctness-only', f'--sdpa-tier={tier}'],
                                            stdout=output, stderr=subprocess.STDOUT, env=run_env, timeout=3600).returncode
                    if rc or (directory / 'FOREIGN').exists():
                        break
                if rc or (directory / 'FOREIGN').exists():
                    break
        if (directory / 'FOREIGN').exists():
            raise Stop(f'{step["name"]}: foreign workload recorded during the SDPA tiers')
        records = [json.loads(l) for l in (directory / 'invocations.jsonl').read_text().splitlines()]
        step['invocations'] = len(records)
        step['instrumentation_ok'] = len(records) == sum(n for _, n in config['sdpa_tiers']) and all(
            not r.get('instrumentation_error') and not r['sensor_error'] and r.get('sampler_children') == [] and r['sampler_rc'] == 0
            for r in records)
        with (directory / 'sdpa-check.txt').open('x') as output:
            step['sdpa_check_rc'] = subprocess.run([sys.executable, str(stage / 'tools' / 'sdpa_check.py'), str(directory),
                                                    str(directory / 'env')], stdout=output, stderr=subprocess.STDOUT).returncode
        step['verdict'] = 'ACCEPT' if step['sdpa_check_rc'] == 0 and step['instrumentation_ok'] else 'REJECT'
        finish_step(step, rc)

    def interrupted(signum, frame):
        raise Stop(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        state(status='running', phase='start')
        for name, digest in config['device']['binary_sha256'].items():
            assert subprocess.run(['sha256sum', str(stage / 'bin' / name)], capture_output=True, text=True,
                                  check=True).stdout.split()[0] == digest, name
        control_dir, _ = verify('control', 'control', config['control_env'])
        aa = session('aa', 'control-aa', config['control_env'], calibrate=True)
        rows = (stage / 'steps' / 'session-aa' / 'raw' / 'runs.csv').read_text().splitlines()
        header = rows[0].split(',')
        valid = [dict(zip(header, r.split(','))) for r in rows[1:]]
        valid = [r for r in valid if r['log'].startswith('logs/prefill') and r['valid'] == '1']
        if not aa['complete'] or not valid:
            raise Stop('A/A session incomplete; no clock floor')
        lowest = min(float(r['clk_med_mhz']) for r in valid)
        floor = 0.97 * lowest
        cells = {f'{m}-{q}': {'clkmin_mhz': math.ceil(floor)} for m in ('1b', '3b', '8b') for q in ('4w', '8da4w')}
        (stage / 'floor.json').write_text(json.dumps({
            'rule': '0.97 x lowest valid per-run median clock of the A/A session, device-wide; e2e5.sh takes whole MHz, '
                    'so the exact floor is rounded up (never down)',
            'lowest_valid_run_median_mhz': lowest, 'floor_mhz_exact': floor, 'valid_runs': len(valid), 'cells': cells}, indent=1) + '\n')
        for candidate in config['candidates']:
            profile = candidate['profile']
            env = dict(config['control_env'], ET_VK_SARC_DEV_PROFILE=profile)
            directory, step = verify(profile, profile, env)
            screen = directory / 'screen.json'
            result = subprocess.run([sys.executable, str(stage / 'tools' / 'screen.py'), str(control_dir), str(directory), str(screen)],
                                    capture_output=True, text=True, env=environment({}))
            step['screen'] = result.stdout.strip() or result.stderr.strip()[-400:]
            verdict = json.loads(screen.read_text())['verdict'] if screen.exists() else 'SCREEN_ERROR'
            step['verdict'] = verdict
            if verdict in ('REJECTED_PDIFF', 'SCREEN_ERROR'):
                step['timing'] = 'not timed: ' + verdict
            else:
                if config.get('sdpa_tiers'):
                    sdpa(profile, profile, env)
                if not candidate['timed_projection_io']:
                    step['timing'] = 'not timed: storage is not the models\' texture3d projection IO'
                else:
                    session(profile, profile, env, calibrate=False)
            state(status='running', phase='after ' + profile)
        state(status='completed', phase='done')
    except BaseException as error:
        traceback.print_exc()
        state(status='failed', phase='stopped', error=str(error))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
