#!/usr/bin/env python3
"""Deploy and detach the Orin WMMA matrix queue (matrix_device.py), mirror its evidence to the NAS while it runs.

  matrix_queue.py <config.json> --launch

The local state.json mirrors the device queue's status (queued/running/completed/failed); it is failed also when
this bridge loses the target for 30 minutes, with the device queue left running."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from device_snapshot import state

TOOLS = ('device_snapshot.py', 'sampler.py', 'parent_gate_check.py', 'nexttoken.py', 'runrow.py', 'gate_check_paths.py',
         'screen.py', 'matrix_device.py', 'sdpa_check.py')
E2E = ('e2e5.sh', 'common.sh')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    job = Path(config['job_directory'])
    status = job / 'state.json'
    if args.launch:
        job.mkdir(parents=True, exist_ok=False)
        state(status, status='queued', phase='starting')
        with (job / 'queue.log').open('x') as output:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(args.config)],
                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(status)
        return

    def interrupted(signum, frame):
        raise RuntimeError(f'bridge interrupted by signal {signum}; the device queue may still run')
    signal.signal(signal.SIGTERM, interrupted)
    remote = config['remote_stage']
    target = config['target']

    def ssh(script, *arguments):
        command = 'bash -s -- ' + ' '.join(shlex.quote(str(x)) for x in arguments)
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', target, command],
                                input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f'ssh rc={result.returncode}: {result.stderr}')
        return result.stdout

    evidence = Path(config['artifacts']) / 'target-evidence' / config['tag']

    def mirror():
        subprocess.run(['rsync', '-a', '--exclude', 'bin/', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20',
                        f'{target}:{remote}/', str(evidence) + '/'], check=True, timeout=1800)

    try:
        artifacts = Path(config['artifacts'])
        while (artifacts / 'HOLD').exists():
            state(status, status='queued', phase='coordinator_hold')
            time.sleep(30)
        state(status, status='running', phase='prepare_payload', pid=os.getpid())
        tools = Path(__file__).resolve().parent
        build = Path(config['build'])
        manifest = json.loads((build / 'manifest.json').read_text())
        assert manifest['golden_rc'] == 0 and manifest['source_commit'] == config['source_commit']
        payload = job / 'payload'
        for sub in ('bin', 'tools/e2e', 'steps'):
            (payload / sub).mkdir(parents=True)
        for name, digest in manifest['binary_sha256'].items():
            data = (build / 'bundle' / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, name
            (payload / 'bin' / name).write_bytes(data)
            (payload / 'bin' / name).chmod(0o755)
        for name in TOOLS:
            (payload / 'tools' / name).write_bytes((tools / name).read_bytes())
        for name in E2E:
            (payload / 'tools/e2e' / name).write_bytes((tools / 'e2e' / name).read_bytes())
        for name in ('runrow.py', 'nexttoken.py'):
            (payload / 'tools/e2e' / name).write_bytes((tools / name).read_bytes())
        repo = Path(config['repository'])
        for name in ('prompt_2048.txt', 'prompt_real_2048.txt', 'prompt_check.txt'):
            (payload / name).write_bytes((repo / 'openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts' / name).read_bytes())
        (payload / 'r1304.txt').write_bytes((repo / 'openspec/changes/sarc-1.5-orin-fused-port/tools/r1304.txt').read_bytes())
        (payload / 'verify.sh').write_bytes((build / 'source/executorch/sarc/tools/verify.sh').read_bytes())
        (payload / 'CAMPAIGN.md').write_bytes(Path(config['task_file']).read_bytes())
        profiles = json.loads((repo / config['candidate_profiles']).read_text())
        by_profile = {c['profile']: c for c in profiles['candidates']}
        device = dict(config['device_config'], source_commit=manifest['source_commit'], binary_sha256=manifest['binary_sha256'],
                      build_manifest_sha256=hashlib.sha256((build / 'manifest.json').read_bytes()).hexdigest(),
                      build=build.name)
        plan = {'tag': config['tag'], 'deployment_root': device['deployment_root'], 'lock_uuid': device['lock_uuid'],
                'models': device['models'], 'reps': config['reps'], 'extra': config['extra'],
                'sdpa_tiers': config.get('sdpa_tiers'),
                'control_env': device['environment'], 'device': device,
                'candidates': [by_profile[p] for p in config['order']]}
        assert sorted(config['order']) == sorted(by_profile), 'order must list every manifest candidate once'
        (payload / 'config.json').write_text(json.dumps(plan, indent=1) + '\n')
        (payload / 'source-provenance.json').write_text(json.dumps({
            'source_commit': manifest['source_commit'], 'binary_sha256': manifest['binary_sha256'],
            'orchestration_commit': config['orchestration_commit'],
            'payload_hashes': {str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(payload.rglob('*')) if p.is_file()},
            'task_sha256': hashlib.sha256((payload / 'CAMPAIGN.md').read_bytes()).hexdigest()}, indent=1) + '\n')
        sensors = ssh('set -eu\npython3 - "$1" <<\'PY\'\nimport json,sys\nfrom pathlib import Path\nroot=Path(sys.argv[1])\nassert not (root/"GPU_GONE").exists()\ndef find(base, field, expected):\n return next(p for p in Path(base).iterdir() if (p/field).exists() and (p/field).read_text().strip()==expected)\nt=find("/sys/class/thermal", "type", "gpu-thermal")\nh=find("/sys/class/hwmon", "name", "ina3221")\ncooling=[{"type":(p/"type").read_text().strip(),"path":str(p/"cur_state")} for p in Path("/sys/class/thermal").glob("cooling_device*") if (p/"type").read_text().strip()!="pwm-fan"]\nprint(json.dumps({"clock":"/sys/class/devfreq/17000000.gpu/cur_freq","load":"/sys/devices/platform/17000000.gpu/load","temperature":str(t/"temp"),"voltage":str(h/"in1_input"),"current":str(h/"curr1_input"),"cooling":cooling,"gone_marker":str(root/"GPU_GONE")}))\nPY\n', device['deployment_root'])
        (payload / 'sensor-paths.json').write_text(sensors)
        archive = job / 'payload.tar'
        with tarfile.open(archive, 'w') as tar:
            for p in payload.iterdir():
                tar.add(p, arcname=p.name)
        ssh('set -eu\nmkdir -p "$(dirname "$1")"\n[ ! -e "$1" ]\n', remote)
        subprocess.run(['scp', '-q', str(archive), f'{target}:{remote}.tar'], check=True, timeout=900)
        while (artifacts / 'HOLD').exists():
            state(status, status='queued', phase='coordinator_hold_before_launch')
            time.sleep(30)
        ssh('set -eu\nmkdir "$1"\ntar -xf "$1.tar" -C "$1"\nroot=$2\nif [ -e "$root/CAMPAIGN.md" ]; then cp -p "$root/CAMPAIGN.md" "$1/CAMPAIGN.previous.md"; fi\n'
            'cp "$1/CAMPAIGN.md" "$root/CAMPAIGN.md"\ncd "$1/tools"\nnohup python3 "$1/tools/matrix_device.py" "$1/config.json" > "$1/queue.log" 2>&1 < /dev/null &\n',
            remote, device['deployment_root'])
        evidence.mkdir(parents=True, exist_ok=False)
        failures, last_mirror, target_state = 0, 0.0, {'status': 'queued'}
        while True:
            try:
                raw = ssh('if [ -f "$1/state.json" ]; then cat "$1/state.json"; else echo \'{"status":"queued"}\'; fi\n', remote)
                target_state = json.loads(raw)
                if time.monotonic() - last_mirror > 600 or target_state['status'] in ('completed', 'failed'):
                    mirror()
                    last_mirror = time.monotonic()
                failures = 0
            except (RuntimeError, subprocess.SubprocessError, ValueError) as error:
                failures += 1
                if failures >= 30:
                    raise RuntimeError(f'target unreachable for 30 polls ({error}); the device queue may still run')
            if target_state['status'] in ('completed', 'failed'):
                break
            state(status, status=target_state['status'], phase=target_state.get('phase', 'remote'), remote_stage=remote,
                  evidence=str(evidence), target_steps=[{k: s.get(k) for k in ('name', 'rc', 'verdict', 'timing')}
                                                        for s in target_state.get('steps', [])])
            time.sleep(60)
        state(status, status=target_state['status'], phase='mirrored', target_state=target_state,
              evidence=str(evidence), remote_stage=remote)
    except BaseException as error:
        traceback.print_exc()
        state(status, status='failed', error=str(error), remote_stage=remote,
              note='Inspect the target queue and the mirrored evidence before any retry')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
