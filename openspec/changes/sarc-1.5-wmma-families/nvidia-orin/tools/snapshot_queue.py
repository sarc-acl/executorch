#!/usr/bin/env python3
"""Detach a parent-snapshot queue and mirror terminal target evidence to the NAS."""
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
        try:
            with (job / 'queue.log').open('x') as output:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(args.config)],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        except BaseException as error:
            state(status, status='failed', error=str(error))
            raise
        print(status)
        return
    def interrupted(signum, frame):
        raise RuntimeError(f'bridge interrupted by signal {signum}; inspect target job before resuming')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    remote = config['remote_stage']
    def ssh(script, *arguments):
        command = 'bash -s -- ' + ' '.join(shlex.quote(str(x)) for x in arguments)
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', config['target'], command],
                                input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'ssh rc={result.returncode}: {result.stderr}')
        return result.stdout
    def scp(source, destination):
        subprocess.run(['scp', '-q', source, destination], check=True, timeout=600)
    try:
        artifacts = Path(config['artifacts'])
        while (artifacts / 'HOLD').exists():
            state(status, status='queued', phase='coordinator_hold')
            time.sleep(5)
        assert not (artifacts / 'GPU_GONE').exists()
        state(status, status='running', phase='prepare_payload', pid=os.getpid())
        tools = Path(__file__).resolve().parent
        build = Path(config['build'])
        manifest = json.loads((build / 'manifest.json').read_text())
        assert manifest['golden_rc'] == 0
        assert manifest['source_commit'] == config['source_commit']
        payload = job / 'payload'
        payload.mkdir()
        (payload / 'bin').mkdir()
        (payload / 'tools').mkdir()
        (payload / 'invocations').mkdir()
        for name, digest in manifest['binary_sha256'].items():
            source = build / 'bundle' / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            (payload / 'bin' / name).write_bytes(source.read_bytes())
            (payload / 'bin' / name).chmod(0o755)
        (payload / 'libllama_runner.so').symlink_to('bin/libllama_runner.so')
        for name in ('device_snapshot.py', 'sampler.py', 'parent_gate_check.py', 'nexttoken.py', 'runrow.py', 'gate_check_paths.py'):
            (payload / 'tools' / name).write_bytes((tools / name).read_bytes())
        repo = Path(config['repository'])
        for name in ('prompt_2048.txt', 'prompt_real_2048.txt', 'prompt_check.txt'):
            (payload / name).write_bytes((repo / 'openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts' / name).read_bytes())
        (payload / 'r1304.txt').write_bytes((repo / 'openspec/changes/sarc-1.5-orin-fused-port/tools/r1304.txt').read_bytes())
        (payload / 'verify.sh').write_bytes((build / 'source/executorch/sarc/tools/verify.sh').read_bytes())
        (payload / 'CAMPAIGN.md').write_bytes(Path(config['task_file']).read_bytes())
        for name in ('llama_main', 'test_llama_microbench'):
            (payload / name).write_text('#!/bin/bash\nexec python3 "$(dirname "$0")/tools/device_snapshot.py" "$WMMA_CONFIG" --invoke ' + name + ' "$@"\n')
            (payload / name).chmod(0o755)
        device = dict(config['device_config'], stage=remote,
                      source_commit=manifest['source_commit'], binary_sha256=manifest['binary_sha256'],
                      build_manifest_sha256=hashlib.sha256((build / 'manifest.json').read_bytes()).hexdigest(),
                      session=config['session'], tag=config['session'], owner_pid=0)
        (payload / 'config.json').write_text(json.dumps(device, indent=2) + '\n')
        (payload / 'source-provenance.json').write_text(json.dumps({
            'source_commit': manifest['source_commit'], 'binary_sha256': manifest['binary_sha256'],
            'orchestration_commit': config['orchestration_commit'],
            'payload_hashes': {str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in payload.rglob('*') if p.is_file() and not p.is_symlink()},
            'task_sha256': hashlib.sha256((payload / 'CAMPAIGN.md').read_bytes()).hexdigest()
        }, indent=2) + '\n')
        sensors = ssh('set -eu\npython3 - "$1" <<\'PY\'\nimport json,sys\nfrom pathlib import Path\nroot=Path(sys.argv[1])\nassert not (root/"GPU_GONE").exists()\ndef find(base, field, expected):\n return next(p for p in Path(base).iterdir() if (p/field).exists() and (p/field).read_text().strip()==expected)\nt=find("/sys/class/thermal", "type", "gpu-thermal")\nh=find("/sys/class/hwmon", "name", "ina3221")\ncooling=[{"type":(p/"type").read_text().strip(),"path":str(p/"cur_state")} for p in Path("/sys/class/thermal").glob("cooling_device*") if (p/"type").read_text().strip()!="pwm-fan"]\nprint(json.dumps({"clock":"/sys/class/devfreq/17000000.gpu/cur_freq","load":"/sys/devices/platform/17000000.gpu/load","temperature":str(t/"temp"),"voltage":str(h/"in1_input"),"current":str(h/"curr1_input"),"cooling":cooling,"gone_marker":str(root/"GPU_GONE")}))\nPY\n', device['deployment_root'])
        (payload / 'sensor-paths.json').write_text(sensors)
        archive = job / 'payload.tar'
        with tarfile.open(archive, 'w') as tar:
            for p in payload.iterdir():
                tar.add(p, arcname=p.name)
        ssh('set -eu\nmkdir -p "$(dirname "$1")"\n[ ! -e "$1" ]\n', remote)
        scp(str(archive), config['target'] + ':' + remote + '.tar')
        while (artifacts / 'HOLD').exists():
            state(status, status='queued', phase='coordinator_hold_before_gate')
            time.sleep(5)
        ssh('set -eu\nmkdir "$1"\ntar -xf "$1.tar" -C "$1"\nroot=$2\nif [ -e "$root/CAMPAIGN.md" ]; then cp -p "$root/CAMPAIGN.md" "$1/CAMPAIGN.previous.md"; fi\ncp "$1/CAMPAIGN.md" "$root/CAMPAIGN.md"\nnohup python3 "$1/tools/device_snapshot.py" "$1/config.json" > "$1/queue.log" 2>&1 < /dev/null &\n', remote, device['deployment_root'])
        state(status, status='running', phase='remote_parent_snapshot', remote_stage=remote)
        while True:
            raw = ssh('if [ -f "$1/state.json" ]; then cat "$1/state.json"; else echo \'{"status":"queued"}\'; fi\n', remote)
            target_state = json.loads(raw)
            if target_state['status'] in ('completed', 'failed'):
                break
            state(status, status=target_state['status'], phase='remote_parent_snapshot',
                  remote_stage=remote, target_state=target_state)
            time.sleep(15)
        state(status, status='running', phase='copy_target_evidence', target_state=target_state)
        ssh('set -eu\ntar -czf "$1.evidence.tgz" -C "$1" .\n', remote)
        output = job / 'target-evidence.tgz'
        scp(config['target'] + ':' + remote + '.evidence.tgz', str(output))
        evidence = artifacts / 'target-evidence' / config['session']
        evidence.mkdir(parents=True, exist_ok=False)
        with tarfile.open(output) as tar:
            tar.extractall(evidence, filter='data')
        state(status, status=target_state['status'], phase='parent_snapshot_mirrored',
              target_state=target_state, evidence=str(evidence), remote_stage=remote)
    except BaseException as error:
        traceback.print_exc()
        state(status, status='failed', error=str(error), remote_stage=remote,
              note='Inspect target state and retained data before any retry')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
