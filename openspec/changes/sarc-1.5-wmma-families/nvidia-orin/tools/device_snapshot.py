#!/usr/bin/env python3
"""Run the unchanged parent snapshot with per-invocation evidence; no timing adoption."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from sampler import foreign_processes


def state(path, **values):
    values['updated_epoch_us'] = time.time_ns() // 1000
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2) + '\n')
    tmp.replace(path)


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def memory():
    result = {}
    for source, names in [('/proc/meminfo', {'MemAvailable:', 'SwapFree:'}),
                          ('/proc/vmstat', {'pswpin', 'pswpout'})]:
        for line in Path(source).read_text().splitlines():
            key, value, *_ = line.split()
            if key in names:
                result[key.rstrip(':')] = int(value)
    return result


def check(config):
    root = Path(config['deployment_root'])
    while (root / 'HOLD').exists():
        time.sleep(5)
    if (root / 'GPU_GONE').exists() or (Path(config['stage']) / 'FOREIGN').exists():
        raise RuntimeError('device-loss or foreign-workload marker; no further GPU work')
    foreign = foreign_processes(config['owner_pid'], config['tag'])
    if foreign:
        raise RuntimeError('foreign workload: ' + json.dumps(foreign))


def cool(config, paths):
    start = reference_time = time.monotonic()
    reference = int(Path(paths['temperature']).read_text())
    while True:
        check(config)
        current = int(Path(paths['temperature']).read_text())
        now = time.monotonic()
        if current <= config['idle_temperature_milli'] + 5000:
            return
        if now - reference_time >= 20:
            if reference - current < 250:
                return
            reference, reference_time = current, now
        if now - start >= 120:
            return
        time.sleep(5)


def invoke(config, program, arguments):
    stage = Path(config['stage'])
    paths = json.loads((stage / 'sensor-paths.json').read_text())
    check(config)
    cool(config, paths)
    model = None
    if '--model_path' in arguments:
        model = Path(arguments[arguments.index('--model_path') + 1])
        if '8b_' in model.name and memory()['MemAvailable'] < 5000 * 1024:
            raise RuntimeError('less than 5000 MiB available before 8B process')
        with model.open('rb') as file:
            while file.read(4 * 1024 * 1024):
                pass
    check(config)
    identity = str(time.time_ns()) + '-' + str(os.getpid())
    sensor = stage / 'invocations' / (identity + '.jsonl')
    record = {'id': identity, 'program': program, 'args': arguments,
              'commit': config['source_commit'], 'build_manifest_sha256': config['build_manifest_sha256'],
              'binary_sha256': config['binary_sha256'][program],
              'profile': os.environ['ET_VK_SARC_DEV_PROFILE'], 'candidate': 'control',
              'device': config['device'], 'unit': config['unit'], 'driver': config['driver'],
              'session': config['session'], 'sensor_path': str(sensor),
              'environment': {k: v for k, v in os.environ.items() if k.startswith(('ET_VK', 'ETVK'))},
              'page_cache_filled': model is not None, 'memory_before': memory(),
              'foreign_before': foreign_processes(config['owner_pid'], config['tag']),
              'stdout_log': os.readlink('/proc/self/fd/1')}
    sampler = child = None
    interrupted = []
    def on_signal(signum, frame):
        interrupted.append(signum)
        stop(child)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        sampler = subprocess.Popen([sys.executable, str(Path(__file__).with_name('sampler.py')),
                                   str(stage / 'sensor-paths.json'), str(sensor), '--owner-pid',
                                   str(config['owner_pid']), '--tag', config['tag']], start_new_session=True)
        time.sleep(0.15)
        if sampler.poll() is not None:
            raise RuntimeError('sampler failed before invocation')
        record['process_start_epoch_us'] = time.time_ns() // 1000
        child = subprocess.Popen([str(stage / 'bin' / program)] + arguments,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        while child.poll() is None:
            if Path(paths['gone_marker']).exists() or sampler.poll() is not None:
                stop(child)
                raise RuntimeError('sensor failure/device loss during invocation')
            time.sleep(0.1)
        record['rc'] = child.returncode
        if interrupted:
            record['rc'] = 128 + interrupted[0]
        record['process_end_epoch_us'] = time.time_ns() // 1000
        record['sampler_children'] = Path(f'/proc/{sampler.pid}/task/{sampler.pid}/children').read_text().strip()
        if record['sampler_children']:
            raise RuntimeError('sampler unexpectedly has children')
    finally:
        stop(child)
        stop(sampler)
        record['memory_after'] = memory()
        samples = [json.loads(line) for line in sensor.read_text().splitlines()] if sensor.exists() else []
        record['sensor_samples'] = len(samples)
        record['sampler_rc'] = sampler.returncode if sampler else None
        record['foreign_during'] = [x for sample in samples for x in sample.get('foreign', [])]
        record['foreign_after'] = foreign_processes(config['owner_pid'], config['tag'])
        record['sensor_error'] = any('sensor_error' in sample for sample in samples) or not samples
        if record['foreign_before'] or record['foreign_during'] or record['foreign_after']:
            (stage / 'FOREIGN').write_text(json.dumps(record) + '\n')
        with (stage / 'invocations.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
            file.flush()
            os.fsync(file.fileno())
        if program == 'llama_main':
            def arg(name, default=''):
                return arguments[arguments.index(name) + 1] if name in arguments else default
            previous = {'rc': record.get('rc', -1), 'model': Path(arg('--model_path')).name,
                        'prompt': Path(arg('--prompt_file')).name, 'max_new_tokens': arg('--max_new_tokens'),
                        'warmup': int('--warmup' in arguments),
                        'env': ' '.join(k + '=' + v for k, v in record['environment'].items())}
            with (stage / 'verify-runs.jsonl').open('a') as file:
                file.write(json.dumps(previous) + '\n')
    check(config)
    return record['rc']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--invoke', choices=['llama_main', 'test_llama_microbench'])
    args, remaining = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    if args.invoke:
        raise SystemExit(invoke(config, args.invoke, remaining))
    stage = Path(config['stage'])
    status = stage / 'state.json'
    try:
        config['owner_pid'] = os.getpid()
        paths = json.loads((stage / 'sensor-paths.json').read_text())
        check(config)
        config['idle_temperature_milli'] = int(Path(paths['temperature']).read_text())
        args.config.write_text(json.dumps(config, indent=2) + '\n')
        os.environ['WMMA_CONFIG'] = str(args.config)
        os.environ['SARC_CAMPAIGN_TAG'] = config['tag']
        for key in list(os.environ):
            if key.startswith(('ET_VK', 'ETVK')):
                del os.environ[key]
        os.environ.update(config['environment'])
        os.environ['LD_LIBRARY_PATH'] = str(stage / 'bin')
        for name, digest in config['binary_sha256'].items():
            assert hashlib.sha256((stage / 'bin' / name).read_bytes()).hexdigest() == digest
        state(status, status='running', phase='parent_snapshot', pid=os.getpid())
        command = ['bash', str(stage / 'verify.sh'), '--dir', str(stage), '--lock', config['lock_uuid'],
                   '--models', '1b,3b,8b', '--schemes', '4w,8da4w', '--pdiff', '--flat-models', config['models']]
        with (stage / 'verify.out').open('x') as file:
            result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, timeout=6 * 3600)
            file.write(f'VERIFY_DONE rc={result.returncode}\n')
        check(config)
        records = [json.loads(line) for line in (stage / 'invocations.jsonl').read_text().splitlines()]
        assert len(records) == 37, f'expected 37 gate invocations, got {len(records)}'
        assert all(not x['sensor_error'] and x['sampler_children'] == '' and x['sampler_rc'] == 0 for x in records)
        assert sum(x['program'] == 'llama_main' for x in records) == 22
        with (stage / 'content-check.txt').open('x') as file:
            content = subprocess.run([sys.executable, str(stage / 'tools/parent_gate_check.py'), 'verify', str(stage), str(stage)],
                                     stdout=file, stderr=subprocess.STDOUT)
        assert result.returncode == 0 and content.returncode == 0, 'parent snapshot content check failed; diagnose before retry'
        state(status, status='completed', phase='parent_snapshot_recorded', gpu_invocations=len(records),
              note='Snapshot only; not candidate acceptance or A/A qualification')
    except BaseException as error:
        traceback.print_exc()
        state(status, status='failed', error=str(error), phase='parent_snapshot')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
