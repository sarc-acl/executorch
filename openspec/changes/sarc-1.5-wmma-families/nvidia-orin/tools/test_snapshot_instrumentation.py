#!/usr/bin/env python3
"""Exercise actual invocation/sampling children with a synthetic CPU runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stage = args.output.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    for name in ('bin', 'invocations', 'sensors'):
        (stage / name).mkdir()
    paths = {}
    for name, value in [('clock', 612000000), ('load', 0), ('temperature', 49000),
                        ('voltage', 1000), ('current', 1000), ('cooling', 0)]:
        p = stage / 'sensors' / name
        p.write_text(str(value))
        paths[name] = str(p)
    paths['cooling'] = [{'type': 'synthetic-thermal-cap', 'path': paths['cooling']}]
    paths['gone_marker'] = str(stage / 'GPU_GONE')
    (stage / 'sensor-paths.json').write_text(json.dumps(paths))
    runner = stage / 'bin/llama_main'
    runner.write_text('#!/usr/bin/env python3\nimport sys,time\na=time.time()*1000\ntime.sleep(0.8 if "--long" not in sys.argv else 20)\nprint("PyTorchObserver {\\"prompt_tokens\\":2048,\\"generated_tokens\\":0,\\"prefill_token_per_sec\\":2560}")\nsys.exit(7 if "--fail" in sys.argv else 0)\n')
    runner.chmod(0o755)
    config = {'stage': str(stage), 'deployment_root': str(stage), 'owner_pid': os.getpid(),
              'tag': 'synthetic-snapshot-' + str(os.getpid()), 'source_commit': 'synthetic',
              'build_manifest_sha256': 'synthetic', 'binary_sha256': {'llama_main': hashlib.sha256(runner.read_bytes()).hexdigest()},
              'device': 'synthetic', 'unit': 'synthetic', 'driver': 'synthetic',
              'session': 'synthetic', 'idle_temperature_milli': 49000}
    path = stage / 'config.json'; path.write_text(json.dumps(config))
    env = dict(os.environ, SARC_CAMPAIGN_TAG=config['tag'], ET_VK_SARC_DEV_PROFILE='orin-fused1')
    script = Path(__file__).with_name('device_snapshot.py').resolve()
    for name, extra, expected in [('success', [], 0), ('failure', ['--fail'], 7), ('signal', ['--long'], 143)]:
        with (stage / (name + '.log')).open('w') as output:
            process = subprocess.Popen([sys.executable, str(script), str(path), '--invoke', 'llama_main'] + extra,
                                       env=env, stdout=output, stderr=subprocess.STDOUT)
            if name == 'signal':
                time.sleep(0.5)
                process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == expected, name
    records = [json.loads(line) for line in (stage / 'invocations.jsonl').read_text().splitlines()]
    assert [x['rc'] for x in records] == [0, 7, 143]
    assert all(x['sampler_children'] == '' and x['sampler_rc'] == 0 and not x['sensor_error'] for x in records)
    assert all(not x['foreign_before'] and not x['foreign_during'] and not x['foreign_after'] for x in records)
    assert min(x['sensor_samples'] for x in records[:2]) >= 5
    assert len((stage / 'verify-runs.jsonl').read_text().splitlines()) == 3
    print('PASS: success/failure rc retention, SIGTERM propagation, childless sampler cleanup, raw sensor/provenance and gate exit records (CPU only)')


if __name__ == '__main__':
    main()
