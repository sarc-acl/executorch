import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('helper', type=Path)
parser.add_argument('models', type=Path)
parser.add_argument('hashes', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
models = json.loads(args.models.read_text())
assert len(models) == 6
expected_hashes = dict(line.split(None, 1)[::-1] for line in args.hashes.read_text().splitlines())
rows = []
for model in models:
    path = Path(model['path'])
    assert path.stat().st_size == model['bytes']
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    assert digest.hexdigest() == expected_hashes[str(path)]
    process = subprocess.run([str(args.helper), str(path)], text=True, capture_output=True)
    (args.output.parent / (path.name + '.metadata.stdout')).write_text(process.stdout)
    (args.output.parent / (path.name + '.metadata.stderr')).write_text(process.stderr)
    assert process.returncode == 0, (path.name, process.returncode, process.stderr)
    runtime = json.loads(process.stdout)['constant_methods']
    assert runtime == model['constant_methods'], (path.name, runtime)
    assert runtime['get_max_context_len'] == runtime['get_max_seq_len'] == [2560]
    assert runtime['get_bos_id'] == [128000]
    assert all(runtime[name] == [True] for name in
               ('use_kv_cache', 'use_sdpa_with_kv_cache', 'enable_dynamic_shape'))
    rows.append(dict(path=str(path), bytes=model['bytes'], sha256=digest.hexdigest(),
                     runtime_constants=runtime, rc=process.returncode,
                     group_embedding_evidence='Same bytes as preserved serialized graph/tensor inspection; these fields have no runtime constant methods'))
args.output.write_text(json.dumps(dict(
    status='PASS_RUNTIME_CONSTANTS', models=rows,
    helper_sha256=hashlib.sha256(args.helper.read_bytes()).hexdigest(),
    monitoring_visibility='own_user_plus_device_busy_other_users_unobservable',
    monitoring_owner_decision='2026-10-15 03:57 UTC',
    gpu_model_execution=False), indent=2) + '\n')
