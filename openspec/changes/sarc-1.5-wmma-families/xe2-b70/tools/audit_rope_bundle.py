#!/usr/bin/env python3
"""Audit the delivered partial recipe against local originals without allocating GPU resources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--original-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
with args.archive.open('rb') as handle:
    assert hashlib.file_digest(handle, 'sha256').hexdigest() == 'ba113d7778555427639cea0da7444bdfa018791428fd1e218e8ebdc3f343005b'
provenance = json.loads((args.bundle / 'recipe-provenance.json').read_text())
assert provenance['commit'] == '8c6c59c0ff922bd437d78fea7ec7c08dba1815fb'
for name, expected in provenance['files'].items():
    assert hashlib.sha256((args.bundle / 'recipe' / name).read_bytes()).hexdigest() == expected, name

sys.dont_write_bytecode = True
sys.path.insert(0, str(args.bundle / 'recipe'))
from prepare_rope_inputs import constant_records, inspect, patches_for, write_json

args.output.mkdir(parents=True, exist_ok=False)
delivered = json.loads((args.bundle / 'evidence.json').read_text())
assert len(delivered['models']) == 6
assert delivered['complete'] is True and 'Not six conforming inputs' in delivered['scope']
report = {
    'scope': 'Local original/virtual-RoPE hash and serialized-constant audit only; no PTE created, GPU work or gate pass',
    'recipe_commit': provenance['commit'],
    'models': [],
    'cpu_audit_complete': False,
    'whole_model_legality': 'UNRESOLVED',
    'runtime_on_revised_files': 'NOT_RUN_NO_REVISED_FILES_CREATED',
    'gpu_correctness': 'NOT_RUN',
    'timing_authorized': False,
}
seen = set()
for row in delivered['models']:
    relative = Path(row['source']['path']).relative_to(args.original_root)
    assert relative not in seen
    seen.add(relative)
    source = args.original_root / relative
    patches = patches_for(source)
    assert patches == row['patches'] and len(patches) == row['changed_bytes'] == 2
    assert row['all_other_bytes_identical'] is True
    original_hash, revised_hash = hashlib.sha256(), hashlib.sha256()
    offset = 0
    with source.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            original_hash.update(chunk)
            revised = bytearray(chunk)
            for patch in patches:
                index = patch['offset'] - offset
                if 0 <= index < len(chunk):
                    assert revised[index] == patch['before'] == 0 and patch['after'] == 1
                    revised[index] = patch['after']
            revised_hash.update(revised)
            offset += len(chunk)
    assert offset == row['source']['bytes'] == source.stat().st_size
    assert original_hash.hexdigest() == row['source']['sha256'], source
    assert revised_hash.hexdigest() == row['sha256'], source
    constants = constant_records(source)
    expected = json.loads((args.bundle / relative.with_suffix('.constant-hashes.json')).read_text())
    assert constants == expected and len(constants) > 0, source
    metadata = inspect(source)
    assert metadata['constants'] == row['runtime_constants'], source
    assert metadata['constants']['get_max_context_len'] == [2560]
    report['models'].append({
        'source': str(source), 'source_sha256': original_hash.hexdigest(),
        'bytes': offset, 'virtual_revised_sha256': revised_hash.hexdigest(),
        'patches': patches, 'constant_records_matched': len(constants),
        'serialized_constants_match_delivered_runtime_observations': True,
        'status': 'CPU_PARTIAL_RECIPE_APPLICABLE; WHOLE_MODEL_LEGALITY_UNRESOLVED',
    })
    write_json(args.output / 'evidence.json', report)
    print(f'{source.name}: original, virtual revision and {len(constants)} constant records match', flush=True)
expected = {Path(model) / 'exported' / f'{stem}_vulkan_{scheme}.pte'
            for model, stem in [('llama-3.1-8b', 'llama3_1-8b'), ('llama-3.2-1b', 'llama3_2-1b'), ('llama-3.2-3b', 'llama3_2-3b')]
            for scheme in ('4w', '8da4w')}
assert seen == expected
report['cpu_audit_complete'] = True
write_json(args.output / 'evidence.json', report)
print('CPU_BUNDLE_AUDIT_COMPLETE; whole-model legality, revised runtime, GPU correctness and matrix remain pending')
