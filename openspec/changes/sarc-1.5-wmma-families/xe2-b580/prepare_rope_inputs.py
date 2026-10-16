#!/usr/bin/env python3
"""Prepare a partial input repair; never run a Vulkan method or claim gate success."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
from inspect_models import Table, inspect


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def patches_for(path):
    with path.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data.find(b'VH00', 0, 2 * 1024 * 1024) - 4
        assert header >= 0
        start = header + struct.unpack_from('<I', data, header + 10)[0]
        assert data[start + 4:start + 8] == b'VK00'
        graph = Table(data, start + struct.unpack_from('<I', data, start)[0])
        values = graph.vector(2, tables=True)
        calls = [(op.string(1), op.vector(2)) for op in graph.vector(1, tables=True)]
        patches = []
        for name, args in calls:
            if name != 'et_vk.prepack.default':
                continue
            source, target = [values[index].table(1) for index in args]
            dims = target.vector(1, 'I')
            if dims not in ([5120, 32], [5120, 64]):
                continue
            assert source.vector(1, 'I') == dims and source.scalar(2) >= 0
            assert target.scalar(0, 'b') == 4 and target.scalar(2) == -1
            assert target.scalar(3) == -1 and target.scalar(4, 'B', 255) == 1
            assert target.scalar(5, 'B', 255) == 0 and target.field(5)
            users = [(op, refs) for op, refs in calls if args[1] in refs]
            assert len(users) == 2
            assert sum(op == 'aten.slice_copy.Tensor' and refs[0] == args[1]
                       for op, refs in users) == 1
            patches.append({'value_id': args[1], 'offset': target.field(5),
                            'before': 0, 'after': 1, 'dims': dims,
                            'old_extent': [dims[1] // 4, 5120, 1],
                            'new_extent': [dims[1], 1280, 1]})
        assert len(patches) == 2 and len({p['offset'] for p in patches}) == 2
        return sorted(patches, key=lambda p: p['offset'])


def constant_records(path):
    with path.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data.find(b'VH00', 0, 2 * 1024 * 1024) - 4
        start = header + struct.unpack_from('<I', data, header + 10)[0]
        base = header + struct.unpack_from('<I', data, header + 18)[0]
        graph = Table(data, start + struct.unpack_from('<I', data, start)[0])
        constants = graph.vector(5, tables=True)
        assert data[8:12] == b'eh00'
        segment_base = struct.unpack_from('<Q', data, 24)[0]
        program = Table(data, struct.unpack_from('<I', data)[0])
        segments = program.vector(4, tables=True)
        named_segments = {item.string(0): segments[item.scalar(1, 'I')]
                          for item in program.vector(7, tables=True)}
        rows = []
        for index, value in enumerate(graph.vector(2, tables=True)):
            if value.scalar(0, 'B') != 5:
                continue
            tensor = value.table(1)
            constant_id = tensor.scalar(2)
            if constant_id < 0:
                continue
            constant = constants[constant_id]
            offset, length = constant.scalar(0, 'Q'), constant.scalar(1, 'Q')
            if offset == 2**64 - 1:
                segment = named_segments[constant.string(2)]
                assert segment.scalar(1, 'Q') >= length
                absolute = segment_base + segment.scalar(0, 'Q')
            else:
                absolute = base + offset
            assert absolute + length <= len(data)
            digest = hashlib.sha256()
            for position in range(absolute, absolute + length, 8 * 1024 * 1024):
                digest.update(data[position:min(position + 8 * 1024 * 1024, absolute + length)])
            rows.append({'value_id': index, 'constant_id': constant_id,
                         'dims': tensor.vector(1, 'I'), 'dtype': tensor.scalar(0, 'b'),
                         'staging_dtype': tensor.scalar(6, 'b', 127),
                         'length': length, 'logical_payload_sha256': digest.hexdigest()})
        return rows


def prepare(record, output_root, helper):
    source = Path(record['path'])
    target = output_root / source.parent.parent.name / 'exported' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    patches = patches_for(source)
    original_hash = hashlib.sha256()
    with source.open('rb') as src, target.with_suffix('.partial').open('xb') as dst:
        position = 0
        while chunk := src.read(8 * 1024 * 1024):
            original_hash.update(chunk)
            revised = bytearray(chunk)
            for patch in patches:
                index = patch['offset'] - position
                if 0 <= index < len(chunk):
                    assert revised[index] == patch['before']
                    revised[index] = patch['after']
            dst.write(revised)
            position += len(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    assert position == record['bytes']
    assert original_hash.hexdigest() == record['sha256'], source
    revised_hash = hashlib.sha256()
    with source.open('rb') as src, target.with_suffix('.partial').open('rb') as dst:
        position = 0
        differences = []
        while original := src.read(8 * 1024 * 1024):
            revised = dst.read(len(original))
            assert len(revised) == len(original)
            revised_hash.update(revised)
            if original != revised:
                differences.extend(position + i for i, (a, b) in enumerate(zip(original, revised))
                                   if a != b)
            position += len(original)
        assert dst.read(1) == b''
    assert differences == [p['offset'] for p in patches]
    target.with_suffix('.partial').replace(target)
    before, after = inspect(source), inspect(target)
    for key in before.keys() - {'path'}:
        if key != 'linear_embedding_calls':
            assert before[key] == after[key], key
    assert before['linear_embedding_calls'] == after['linear_embedding_calls']
    constant_path = target.with_suffix('.constant-hashes.json')
    write_json(constant_path, constant_records(source))
    constants = subprocess.run([str(helper), str(target)], capture_output=True, text=True, check=True)
    target.with_suffix('.constants.csv').write_text(constants.stdout)
    target.with_suffix('.constants.stderr').write_text(constants.stderr)
    observed = {}
    for line in constants.stdout.splitlines():
        _, method, output, value = line.rsplit(',', 3)
        decoded = {'true': True, 'false': False}.get(value)
        if decoded is None:
            decoded = int(value)
        observed.setdefault(method, []).append((int(output), decoded))
    assert {name: [value for _, value in sorted(outputs)] for name, outputs in observed.items()} == after['constants']
    assert after['constants']['get_max_context_len'] == [2560]
    return {'source': record, 'output': str(target), 'sha256': revised_hash.hexdigest(),
            'patches': patches, 'changed_bytes': len(differences),
            'all_other_bytes_identical': True, 'runtime_constants': after['constants'],
            'constant_hashes': str(constant_path),
            'status': 'PARTIAL_REPAIR_NOT_GPU_VALIDATED',
            'remaining': ['8B wide texture3d images'] if '8b_' in source.name else [],
            'gpu_correctness': 'NOT_RUN', 'timing_authorized': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--helper-sha256', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'job-state.json'
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'purpose': 'CPU-only partial RoPE input preparation; no GPU or timing'}
    write_json(state_path, state)
    try:
        assert hashlib.sha256(args.helper.read_bytes()).hexdigest() == args.helper_sha256
        records = json.loads(args.hash_record.read_text())
        models = [record for record in records if record['path'].endswith('.pte')]
        assert len(models) == 6
        rows = []
        for record in models:
            state['current'] = Path(record['path']).name
            write_json(state_path, state)
            rows.append(prepare(record, args.output, args.helper))
            write_json(args.output / 'evidence.json', {'models': rows, 'complete': False})
        for record in records:
            if not record['path'].endswith('tokenizer.model'):
                continue
            source = Path(record['path'])
            content = source.read_bytes()
            assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
            target = args.output / source.parent.parent.name / 'original' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as file:
                file.write(content)
        write_json(args.output / 'evidence.json', {'models': rows, 'complete': True,
                   'scope': 'Only two RoPE layout bytes per model; all weights, methods, operators and other bytes identical. '
                            'Not six conforming inputs: 8B width, operator temporaries and GPU correctness remain unresolved.'})
        state.update(status='completed', completed=datetime.now(timezone.utc).isoformat(),
                     evidence=str(args.output / 'evidence.json'))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(state_path, state)


if __name__ == '__main__':
    main()
