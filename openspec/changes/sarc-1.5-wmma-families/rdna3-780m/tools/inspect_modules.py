import argparse
import hashlib
import json
from pathlib import Path
import struct


def inspect(path):
    data = path.read_bytes()
    words = struct.unpack(f'<{len(data) // 4}I', data)
    assert words[0] == 0x07230203
    constants, types, matrices, value_types = {}, {}, {}, {}
    instructions = []
    cursor = 5
    while cursor < len(words):
        size, op = words[cursor] >> 16, words[cursor] & 65535
        assert size and cursor + size <= len(words)
        operands = words[cursor + 1:cursor + size]
        instructions.append((op, operands))
        if op == 43:  # OpConstant; specialization-dependent shared arrays are rejected.
            constants[operands[1]] = operands[2]
        elif op in (21, 22, 23, 28, 32):
            types[operands[0]] = (op, operands[1:])
        elif op == 4456:  # OpTypeCooperativeMatrixKHR
            matrices[operands[0]] = operands[1:]
        cursor += size
    for op, operands in instructions:
        if len(operands) > 1 and operands[0] in matrices:
            value_types[operands[1]] = operands[0]

    def size_of(type_id):
        op, operands = types[type_id]
        if op in (21, 22):
            assert operands[0] in (8, 16, 32, 64)
            return operands[0] // 8
        if op == 23:
            assert operands[1] in (2, 4)
            return size_of(operands[0]) * operands[1]
        if op == 28:
            return size_of(operands[0]) * constants[operands[1]]
        raise ValueError(f'Unsupported Workgroup type: {op}')

    shared = []
    used_mma = set()
    for op, operands in instructions:
        if op == 59 and operands[2] == 4:  # OpVariable, Workgroup storage class
            pointer_op, pointer = types[operands[0]]
            assert pointer_op == 32 and pointer[0] == 4
            shared.append(dict(variable=operands[1], bytes=size_of(pointer[1])))
        if op == 4459:  # OpCooperativeMatrixMulAddKHR
            type_ids = [value_types[value] for value in operands[2:5]] + [operands[0]]
            signature = []
            for type_id in type_ids:
                component, scope, rows, columns, use = matrices[type_id]
                component_op, component_type = types[component]
                signature.append((component_op, *component_type,
                                  constants[scope], constants[rows],
                                  constants[columns], constants[use]))
            used_mma.add(tuple(signature))
    assert shared and used_mma
    return dict(sha256=hashlib.sha256(data).hexdigest(), shared_variables=shared,
                workgroup_payload_bytes=sum(item['bytes'] for item in shared),
                used_mma=[list(signature) for signature in sorted(used_mma)])


parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
parser.add_argument('profile', type=Path)
parser.add_argument('build', type=Path)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
limit = json.loads(args.profile.read_text())['capabilities']['device']['properties'][
    'VkPhysicalDeviceProperties']['limits']['maxComputeSharedMemorySize']
rows = []
for index, candidate in enumerate(manifest['candidates'], 1):
    row = dict(candidate=f'rdna3-{index:02d}', declared_shared_bytes=candidate['declared_shared_bytes'],
               limit=limit, modules=[])
    for alias in candidate['dispatch_aliases']:
        for arm in ('backend', 'llama'):
            result = inspect(args.build / arm / 'vulkan_compute_shaders' / (alias['shader'] + '.spv'))
            assert result['workgroup_payload_bytes'] <= candidate['declared_shared_bytes'] <= limit
            for signature in result['used_mma']:
                if candidate['scheme'] == '4w':
                    expected = ((22, 16, 3, 16, 16, 0), (22, 16, 3, 16, 16, 1),
                                (22, 32, 3, 16, 16, 2), (22, 32, 3, 16, 16, 2))
                else:
                    expected = ((21, 8, 1, 3, 16, 16, 0), (21, 8, 1, 3, 16, 16, 1),
                                (21, 32, 1, 3, 16, 16, 2), (21, 32, 1, 3, 16, 16, 2))
                assert tuple(map(tuple, signature)) == expected, (alias['shader'], signature)
            result.update(shader=alias['shader'], build=arm)
            row['modules'].append(result)
    rows.append(row)
print(json.dumps(dict(status='PASS_SPIRV_REQUIREMENTS', candidates=rows,
                      limitation='Workgroup payload and used MMA types only; driver LDS allocation, runtime storage, origin binding and gates remain pending'), indent=2))
