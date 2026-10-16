#!/usr/bin/env python3
"""Print exact compiled Xe2 module capacities and hashes as artifact evidence."""
import argparse
import hashlib
import json
import pathlib
import struct

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=pathlib.Path)
parser.add_argument("spv_dir", type=pathlib.Path)
args = parser.parse_args()
rows = []
for candidate in json.loads(args.manifest.read_text())["candidates"]:
    path = args.spv_dir / (candidate["shader"] + ".spv")
    data = path.read_bytes()
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    assert words[0] == 0x07230203
    types, constants, strides, shared, capabilities = {}, {}, {}, [], []
    position = 5
    while position < len(words):
        count, opcode = words[position] >> 16, words[position] & 0xFFFF
        assert count > 0
        operands = words[position + 1 : position + count]
        if opcode == 17:
            capabilities.append(operands[0])
        elif opcode in (21, 22, 23, 28, 30, 32):
            types[operands[0]] = (opcode, operands[1:])
        elif opcode == 43:
            constants[operands[1]] = operands[2]
        elif opcode == 71 and operands[1] == 6:
            strides[operands[0]] = operands[2]
        elif opcode == 59 and operands[2] == 4:
            shared.append(operands[0])
        position += count

    def size_alignment(type_id):
        opcode, parameters = types[type_id]
        if opcode in (21, 22):
            size = parameters[0] // 8
            return size, size
        if opcode == 23:
            size, align = size_alignment(parameters[0])
            count = parameters[1]
            return size * count, align * (4 if count == 3 else count)
        if opcode == 28:
            size, align = size_alignment(parameters[0])
            stride = strides.get(type_id, (size + align - 1) // align * align)
            return stride * constants[parameters[1]], align
        if opcode == 32:
            assert parameters[0] == 4
            return size_alignment(parameters[1])
        if opcode == 30:
            offset, alignment = 0, 1
            for member in parameters:
                size, align = size_alignment(member)
                offset = (offset + align - 1) // align * align + size
                alignment = max(alignment, align)
            return (offset + alignment - 1) // alignment * alignment, alignment
        raise ValueError(f"Unsupported shared type opcode {opcode}")

    allocations = [size_alignment(type_id) for type_id in shared]
    shared_bytes = 0
    for size, align in allocations:
        shared_bytes = (shared_bytes + align - 1) // align * align + size
    rows.append(
        {
            "shader": candidate["shader"],
            "binary_sha256": hashlib.sha256(data).hexdigest(),
            "module_shared_capacity_bytes": shared_bytes,
            "declared_shared_bytes": candidate["declared_shared_bytes"],
            "shared_allocations_size_alignment": allocations,
            "capabilities": capabilities,
            "capacity_within_source_bound": shared_bytes
            <= candidate["declared_shared_bytes"],
            "status": "MODULE_CAPACITY_ONLY; queried limits, features, formats and correctness checked separately",
        }
    )
print(json.dumps(rows, indent=2))
raise SystemExit(any(not row["capacity_within_source_bound"] for row in rows))
