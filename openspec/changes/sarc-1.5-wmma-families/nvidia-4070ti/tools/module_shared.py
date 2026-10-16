"""Reflect fixed Workgroup arrays from the exact compiled manifest modules."""

import hashlib
import json
from pathlib import Path
import struct
import sys


manifest = json.loads(Path(sys.argv[1]).read_text())
directory = Path(sys.argv[2])
rows = []
for candidate in manifest["candidates"]:
    path = directory / (candidate["shader"] + ".spv")
    data = path.read_bytes()
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    assert words[0] == 0x07230203
    types, constants, names, strides = {}, {}, {}, {}
    variables, capabilities = [], []
    cursor = 5
    while cursor < len(words):
        length, opcode = words[cursor] >> 16, words[cursor] & 65535
        assert length > 0
        operands = words[cursor + 1:cursor + length]
        if opcode == 5:
            names[operands[0]] = struct.pack("<" + "I" * (len(operands) - 1), *operands[1:]).split(b"\0")[0].decode()
        elif opcode == 17:
            capabilities.append(operands[0])
        elif opcode in [21, 22, 23, 24, 28, 30, 32]:
            types[operands[0]] = (opcode, operands[1:])
        elif opcode == 43:
            constants[operands[1]] = operands[2]
        elif opcode == 59 and operands[2] == 4:
            variables.append((operands[1], operands[0]))
        elif opcode == 71 and operands[1] == 6:
            strides[operands[0]] = operands[2]
        cursor += length

    def size_align(type_id):
        opcode, args = types[type_id]
        if opcode in [21, 22]:
            return args[0] // 8, args[0] // 8
        if opcode == 23:
            size, alignment = size_align(args[0])
            return size * args[1], alignment * (4 if args[1] == 3 else args[1])
        if opcode == 28:
            size, alignment = size_align(args[0])
            stride = strides.get(type_id, (size + alignment - 1) // alignment * alignment)
            return stride * constants[args[1]], alignment
        if opcode == 32:
            return size_align(args[1])
        raise ValueError(f"unsupported shared type {type_id}: {opcode}")

    arrays = []
    offset = 0
    for variable, type_id in variables:
        size, alignment = size_align(type_id)
        offset = (offset + alignment - 1) // alignment * alignment
        arrays.append({"name": names.get(variable, str(variable)), "bytes": size,
                       "natural_alignment": alignment, "offset": offset})
        offset += size
    raw = sum(array["bytes"] for array in arrays)
    assert raw == candidate["declared_shared_bytes"], (candidate["shader"], raw, candidate["declared_shared_bytes"])
    assert offset <= 49152
    rows.append({"shader": candidate["shader"], "sha256": hashlib.sha256(data).hexdigest(),
                 "capabilities": capabilities, "workgroup_arrays": arrays,
                 "array_bytes": raw, "natural_aligned_bytes": offset,
                 "matches_source_declarations": True})
print(json.dumps({"modules": rows, "scope": "SPIR-V declarations and natural type alignment; not driver private allocation statistics"}, indent=2))
