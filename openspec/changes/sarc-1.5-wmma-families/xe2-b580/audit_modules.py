#!/usr/bin/env python3
"""Audit declared Workgroup storage in the exact compiled candidate modules."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
limit = int(sys.argv[3])
rows = []
for candidate in manifest["candidates"]:
    path = root / (candidate["shader"] + ".spv")
    data = path.read_bytes()
    disassembly = subprocess.check_output(["spirv-dis", "--raw-id", str(path)], text=True)
    definitions = {}
    strides = {}
    workgroup = []
    for line in disassembly.splitlines():
        words = line.split()
        if len(words) >= 4 and words[1] == "=":
            definitions[words[0]] = words[2:]
            if words[2] == "OpVariable" and words[4] == "Workgroup":
                workgroup.append(words[3])
        if len(words) == 4 and words[0] == "OpDecorate" and words[2] == "ArrayStride":
            strides[words[1]] = int(words[3])

    def size_align(type_id):
        op, *args = definitions[type_id]
        if op in ("OpTypeInt", "OpTypeFloat"):
            size = int(args[0]) // 8
            return size, size
        if op == "OpTypeVector":
            size, align = size_align(args[0])
            count = int(args[1])
            return size * count, align * (4 if count == 3 else count)
        if op == "OpTypeArray":
            size, align = size_align(args[0])
            constant = definitions[args[1]]
            assert constant[0] == "OpConstant", constant
            count = int(constant[2])
            stride = strides.get(type_id, (size + align - 1) // align * align)
            return stride * count, align
        if op == "OpTypePointer":
            assert args[0] == "Workgroup"
            return size_align(args[1])
        raise ValueError(f"unsupported Workgroup declaration: {type_id} {op}")

    total = 0
    allocations = []
    for pointer in workgroup:
        size, align = size_align(pointer)
        total = (total + align - 1) // align * align
        allocations.append({"pointer_type": pointer, "bytes": size, "alignment": align})
        total += size
    assert total <= limit, (path.name, total, limit)
    assert total <= candidate["declared_shared_bytes"], (path.name, total, candidate["declared_shared_bytes"])
    matrix_types = [line.strip() for line in disassembly.splitlines() if "OpTypeCooperativeMatrixKHR" in line]
    assert matrix_types, f"missing cooperative matrix in {path}"
    rows.append({"shader": candidate["shader"], "sha256": hashlib.sha256(data).hexdigest(),
                 "spirv_version": int.from_bytes(data[4:8], "little"),
                 "workgroup_declared_bytes": total, "source_declared_bytes": candidate["declared_shared_bytes"],
                 "allocations": allocations, "matrix_types": matrix_types,
                 "capabilities": re.findall(r"OpCapability (\S+)", disassembly),
                 "physical_driver_allocation": "not reported by module; declaration bound only"})
print(json.dumps(rows, indent=2))
