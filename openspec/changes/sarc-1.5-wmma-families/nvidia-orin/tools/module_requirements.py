#!/usr/bin/env python3
"""Record requirements of exact manifest SPIR-V modules without GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def inspect(module, output):
    disassembly = subprocess.check_output(["spirv-dis", "--raw-id", str(module)], text=True)
    output.write_text(disassembly)
    definitions = {}
    strides = {}
    capabilities = []
    extensions = []
    for line in disassembly.splitlines():
        fields = line.split()
        if "=" in fields:
            definitions[fields[0]] = fields[2:]
        elif fields[:1] == ["OpDecorate"] and fields[2:3] == ["ArrayStride"]:
            strides[fields[1]] = int(fields[3])
        elif fields[:1] == ["OpCapability"]:
            capabilities.append(fields[1])
        elif fields[:1] == ["OpExtension"]:
            extensions.append(fields[1].strip('"'))

    def constant(identifier):
        op, *args = definitions[identifier]
        if op != "OpConstant" or definitions[args[0]][0] != "OpTypeInt":
            raise ValueError(f"nonconstant integer {identifier}: {op}")
        return int(args[1], 0)

    def size(identifier):
        op, *args = definitions[identifier]
        if op in ("OpTypeFloat", "OpTypeInt"):
            return int(args[0]) // 8
        if op == "OpTypeVector":
            if int(args[1]) == 3:
                raise ValueError("three-component shared vector needs explicit alignment review")
            return size(args[0]) * int(args[1])
        if op == "OpTypeArray":
            return strides.get(identifier, size(args[0])) * constant(args[1])
        raise ValueError(f"unsupported shared type {identifier}: {op}")

    shared = []
    matrices = []
    images = []
    for identifier, (op, *args) in definitions.items():
        if op == "OpVariable" and args[1] == "Workgroup":
            pointer = definitions[args[0]]
            assert pointer[:2] == ["OpTypePointer", "Workgroup"]
            shared.append({"id": identifier, "type": pointer[2], "bytes": size(pointer[2])})
        if op == "OpTypeCooperativeMatrixKHR":
            matrices.append({"component": definitions[args[0]], "scope": constant(args[1]),
                             "rows": constant(args[2]), "columns": constant(args[3]),
                             "use": constant(args[4])})
        if op == "OpTypeImage":
            images.append({"component": definitions[args[0]], "operands": args[1:]})
    validation = subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(module)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return {"module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "capabilities": capabilities, "extensions": extensions,
            "shared_variables": shared, "shared_type_bytes": sum(item["bytes"] for item in shared),
            "matrix_types": matrices, "image_types": images,
            "spirv_val_rc": validation.returncode, "spirv_val_output": validation.stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("spirv_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for candidate in json.loads(args.manifest.read_text())["candidates"]:
        shader = candidate["shader"]
        entry = inspect(args.spirv_directory / (shader + ".spv"),
                        args.output_directory / (shader + ".spvasm"))
        entry.update(shader=shader, source_declared_bytes=candidate["declared_shared_bytes"])
        entry["declared_matches_module"] = entry["source_declared_bytes"] == entry["shared_type_bytes"]
        entries.append(entry)
    (args.output_directory / "requirements.json").write_text(json.dumps(entries, indent=2) + "\n")
    for entry in entries:
        print(entry["shader"], entry["shared_type_bytes"], entry["declared_matches_module"],
              "spirv-val", entry["spirv_val_rc"])


if __name__ == "__main__":
    main()
