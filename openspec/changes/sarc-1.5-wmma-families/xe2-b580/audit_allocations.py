#!/usr/bin/env python3
"""Find definite image-limit violations before constructing any model resource."""
import collections
import hashlib
import json
import mmap
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
from inspect_models import Table

models = Path(sys.argv[1])
properties = Path(sys.argv[2])
query = json.loads(properties.read_text())
matching = [row for row in query["formats"] if row.get("format") == "rgba16f"
            and row.get("image_type") == 2 and row.get("tiling") == 1
            and row.get("usage") == 15]
assert len(matching) == 1 and matching[0]["rc"] == 0
limits = matching[0]["maxExtent"]
results = []
for path in sorted(models.glob("*/exported/*vulkan*.pte")):
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data.find(b"VH00", 0, 2 * 1024 * 1024) - 4
        assert header >= 0
        offset, _ = struct.unpack_from("<II", data, header + 10)
        start = header + offset
        assert data[start + 4:start + 8] == b"VK00"
        graph = Table(data, start + struct.unpack_from("<I", data, start)[0])
        calls = [(op.string(1), op.vector(2)) for op in graph.vector(1, tables=True)]
        assert not any("conv2d" in name for name, _ in calls)
        images = []
        violations = []
        for index, value in enumerate(graph.vector(2, tables=True)):
            if value.scalar(0, "B") != 5:
                continue
            tensor = value.table(1)
            if tensor.scalar(2, "i") >= 0 or tensor.scalar(4, "B", 255) != 1:
                continue
            assert tensor.scalar(5, "B", 255) == 0, "non-width layout needs separate audit"
            dims = tensor.vector(1, "I")
            assert 1 <= len(dims) <= 4
            padded = [1] * (4 - len(dims)) + dims
            padded[-1] = (padded[-1] + 3) // 4 * 4
            extents = [padded[3] // 4, padded[2], padded[1] * padded[0]]
            images.append({"value_id": index, "dims": dims, "extents": extents})
            if any(size > limit for size, limit in zip(extents, limits)):
                assert tensor.scalar(0, "b") == 4, "different format needs matching query"
                violations.append({"value_id": index, "dims": dims, "extents": extents,
                                   "mem_obj_id": tensor.scalar(3, "i"),
                                   "calls": [{"op": name, "args": args} for name, args in calls
                                             if index in args]})
        counts = collections.Counter(tuple(row["extents"]) for row in violations)
        results.append({"model": path.name, "path": str(path), "bytes": path.stat().st_size,
                        "serialized_width_packed_images": len(images),
                        "violating_images": len(violations),
                        "violating_extents": [{"extents": list(key), "count": count}
                                              for key, count in sorted(counts.items())],
                        "violations": violations,
                        "status": "BLOCKED" if violations else "NO_SERIALIZED_VIOLATION_FOUND"})
assert len(results) == 6
print(json.dumps({"query_path": str(properties),
                  "query_sha256": hashlib.sha256(properties.read_bytes()).hexdigest(),
                  "matching_query": matching[0], "image_flags": 0, "image_pNext": None,
                  "axis_map": [0, 1, 2, 2], "models": results,
                  "scope": "definite serialized-resource violations; not a proof for operator temporaries"},
                 indent=2))
