"""Check original graph resources and operator-created allocation bounds."""

import json
import math
from pathlib import Path
import re
import sys


models = json.loads(Path(sys.argv[1]).read_text())
formats = json.loads(Path(sys.argv[2]).read_text())["formats"]
device_text = Path(sys.argv[3]).read_text().split("GPU1:", 1)[0]
buffer_limit = int(re.search(r"maxStorageBufferRange\s*=\s*(\d+)", device_text)[1])
format_for_dtype = {2: "R8G8B8A8_SINT", 3: "R32G32B32A32_SINT", 4: "R16G16B16A16_SFLOAT"}
element_bytes = {1: 1, 2: 1, 3: 4, 4: 2}
rows = []
for model in models:
    resources = []
    assert len(model["graphs"]) == 1
    graph = model["graphs"][0]
    assert graph["storage_override"] == graph["layout_override"] == 255

    def add_resource(label, dims, dtype, storage):
        assert 0 < len(dims) <= 4 and all(d > 0 for d in dims), (label, dims)
        padded = [1] * (4 - len(dims)) + dims
        n, channels, height, width = padded
        if storage == 0:
            # Round every dimension up to four: a conservative bound for either
            # scalar or vectorized width/4h4w buffers used by these operators.
            size = math.prod((d + 3) // 4 * 4 for d in padded) * element_bytes[dtype]
            assert size <= buffer_limit, (label, size, buffer_limit)
            resources.append({"label": label, "storage": "buffer", "bytes_upper_bound": size})
        else:
            extent = [(width + 3) // 4, height, n * channels]
            image_type = 2 if storage == 1 else 1
            feature = next(f for f in formats if f["format"] == format_for_dtype[dtype] and f["image_type"] == image_type)
            assert all(d <= cap for d, cap in zip(extent, feature["max_extent"])), (label, extent, feature)
            if image_type == 1:
                assert extent[2] == 1
            size = math.prod(extent) * 4 * element_bytes[dtype]
            assert size <= feature["max_resource_bytes"], (label, size)
            resources.append({"label": label, "storage": "image", "format": feature["format"],
                              "image_type": image_type, "extent": extent, "bytes": size})

    for index, value in enumerate(graph["values"]):
        if not isinstance(value, dict) or "dims" not in value or value["constant"] >= 0:
            continue
        assert value["layout"] in [0, 255]
        storage = 1 if value["storage"] == 255 else value["storage"]
        add_resource(f"serialized value {index}", value["dims"], value["datatype"], storage)

    constant_ops = {"et_vk.prepack.default", "et_vk.rms_norm.default", "et_vk.embedding_q4gsw.default",
                    "et_vk.linear_q4gsw.default", "et_vk.linear_dq8ca_q4gsw.default", "sdpa_with_kv_cache.default"}
    for index, call in enumerate(graph["calls"]):
        args = call["args"]
        name = call["name"]
        constants = [a for a in args if isinstance(a, dict) and a.get("constant", -1) >= 0]
        if constants:
            assert name in constant_ops, name
        label = f"operator {index} {name}"
        if name == "et_vk.rms_norm.default":
            add_resource(label + " weight", args[1]["dims"], 4, args[0]["storage"])
        elif name == "et_vk.embedding_q4gsw.default":
            assert args[4] is True
            n, packed_k = args[0]["dims"]
            add_resource(label + " packed weight", [(n + 7) // 8, ((packed_k * 2 + 3) // 4) * 4], 3, 2)
            add_resource(label + " scales", args[1]["dims"], 4, 0)
        elif name in ["et_vk.linear_q4gsw.default", "et_vk.linear_dq8ca_q4gsw.default"]:
            weight = args[1] if name == "et_vk.linear_q4gsw.default" else args[3]
            n, packed_k = weight["dims"]
            add_resource(label + " packed weight", [(n + 7) // 8, ((packed_k * 2 + 3) // 4) * 4], 3, 2)
            for constant in constants:
                add_resource(label + " constant buffer bound", constant["dims"], constant["datatype"], 0)
            if name == "et_vk.linear_dq8ca_q4gsw.default":
                add_resource(label + " packed int8 bound", args[0]["dims"], 3, 0)
                add_resource(label + " sums bound", args[0]["dims"], 3, 0)
        elif name == "sdpa_with_kv_cache.default":
            assert args[0]["storage"] == 0
            for cache in args[3:5]:
                add_resource(label + " cache and fused-copy bound", cache["dims"], 4, 0)
            # SDPA.cpp reserves two [1,H,S,context] buffers; the fused K/V
            # copies are smaller than the cache bounds above.
            heads = args[0]["dims"][-2]
            sequence = args[0]["dims"][-3]
            context = args[3]["dims"][-3]
            for _ in range(2):
                add_resource(label + " attention buffer", [1, heads, sequence, context], 4, 0)

    images = [r for r in resources if r["storage"] == "image"]
    rows.append({"model": Path(model["path"]).name, "resource_checks": resources,
                 "image_count": len(images), "buffer_count": len(resources) - len(images),
                 "max_image_axes": [max(r["extent"][axis] for r in images) for axis in range(3)],
                 "rope_image_extents": sorted({tuple(r["extent"]) for r in images if r["extent"][1] == 5120}),
                 "status": "PASS_PER_RESOURCE_LIMITS",
                 "scope": "serialized tensors and source-derived operator-created bounds; not aggregate peak memory or performance"})
print(json.dumps({"models": rows}, indent=2))
