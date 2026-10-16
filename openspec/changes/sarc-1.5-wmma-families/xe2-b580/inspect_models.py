#!/usr/bin/env python3
"""Read serialized metadata without loading methods or executing delegates."""
import json
import mmap
from pathlib import Path
import struct
import sys


class Table:
    def __init__(self, data, offset):
        self.data = data
        self.offset = offset

    def field(self, index):
        vt = self.offset - struct.unpack_from("<i", self.data, self.offset)[0]
        size = struct.unpack_from("<H", self.data, vt)[0]
        if 4 + 2 * index >= size:
            return 0
        offset = struct.unpack_from("<H", self.data, vt + 4 + 2 * index)[0]
        return self.offset + offset if offset else 0

    def scalar(self, index, fmt="i", default=0):
        offset = self.field(index)
        return struct.unpack_from("<" + fmt, self.data, offset)[0] if offset else default

    def indirect(self, index):
        offset = self.field(index)
        return offset + struct.unpack_from("<I", self.data, offset)[0] if offset else 0

    def table(self, index):
        return Table(self.data, self.indirect(index))

    def string(self, index):
        offset = self.indirect(index)
        size = struct.unpack_from("<I", self.data, offset)[0]
        return self.data[offset + 4:offset + 4 + size].decode()

    def vector(self, index, fmt="i", tables=False):
        offset = self.indirect(index)
        if not offset:
            return []
        count = struct.unpack_from("<I", self.data, offset)[0]
        offset += 4
        if tables:
            return [Table(self.data, offset + 4 * i + struct.unpack_from("<I", self.data, offset + 4 * i)[0]) for i in range(count)]
        return list(struct.unpack_from("<" + fmt * count, self.data, offset))


def inspect(path):
    with path.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            assert data[4:8] == b"ET12"
            program = Table(data, struct.unpack_from("<I", data)[0])
            constants = {}
            operators = []
            for plan in program.vector(1, tables=True):
                values = plan.vector(2, tables=True)
                outputs = plan.vector(4)
                name = plan.string(0)
                if name != "forward":
                    decoded = []
                    for index in outputs:
                        value = values[index]
                        kind = value.scalar(0, "B")
                        fmt = {2: "q", 3: "?", 4: "d"}.get(kind)
                        decoded.append(value.table(1).scalar(0, fmt) if fmt else {"type": kind})
                    constants[name] = decoded
                for op in plan.vector(6, tables=True):
                    operators.append(op.string(0))
            header = data.find(b"VH00", 0, 2 * 1024 * 1024) - 4
            assert header >= 0
            fb_offset, fb_size = struct.unpack_from("<II", data, header + 10)
            graph_start = header + fb_offset
            assert data[graph_start + 4:graph_start + 8] == b"VK00"
            graph = Table(data, graph_start + struct.unpack_from("<I", data, graph_start)[0])
            vk_values = graph.vector(2, tables=True)
            calls = []
            for op in graph.vector(1, tables=True):
                name = op.string(1)
                if "linear" not in name and "embedding" not in name:
                    continue
                args = []
                for index in op.vector(2):
                    value = vk_values[index]
                    kind = value.scalar(0, "B")
                    payload = value.table(1)
                    if kind == 2:
                        args.append(payload.scalar(0, "q"))
                    elif kind == 5:
                        args.append({"tensor": payload.vector(1, "I"), "dtype": payload.scalar(0, "b"),
                                     "storage": payload.scalar(4, "B", 255), "layout": payload.scalar(5, "B", 255)})
                    elif kind == 1:
                        args.append(None)
                    else:
                        args.append({"type": kind})
                calls.append({"op": name, "args": args})
            return {"path": str(path), "bytes": path.stat().st_size, "constants": constants,
                    "cpu_operators": sorted(set(operators)), "graph_storage_override": graph.scalar(7, "B", 255),
                    "graph_layout_override": graph.scalar(8, "B", 255), "linear_embedding_calls": calls}


if __name__ == "__main__":
    paths = sorted(Path(sys.argv[1]).glob("*/exported/*vulkan*.pte"))
    assert len(paths) == 6
    print(json.dumps([inspect(path) for path in paths], indent=2))
