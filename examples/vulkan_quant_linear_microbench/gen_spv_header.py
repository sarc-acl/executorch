#!/usr/bin/env python3
import sys


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <in.spv> <out.h> <c_identifier>", file=sys.stderr)
        sys.exit(1)
    spv_path, out_path, ident = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(spv_path, "rb") as f:
        data = f.read()
    assert (
        len(data) % 4 == 0
    ), f"{spv_path} is not a valid SPIR-V blob (size not multiple of 4)"
    with open(out_path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n#include <cstddef>\n")
        f.write(f"alignas(4) static const unsigned char {ident}_bytes[] = {{\n")
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            f.write("  " + ",".join(str(b) for b in chunk) + ",\n")
        f.write("};\n")
        f.write(
            f"static const uint32_t* {ident} = reinterpret_cast<const uint32_t*>({ident}_bytes);\n"
        )
        f.write(f"static const size_t {ident}_size = sizeof({ident}_bytes);\n")


if __name__ == "__main__":
    main()
