#!/usr/bin/env python3
"""Enumerate the legal 8da4w (dq8ca/q4gsw) tile/subgroup configuration space.

Per research.md Decision 1: subgroup_size is fixed at 64 (32 crashes the Xclipse PAL
compiler for int8 WMMA) -- this is NOT 022's 642-candidate 4w enumeration re-used, it's a
fresh derivation using tile_constraints.py's 8da4w-specific lds_bytes/accumulators_per_sg
formula. Loop structure is out of scope here (fixed at dbuf2, per dbuf_reconfirmation.json)
-- this script only varies tile shape / subgroup grid.
"""
import argparse
import json
import sys

import tile_constraints as tc

WG_TILE_MN = [16, 32, 64, 128, 256]
WG_TILE_K = [16, 32, 64, 128]  # must be a multiple of MMA_K=16
SG_GRID = [1, 2, 4, 8]
SUBGROUP_SIZE = 64  # research.md Decision 1 -- the only value ever enumerated


def enumerate_all(group_size=128):
    seen_tokens = set()
    out = []
    for m in WG_TILE_MN:
        for n in WG_TILE_MN:
            for k in WG_TILE_K:
                for sgx in SG_GRID:
                    for sgy in SG_GRID:
                        c = tc.derive(
                            m, n, k, sgx, sgy, SUBGROUP_SIZE, group_size=group_size
                        )
                        if not c["valid"]:
                            continue
                        if c["token"] in seen_tokens:
                            continue
                        seen_tokens.add(c["token"])
                        del c["invalid_reasons"]
                        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop-structure", default="dbuf2")
    ap.add_argument(
        "--group-size", type=int, default=32
    )  # this workstream's real 8da4w production group_size
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    configs = enumerate_all(group_size=args.group_size)
    with open(args.out, "w") as f:
        json.dump(configs, f, indent=2)

    subgroup32 = [c for c in configs if c["subgroup_size"] == 32]
    assert not subgroup32, "enumeration must never emit subgroup_size=32 candidates"
    shipped_present = any(c["token"] == "tsweep_t128x64k32g22s64" for c in configs)

    print(f"loop_structure={args.loop_structure} (fixed, not swept)", file=sys.stderr)
    print(f"total_valid_universe={len(configs)}", file=sys.stderr)
    print(f"shipped_config_present={shipped_present}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
