#!/usr/bin/env python3
"""Enumerate the full valid, buffer-storage-only tile-geometry universe for
linear_q4gsw_coopmat_tsweep (dbuf1 loop shape). See specs/022-linear-coopmat-autotune/
contracts/autotune-report-schema.md #1 for the output contract."""

import argparse
import json
from pathlib import Path

from tile_constraints import group_size_compatible, TileConfig

M_VALS = [16, 32, 64, 128, 256]
N_VALS = [16, 32, 64, 128, 256]
K_VALS = [8, 16, 32, 64, 128]
SGX_VALS = [1, 2, 4, 8]
SGY_VALS = [1, 2, 4, 8]
SUB_VALS = [32, 64]


def enumerate_valid_configs():
    configs = []
    for m in M_VALS:
        for n in N_VALS:
            for k in K_VALS:
                if not group_size_compatible(k):
                    continue
                for sgx in SGX_VALS:
                    for sgy in SGY_VALS:
                        for sub in SUB_VALS:
                            tc = TileConfig(m, n, k, sgx, sgy, sub)
                            if not tc.is_valid():
                                continue
                            configs.append(
                                {
                                    "token": tc.token,
                                    "wg_tile_m": tc.wg_tile_m,
                                    "wg_tile_n": tc.wg_tile_n,
                                    "wg_tile_k": tc.wg_tile_k,
                                    "sg_grid_x": tc.sg_grid_x,
                                    "sg_grid_y": tc.sg_grid_y,
                                    "subgroup_size": tc.subgroup_size,
                                    "wg_size": tc.wg_size,
                                    "lds_bytes": tc.lds_bytes,
                                    "accumulators_per_sg": tc.accumulators_per_sg,
                                    "valid": True,
                                    "compile_status": "not_attempted",
                                }
                            )
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    configs = enumerate_valid_configs()

    # De-duplicate by token (distinct (M,N,K,grid,sub) tuples always give
    # distinct tokens, so this is a sanity check, not expected to remove
    # anything).
    seen = set()
    deduped = []
    for c in configs:
        if c["token"] in seen:
            continue
        seen.add(c["token"])
        deduped.append(c)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deduped, indent=2))
    print(f"Wrote {len(deduped)} valid configs to {out_path}")


if __name__ == "__main__":
    main()
