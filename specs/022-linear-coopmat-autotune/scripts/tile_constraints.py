"""Shared tile-geometry constraint model for the linear_q4gsw_coopmat_tsweep
shader family (buffer weight storage, dbuf1 loop shape only).

Validated against 10 real on-device results on the M5 EVT1 (Samsung Exynos
2500 / Xclipse 970) collected in this workstream's session of 2026-07-07 --
see specs/022-linear-coopmat-autotune/results/known-measurements.json and
research.md. In particular this model correctly predicts that
tsweep_t128x64k16g44s32 fails to compile (B_PASSES divides to zero) while
every other real-tested config compiles, which is the ground-truth check
used to validate this module (see enumerate_configs.py).
"""

from dataclasses import dataclass

MMA = 16
GROUP_SIZE = 128
LDS_LIMIT_BYTES = 65536  # confirmed via test_coopmat_probe on M5 EVT1: maxComputeSharedMemorySize
WG_INVOCATION_LIMIT = 1024  # confirmed via test_coopmat_probe: maxComputeWorkGroupInvocations
FP16_PER_VEC4 = 8


@dataclass(frozen=True)
class TileConfig:
    wg_tile_m: int
    wg_tile_n: int
    wg_tile_k: int
    sg_grid_x: int
    sg_grid_y: int
    subgroup_size: int

    @property
    def token(self) -> str:
        return (
            f"tsweep_t{self.wg_tile_m}x{self.wg_tile_n}"
            f"k{self.wg_tile_k}g{self.sg_grid_x}{self.sg_grid_y}"
            f"s{self.subgroup_size}"
        )

    @property
    def wg_size(self) -> int:
        return self.sg_grid_x * self.sg_grid_y * self.subgroup_size

    @property
    def lds_bytes(self) -> int:
        a_stride_vec4 = (self.wg_tile_k + FP16_PER_VEC4) // FP16_PER_VEC4
        b_stride_vec4 = (self.wg_tile_n + FP16_PER_VEC4) // FP16_PER_VEC4
        ash = self.wg_tile_m * a_stride_vec4
        bsh = self.wg_tile_k * b_stride_vec4
        return 2 * (ash + bsh) * 16  # uvec4 = 16 bytes, double-buffered

    @property
    def accumulators_per_sg(self) -> int:
        if self.wg_tile_m % self.sg_grid_y != 0 or self.wg_tile_n % self.sg_grid_x != 0:
            return 0
        sg_tile_m = self.wg_tile_m // self.sg_grid_y
        sg_tile_n = self.wg_tile_n // self.sg_grid_x
        if sg_tile_m % MMA != 0 or sg_tile_n % MMA != 0:
            return 0
        return (sg_tile_m // MMA) * (sg_tile_n // MMA)

    def is_valid(self) -> bool:
        """Mirrors the four constraints validated against real hardware
        this session: thread-count limit, exact-division MMA-alignment,
        positive-integer staging-pass counts (A_PASSES/B_PASSES), and the
        confirmed 64KB LDS limit."""
        wg_size = self.wg_size
        if wg_size > WG_INVOCATION_LIMIT:
            return False
        if self.wg_tile_m % self.sg_grid_y != 0 or self.wg_tile_n % self.sg_grid_x != 0:
            return False
        sg_tile_m = self.wg_tile_m // self.sg_grid_y
        sg_tile_n = self.wg_tile_n // self.sg_grid_x
        if sg_tile_m % MMA != 0 or sg_tile_n % MMA != 0:
            return False
        invs_per_row_a = self.wg_tile_k // FP16_PER_VEC4
        invs_per_row_b = self.wg_tile_n // FP16_PER_VEC4
        if invs_per_row_a == 0 or invs_per_row_b == 0:
            return False
        if wg_size % invs_per_row_a != 0 or wg_size % invs_per_row_b != 0:
            return False
        a_rows_per_pass = wg_size // invs_per_row_a
        b_rows_per_pass = wg_size // invs_per_row_b
        if a_rows_per_pass == 0 or b_rows_per_pass == 0:
            return False
        if self.wg_tile_m % a_rows_per_pass != 0 or self.wg_tile_k % b_rows_per_pass != 0:
            return False
        if self.lds_bytes > LDS_LIMIT_BYTES:
            return False
        return True


def group_size_compatible(wg_tile_k: int, group_size: int = GROUP_SIZE) -> bool:
    """WG_TILE_K must divide (or be divided by) the quantization group_size
    so a K-tile never straddles a scale-group boundary mid-iteration."""
    return group_size % wg_tile_k == 0 or wg_tile_k % group_size == 0
