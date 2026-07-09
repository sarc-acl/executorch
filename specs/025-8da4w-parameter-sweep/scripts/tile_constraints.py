"""Shared tile/subgroup constraint model for the 8da4w (dq8ca/q4gsw) coopmat sweep.

Formula derived directly from linear_dq8ca_qw_coopmat.glsl's shared-memory layout
(Ash_int8/Bsh_int8 double-buffered staging + izp_sh/ifs_sh/wsum_sh/wsc_sh broadcast
arrays) -- NOT a reuse of 022's linear_qw_coopmat.glsl (4w) formula, per research.md
Decision 1/2: this shader carries a larger footprint (WG_TILE_K=32 default vs 4w's 16,
plus the extra broadcast arrays, plus a second int32 accumulator array alongside the
fp32 one).

SUBGROUP_SIZE is fixed at 64 for every candidate this module considers valid: the
shipped shader's own header comment records that the Xclipse PAL compiler crashes in
vkCreateComputePipelines when int8 WMMA is compiled at forced subgroup size 32 (fp16
WMMA at 32 is fine). A subgroup_size=32 input is therefore always marked invalid here,
not filtered by a caller.
"""

MMA_M = 16
MMA_N = 16
MMA_K = 16
MAX_SHARED_MEM_BYTES = 65536
MAX_WG_INVOCATIONS = 1024
VALID_SUBGROUP_SIZE = (
    64  # research.md Decision 1 -- the only legal value for this shader
)
MIN_WG_SIZE = 128  # 022's minimum-parallelism floor (Decision 2 note), reused as-is


def token(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size):
    return f"tsweep_t{wg_tile_m}x{wg_tile_n}k{wg_tile_k}g{sg_grid_x}{sg_grid_y}s{subgroup_size}"


def derive(
    wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size, group_size=128
):
    """Compute derived properties and validity for one ConfigurationCandidate.

    Returns a dict matching data-model.md's ConfigurationCandidate shape (token,
    wg_size, lds_bytes, accumulators_per_sg, valid), plus the inputs echoed back.
    """
    num_subgroups = sg_grid_x * sg_grid_y
    wg_size = num_subgroups * subgroup_size

    sg_tile_m = wg_tile_m / sg_grid_y if sg_grid_y else 0
    sg_tile_n = wg_tile_n / sg_grid_x if sg_grid_x else 0

    reasons = []

    if subgroup_size != VALID_SUBGROUP_SIZE:
        reasons.append(
            f"subgroup_size={subgroup_size} crashes Xclipse PAL for int8 WMMA (research.md Decision 1)"
        )
    if wg_size > MAX_WG_INVOCATIONS:
        reasons.append(
            f"wg_size={wg_size} exceeds maxComputeWorkGroupInvocations={MAX_WG_INVOCATIONS}"
        )
    if sg_tile_m % MMA_M != 0 or sg_tile_n % MMA_N != 0:
        reasons.append("sg_tile_m/n not divisible by MMA_M/N -- MMA-alignment violated")
    if wg_tile_k % MMA_K != 0:
        reasons.append("wg_tile_k not divisible by MMA_K")
    if group_size % wg_tile_k != 0:
        reasons.append(f"wg_tile_k={wg_tile_k} does not divide group_size={group_size}")

    # B-staging pass count (K_BLOCKS_PER_CHUNK * WG_TILE_N / WG_SIZE) must be a
    # positive integer, or the shader's temp_B[B_SLOTS_PER_THREAD] array is
    # zero-sized -- a real glslc compile failure confirmed during T015-T017
    # (tsweep_t{32,64,128}x16k16g12s64 all failed this way), not merely a
    # theoretical constraint.
    k_blocks_per_chunk = wg_tile_k // 4
    b_total_slots = k_blocks_per_chunk * wg_tile_n
    b_slots_per_thread = b_total_slots // wg_size if wg_size else 0
    if b_slots_per_thread < 1:
        reasons.append(
            f"B_SLOTS_PER_THREAD={b_slots_per_thread} (B_TOTAL_SLOTS={b_total_slots}, "
            f"WG_SIZE={wg_size}) -- zero-sized temp_B array, confirmed glslc failure"
        )

    mmas_per_sg_m = int(sg_tile_m // MMA_M) if sg_tile_m % MMA_M == 0 else 0
    mmas_per_sg_n = int(sg_tile_n // MMA_N) if sg_tile_n % MMA_N == 0 else 0
    accumulators_per_sg = mmas_per_sg_m * mmas_per_sg_n

    num_k_slabs = wg_tile_k // MMA_K if wg_tile_k % MMA_K == 0 else 0
    a_slab_int8 = wg_tile_m * MMA_K
    a_slab_u32 = a_slab_int8 // 4
    ash_slice_u32 = num_k_slabs * a_slab_u32
    ash_bytes = 2 * ash_slice_u32 * 4

    b_useful_u32 = MMA_K // 4
    b_stride_u32 = b_useful_u32 + 1  # +1 skew, per shader comment
    b_slab_u32 = wg_tile_n * b_stride_u32
    bsh_slice_u32 = num_k_slabs * b_slab_u32
    bsh_bytes = 2 * bsh_slice_u32 * 4

    izp_bytes = wg_tile_m * 4
    ifs_bytes = wg_tile_m * 4
    wsum_bytes = 2 * wg_tile_n * 4
    wsc_bytes = 2 * wg_tile_n * 4

    lds_bytes = ash_bytes + bsh_bytes + izp_bytes + ifs_bytes + wsum_bytes + wsc_bytes

    if lds_bytes > MAX_SHARED_MEM_BYTES:
        reasons.append(
            f"lds_bytes={lds_bytes} exceeds maxComputeSharedMemorySize={MAX_SHARED_MEM_BYTES}"
        )

    valid = len(reasons) == 0

    return {
        "wg_tile_m": wg_tile_m,
        "wg_tile_n": wg_tile_n,
        "wg_tile_k": wg_tile_k,
        "sg_grid_x": sg_grid_x,
        "sg_grid_y": sg_grid_y,
        "subgroup_size": subgroup_size,
        "token": token(
            wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size
        ),
        "wg_size": wg_size,
        "lds_bytes": lds_bytes,
        "accumulators_per_sg": accumulators_per_sg,
        "valid": valid,
        "invalid_reasons": reasons,
        "compile_status": "not_attempted",
    }


def below_min_parallelism(candidate):
    """022's minimum-parallelism floor: WG_SIZE < 128 is never top-ranked (still
    appears in the full ranking, per spec FR-009/SC-005 auditability)."""
    return candidate["wg_size"] < MIN_WG_SIZE
