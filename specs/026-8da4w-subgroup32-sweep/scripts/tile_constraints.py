"""Shared tile/subgroup constraint model for the 8da4w subgroup32-reopen sweep.

Extends specs/025-8da4w-parameter-sweep/scripts/tile_constraints.py: same
shared-memory/register formulas (this shader's own layout, not 4w's), but
SUBGROUP_SIZE is no longer hard-fixed at 64 (research.md Decision 1). Legality of
subgroup_size=32 candidates is determined by real on-device compile/pipeline-creation
evidence (a LegalityProbeResult map, data-model.md), not assumed true or false here.
"""

MMA_M = 16
MMA_N = 16
MMA_K = 16
MAX_SHARED_MEM_BYTES = 65536
MAX_WG_INVOCATIONS = 1024
LEGAL_SUBGROUP_SIZES = (
    32,
    64,
)  # research.md Decision 1 -- both are real candidates now
MIN_WG_SIZE = 128  # 022's/025's minimum-parallelism floor, reused as-is


def token(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size):
    return f"tsweep_t{wg_tile_m}x{wg_tile_n}k{wg_tile_k}g{sg_grid_x}{sg_grid_y}s{subgroup_size}"


def derive(
    wg_tile_m,
    wg_tile_n,
    wg_tile_k,
    sg_grid_x,
    sg_grid_y,
    subgroup_size,
    group_size=128,
    legality_probe=None,
):
    """Compute derived properties and validity for one ConfigurationCandidate.

    `legality_probe`: optional dict of {candidate_token: LegalityProbeResult-like
    dict with 'compile_status'/'pipeline_creation_crashed'} from
    subgroup32_legality.json (data-model.md LegalityProbeResult). If a
    subgroup_size=32 candidate's token appears there with
    compile_status == "compile_failed", it is marked invalid with that evidence
    cited -- never assumed invalid purely from subgroup_size == 32 (the whole
    point of this feature vs. specs/025's tile_constraints.py).

    Returns a dict matching data-model.md's ConfigurationCandidate shape.
    """
    num_subgroups = sg_grid_x * sg_grid_y
    wg_size = num_subgroups * subgroup_size

    sg_tile_m = wg_tile_m / sg_grid_y if sg_grid_y else 0
    sg_tile_n = wg_tile_n / sg_grid_x if sg_grid_x else 0

    reasons = []

    if subgroup_size not in LEGAL_SUBGROUP_SIZES:
        reasons.append(
            f"subgroup_size={subgroup_size} not in {LEGAL_SUBGROUP_SIZES} -- "
            "not a legal Vulkan subgroup size on this hardware"
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

    k_blocks_per_chunk = wg_tile_k // 4
    b_total_slots = k_blocks_per_chunk * wg_tile_n
    b_slots_per_thread = b_total_slots // wg_size if wg_size else 0
    if b_slots_per_thread < 1:
        reasons.append(
            f"B_SLOTS_PER_THREAD={b_slots_per_thread} (B_TOTAL_SLOTS={b_total_slots}, "
            f"WG_SIZE={wg_size}) -- zero-sized temp_B array, confirmed glslc failure "
            "(specs/025 T015-T017 precedent)"
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

    tok = token(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size)

    # Real on-device evidence overrides pure arithmetic legality for subgroup_size=32
    # (research.md Decision 1) -- this is the mechanism that stops this module from
    # ever silently assuming a crash that this session already showed is stale.
    compile_status = "not_attempted"
    if subgroup_size == 32 and legality_probe is not None:
        probe = legality_probe.get(tok)
        if probe is not None:
            compile_status = probe.get("compile_status", "not_attempted")
            if compile_status == "compile_failed" or probe.get(
                "pipeline_creation_crashed"
            ):
                reasons.append(
                    f"LegalityProbeResult for {tok}: compile_status={compile_status}, "
                    f"pipeline_creation_crashed={probe.get('pipeline_creation_crashed')}"
                )

    valid = len(reasons) == 0

    return {
        "wg_tile_m": wg_tile_m,
        "wg_tile_n": wg_tile_n,
        "wg_tile_k": wg_tile_k,
        "sg_grid_x": sg_grid_x,
        "sg_grid_y": sg_grid_y,
        "subgroup_size": subgroup_size,
        "token": tok,
        "wg_size": wg_size,
        "lds_bytes": lds_bytes,
        "accumulators_per_sg": accumulators_per_sg,
        "valid": valid,
        "invalid_reasons": reasons,
        "compile_status": compile_status,
    }


def below_min_parallelism(candidate):
    """022's/025's minimum-parallelism floor: WG_SIZE < 128 is never top-ranked
    (still appears in the full ranking, per spec FR-009/SC-005 auditability)."""
    return candidate["wg_size"] < MIN_WG_SIZE
