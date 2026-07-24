"""Device-parametrized tile/subgroup constraint model for both WMMA coopmat
tsweep shader families.

Ports specs/022 (linear_q4gsw_coopmat_tsweep: fp16 staging, dbuf1 loop) and
specs/025 (linear_dq8ca_q4gsw_coopmat_tsweep: int8 staging, dbuf2 loop) into
one module, with two deliberate changes for portability:

- Hardware limits (shared memory, workgroup invocations, supported subgroup
  sizes) are runtime parameters (DeviceLimits), not constants. Populate them
  from device_fingerprint.limits_from_fingerprint() on each new device.
- Device-specific restrictions are named quirks instead of baked-in rules.
  "no_int8_wmma_sg32": Xclipse PAL crashes vkCreateComputePipelines for int8
  WMMA at forced subgroup size 32 (specs/025 Decision 1) -- enable it there,
  leave it off on RADV where sg32 gates cleanly (specs/035 finding 2).

group_size is the quantization group size baked into the .pte being swept
(this box's buffer ptes: 128, see .export_scratch/run_buffer_exports.sh in the
dev worktree). It is always a required argument -- a wrong value silently
mislabels K-tile legality.
"""

from dataclasses import dataclass

MMA = 16
FP16_PER_VEC4 = 8

SHADERS = ("q4gsw", "dq8ca")

WG_TILE_MN_CHOICES = (16, 32, 64, 128, 256)
WG_TILE_K_CHOICES = (16, 32, 64, 128)
SG_GRID_CHOICES = (1, 2, 4, 8)

MIN_WG_SIZE = 128  # specs/022 minimum-parallelism floor; soft flag, not a rejection


@dataclass(frozen=True)
class DeviceLimits:
    max_shared_mem_bytes: int
    max_wg_invocations: int
    subgroup_sizes: tuple
    quirks: frozenset = frozenset()


def token(m, n, k, gx, gy, sub):
    return f"tsweep_t{m}x{n}k{k}g{gx}{gy}s{sub}"


def parse_token(tok):
    """Inverse of token(). Raises ValueError on anything malformed, mirroring
    (and pre-empting) the std::stoul throw in QuantizedLinear.cpp's
    parse_tsweep_tile."""
    if not tok.startswith("tsweep_t"):
        raise ValueError(f"not a tsweep token: {tok!r}")
    body = tok[len("tsweep_t") :]
    try:
        mn, rest = body.split("k", 1)
        m_s, n_s = mn.split("x", 1)
        k_s, rest = rest.split("g", 1)
        grid, sub_s = rest.split("s", 1)
        if len(grid) != 2:
            raise ValueError
        out = {
            "wg_tile_m": int(m_s),
            "wg_tile_n": int(n_s),
            "wg_tile_k": int(k_s),
            "sg_grid_x": int(grid[0]),
            "sg_grid_y": int(grid[1]),
            "subgroup_size": int(sub_s),
        }
    except ValueError:
        raise ValueError(f"malformed tsweep token: {tok!r}")
    if token(*out.values()) != tok:
        raise ValueError(f"non-canonical tsweep token: {tok!r}")
    return out


def lds_bytes_q4gsw(m, n, k):
    """fp16-staged, +1-vec4 skew, double-buffered (022 formula, validated
    against 10 on-device M5 results)."""
    a_stride_vec4 = (k + FP16_PER_VEC4) // FP16_PER_VEC4
    b_stride_vec4 = (n + FP16_PER_VEC4) // FP16_PER_VEC4
    ash = m * a_stride_vec4
    bsh = k * b_stride_vec4
    return 2 * (ash + bsh) * 16


def lds_bytes_dq8ca(m, n, k):
    """int8-staged A/B + izp/ifs/wsum/wsc broadcast arrays, double-buffered
    (025 formula, derived from linear_dq8ca_qw_coopmat.glsl's layout)."""
    num_k_slabs = k // MMA
    a_slab_u32 = (m * MMA) // 4
    ash_bytes = 2 * num_k_slabs * a_slab_u32 * 4
    b_stride_u32 = MMA // 4 + 1  # +1 skew
    bsh_bytes = 2 * num_k_slabs * n * b_stride_u32 * 4
    broadcast_bytes = m * 4 + m * 4 + 2 * n * 4 + 2 * n * 4
    return ash_bytes + bsh_bytes + broadcast_bytes


def derive(shader, m, n, k, gx, gy, sub, group_size, limits):  # noqa: C901
    """Full validity + derived properties for one candidate. Returns a dict
    with token, wg_size, lds_bytes, accumulators_per_sg, valid,
    invalid_reasons."""
    assert shader in SHADERS, shader
    reasons = []
    wg_size = gx * gy * sub

    if sub not in limits.subgroup_sizes:
        reasons.append(f"subgroup_size={sub} not supported by device")
    if shader == "dq8ca" and sub == 32 and "no_int8_wmma_sg32" in limits.quirks:
        reasons.append("quirk no_int8_wmma_sg32: int8 WMMA at sg32 crashes this driver")
    if wg_size > limits.max_wg_invocations:
        reasons.append(f"wg_size={wg_size} exceeds maxComputeWorkGroupInvocations")

    if m % gy != 0 or n % gx != 0:
        reasons.append("wg tile not divisible by subgroup grid")
        sg_tile_m = sg_tile_n = 0
    else:
        sg_tile_m = m // gy
        sg_tile_n = n // gx
        if sg_tile_m % MMA != 0 or sg_tile_n % MMA != 0:
            reasons.append("sg_tile_m/n not divisible by MMA=16")
    if k % MMA != 0:
        reasons.append("wg_tile_k not divisible by MMA=16")
    if group_size % k != 0 and k % group_size != 0:
        reasons.append(f"wg_tile_k={k} incompatible with group_size={group_size}")

    if shader == "q4gsw":
        # Staging-pass divisibility (A_PASSES/B_PASSES must be positive ints).
        invs_per_row_a = k // FP16_PER_VEC4
        invs_per_row_b = n // FP16_PER_VEC4
        if invs_per_row_a == 0 or invs_per_row_b == 0:
            reasons.append("tile row narrower than one vec4 of fp16")
        elif wg_size > 0 and (
            wg_size % invs_per_row_a != 0 or wg_size % invs_per_row_b != 0
        ):
            reasons.append("wg_size does not evenly cover staging rows")
        else:
            a_rows = wg_size // invs_per_row_a if invs_per_row_a else 0
            b_rows = wg_size // invs_per_row_b if invs_per_row_b else 0
            if a_rows == 0 or b_rows == 0 or m % a_rows != 0 or k % b_rows != 0:
                reasons.append("A_PASSES/B_PASSES not positive integers")
        lds = lds_bytes_q4gsw(m, n, k)
    else:
        # B_SLOTS_PER_THREAD >= 1: zero-sized temp_B is a confirmed glslc
        # failure (specs/025 T015-T017).
        b_total_slots = (k // 4) * n
        if wg_size == 0 or b_total_slots // wg_size < 1:
            reasons.append("B_SLOTS_PER_THREAD < 1 -- zero-sized temp_B, glslc failure")
        lds = lds_bytes_dq8ca(m, n, k)

    if lds > limits.max_shared_mem_bytes:
        reasons.append(f"lds_bytes={lds} exceeds maxComputeSharedMemorySize")

    acc = 0
    if sg_tile_m and sg_tile_m % MMA == 0 and sg_tile_n % MMA == 0:
        acc = (sg_tile_m // MMA) * (sg_tile_n // MMA)

    return {
        "shader": shader,
        "wg_tile_m": m,
        "wg_tile_n": n,
        "wg_tile_k": k,
        "sg_grid_x": gx,
        "sg_grid_y": gy,
        "subgroup_size": sub,
        "token": token(m, n, k, gx, gy, sub),
        "wg_size": wg_size,
        "lds_bytes": lds,
        "accumulators_per_sg": acc,
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


def derive_token(shader, tok, group_size, limits):
    return derive(
        shader,
        group_size=group_size,
        limits=limits,
        **{
            {
                "wg_tile_m": "m",
                "wg_tile_n": "n",
                "wg_tile_k": "k",
                "sg_grid_x": "gx",
                "sg_grid_y": "gy",
                "subgroup_size": "sub",
            }[key]: val
            for key, val in parse_token(tok).items()
        },
    )


def enumerate_legal(shader, limits, group_size):
    out = []
    for m in WG_TILE_MN_CHOICES:
        for n in WG_TILE_MN_CHOICES:
            for k in WG_TILE_K_CHOICES:
                for gx in SG_GRID_CHOICES:
                    for gy in SG_GRID_CHOICES:
                        for sub in limits.subgroup_sizes:
                            c = derive(shader, m, n, k, gx, gy, sub, group_size, limits)
                            if c["valid"]:
                                out.append(c)
    return out


def below_min_parallelism(candidate):
    return candidate["wg_size"] < MIN_WG_SIZE
