#!/usr/bin/env python3
"""enum_space.py [--check <executorch tree>] [--list <family>]: the compile-time parameter space of the four
kernel families the 780M prefill uses, pruned statically for this device. GPU-free.

For each family: the number of combinations of the values the sweep yamls and sarc/SWEEP-PARAMETERS.md name, and
how many are left after each static rule, applied in this order (a combination is counted at the first rule it
fails):
  device   SUBGROUP_SIZE in the device range (32..64), an MMA shape the device exposes (16x16x16 only),
           workgroup <= 1024 invocations
  flags    the #error exclusions of the body, flags without effect in the model path (dedupe)
  geometry the grid divides the tile into whole MMAs; the staging thread maps come out exact
  lds      shared memory <= 65536 B (three 4w variants over it reset the GPU on 2026-10-03)
  shape    the tile divides every real prefill shape it would serve
The rules are read off the shader bodies (line references in STATUS.md); --check replays them on every variant
already in the sweep yamls that this device can run, which have all been built and dispatched.
IO is texture3d/texture2d for the linears (the model path) and buffer/buffer for SDPA."""
import collections, itertools, sys

TILE = (32, 64, 128, 256); KS = (16, 32, 64); GRID = (1, 2, 4, 8); SGS = (16, 32, 64)
LDS_MAX = 65536; WG_MAX = 1024; DEV_SG = (32, 64); DEV_MMA = (16, 16, 16)
# real prefill shapes (test_llama_microbench.cpp): M = 2048 tokens
LIN_N = (512, 1024, 2048, 3072, 4096, 8192, 14336); LIN_K = (2048, 3072, 4096, 8192, 14336); GROUP = 128
HEAD_DIM = (64, 128); SEQ = 2048

def geom(M, N, K, X, Y, S, mma):
    mm, mn, mk = mma
    if S not in DEV_SG or mma != DEV_MMA or X * Y * S > WG_MAX: return "device"
    if M % Y or N % X or (M // Y) % mm or (N // X) % mn or K % mk: return "geometry"
    return None

Q4_FLAGS = ("ACC_FP32", "ACC_GROUP_FP32", "ACC_GROUP_FP32_REG", "CSH_IN_ASH", "CSH_FULL", "CSH_POOL", "CSH_BAND",
            "FRAG_LAYOUT", "IMG_A", "IMG_W", "B_COLMAJOR", "SH_F16V4")

def q4_flags_ok(f, bx):
    a = sum(f[x] for x in ("ACC_FP32", "ACC_GROUP_FP32", "ACC_GROUP_FP32_REG"))
    if a > 1: return False
    if f["SH_F16V4"] and (f["CSH_IN_ASH"] or f["FRAG_LAYOUT"] or f["B_COLMAJOR"]): return False
    if f["CSH_POOL"] and (not f["CSH_FULL"] or f["CSH_IN_ASH"] or f["B_COLMAJOR"] or f["FRAG_LAYOUT"]): return False
    if f["B_COLMAJOR"] and f["FRAG_LAYOUT"]: return False
    if f["CSH_FULL"] and (f["CSH_BAND"] or f["CSH_IN_ASH"]): return False
    if f["CSH_BAND"] and (f["CSH_IN_ASH"] or f["ACC_GROUP_FP32_REG"]): return False
    if f["ACC_GROUP_FP32_REG"] and (f["CSH_IN_ASH"] or f["B_COLMAJOR"] or f["FRAG_LAYOUT"]): return False
    # no #error, but the declarations do not line up (Fsh aliases an undeclared Csh; the pool is uvec4 only)
    if f["CSH_POOL"] and (f["ACC_GROUP_FP32_REG"] or f["SH_F16V4"]): return False
    if bx and f["B_COLMAJOR"]: return False
    return True

def q4(M, N, K, X, Y, S, mma, f, bx):
    """4w linear, texture3d IO. Returns the first failed rule or None."""
    WG = X * Y * S; mm, mn, mk = mma
    ka = K // 8
    if K % 8 or ka > WG or WG % ka or M % (WG // ka): return "geometry"
    if N % 8: return "geometry"
    if not f["B_COLMAJOR"]:
        nb = N // 8
        if nb > WG or WG % nb: return "geometry"
        rb = WG // nb
        if K % rb: return "geometry"
        bp = K // rb
        if bx:
            if bp not in (1, 2, 4): return "geometry"
        elif bp != 1 and rb % 4: return "geometry"
    if f["FRAG_LAYOUT"] and (K % 16 or N % 16): return "geometry"
    if f["ACC_GROUP_FP32_REG"] and (mm * mn) % S: return "geometry"
    ash = 4 * M * K if f["FRAG_LAYOUT"] else 4 * M * (K + 8)
    if f["CSH_IN_ASH"] and 2 * Y * mm * N > ash: return "geometry"
    if f["CSH_POOL"]:
        lds = max(4 * M * (K + 8) + 4 * K * (N + 8), 2 * M * N)
    else:
        b = 4 * N * (K + 8) if f["B_COLMAJOR"] else (4 * K * N if f["FRAG_LAYOUT"] else 4 * K * (N + 8))
        rows = M if f["CSH_FULL"] else (mm if f["CSH_BAND"] else Y * mm)
        lds = ash + b + (0 if f["CSH_IN_ASH"] else 2 * rows * N)
    if lds > LDS_MAX: return "lds"
    if 2048 % M or any(n % N for n in LIN_N) or any(k % K for k in LIN_K) or GROUP % K: return "shape"
    return None

def zpg(M, N, K, X, Y, S, mma, full, multi, blocks, bt):
    """8da4w zpg linear, texture3d IO."""
    WG = X * Y * S; mm = mma[0]
    if not multi and blocks != 1: return "flags"          # A_BLOCKS is only read under A_MULTI_BLOCK
    if M % 4 or K % 4 or N % 8: return "geometry"
    need = M * K // 16
    if multi and blocks > 1:
        if need != blocks * WG: return "geometry"
    elif need > WG: return "geometry"
    if full and need != WG * blocks: return "geometry"
    slots = K * N // 16 if bt else (K // 4) * N
    if slots < WG or slots % WG: return "geometry"
    if WG < M // 4 or WG < N: return "geometry"
    if 2 * K * (M + N) + 8 * M + 12 * N + 2 * Y * mm * N > LDS_MAX: return "lds"
    if 2048 % M or any(n % N for n in LIN_N) or any(k % K for k in LIN_K) or GROUP % K: return "shape"
    return None

def qk(M, N, K, X, Y, S, mma, nf, pk):
    """SDPA QK^T: M = sequence rows, N = context columns, K = head_dim chunk."""
    if pk:
        if K % 8: return "geometry"
    elif K != 32: return "geometry"       # the chunk count is a spec constant from the release row (k = 32)
    b = 2 * N * (K + 8) if pk else 2 * K * (N + 8)
    if 2 * M * (K + 8) + b + 2 * M * N > LDS_MAX: return "lds"
    if SEQ % M or SEQ % N or any(h % K for h in HEAD_DIM): return "shape"
    return None

def av(M, N, K, X, Y, S, mma, ml, head_dim):
    """SDPA attn*V: M = sequence rows, N = head_dim columns, K = context chunk."""
    WG = X * Y * S
    if K % 8 or N % 8 or K < 32: return "geometry"  # K below the release row's 32 truncates the context
    if not ml and not (WG == M * K // 8 == K * N // 8): return "geometry"
    if 2 * M * (K + 8) + 2 * K * (N + 8) > LDS_MAX: return "lds"
    if SEQ % M or SEQ % K or head_dim % N: return "shape"
    return None

GEOMS = list(itertools.product(TILE, TILE, KS, GRID, GRID, SGS))

def space(family):
    """Yields (combination dict, first failed rule or None)."""
    if family == "4w":
        mmas = ((16, 16, 16), (8, 16, 16), (64, 32, 16), (16, 32, 32))
        fl = [dict(zip(Q4_FLAGS, v)) for v in itertools.product((False, True), repeat=len(Q4_FLAGS))]
        for g in GEOMS:
            for mma in mmas:
                r0 = geom(*g, mma)
                for bx in (False, True):
                    for f in fl:
                        r = r0 if r0 == "device" else ("flags" if not q4_flags_ok(f, bx) else r0 or q4(*g, mma, f, bx))
                        yield dict(g=g, mma=mma, f=f, bx=bx), r
    elif family == "8da4w":
        mmas = ((16, 16, 16), (8, 16, 32), (16, 16, 32))
        for g in GEOMS:
            for mma in mmas:
                r0 = geom(*g, mma)
                for full, multi, blocks, bt in itertools.product((False, True), (False, True), (1, 2, 4), (False, True)):
                    yield dict(g=g, mma=mma, full=full, multi=multi, blocks=blocks, bt=bt), r0 or zpg(*g, mma, full, multi, blocks, bt)
    elif family == "qk":
        for g in GEOMS:
            r0 = geom(*g, DEV_MMA)
            for nf, pk in itertools.product((False, True), repeat=2):
                yield dict(g=g, nf=nf, pk=pk), r0 or qk(*g, DEV_MMA, nf, pk)
    elif family == "av":
        for g in GEOMS:
            r0 = geom(*g, DEV_MMA)
            for ml in (False, True):
                # a tile is kept if it serves at least one head_dim; the per-shape lists are in --list
                rs = [r0 or av(*g, DEV_MMA, ml, h) for h in HEAD_DIM]
                yield dict(g=g, ml=ml, heads=[h for h, r in zip(HEAD_DIM, rs) if r is None]), (None if None in rs else rs[0])

def token(family, c):
    M, N, K, X, Y, S = c["g"]; t = f"t{M}x{N}k{K}g{X}{Y}s{S}"
    if family == "4w":
        f = c["f"]
        t += "".join(s for k, s in (("ACC_FP32", "f32"), ("ACC_GROUP_FP32", "ga"), ("ACC_GROUP_FP32_REG", "gr"), ("CSH_IN_ASH", "c"),
                                    ("CSH_FULL", "x"), ("CSH_POOL", "p"), ("CSH_BAND", "b"), ("FRAG_LAYOUT", "fl"), ("IMG_A", "i"),
                                    ("IMG_W", "iw"), ("B_COLMAJOR", "bt"), ("SH_F16V4", "h")) if f[k])
        return ("bx_" if c["bx"] else "") + t
    if family == "8da4w":
        return ("bt_" if c["bt"] else "") + t + ("af" if c["full"] else "") + (f"mb{c['blocks']}" if c["multi"] else "")
    if family == "qk": return ("pk_" if c["pk"] else "") + t + ("nf" if c["nf"] else "")
    return ("ml_" if c["ml"] else "") + t + "/hd" + "+".join(map(str, c["heads"]))

def check(tree):
    """Replay the rules on the yaml variants this device can run (subgroup 32/64, MMA 16x16x16, texture3d or SDPA)."""
    import yaml, pathlib
    g = pathlib.Path(tree) / "backends/vulkan/runtime/graph/ops/glsl/sarc_dev"
    bad = n = 0
    def geo(v): return (v["WG_TILE_M"], v["WG_TILE_N"], v["WG_TILE_K"], v["SG_GRID_X"], v["SG_GRID_Y"], v["SUBGROUP_SIZE"])
    def mma(v): return (v["MMA_M"], v["MMA_N"], v["MMA_K"])
    for name, fam in (("sarc_linear_q4gsw_coopmat_sweep", "4w"), ("sarc_dev_linear_q4gsw_coopmat_bx", "4wbx"),
                      ("sarc_linear_dq8ca_coopmat_zpg_sweep", "8da4w"), ("sarc_dev_linear_dq8ca_coopmat_zpg_bt", "8da4wbt"),
                      ("sarc_sdpa_qk_coopmat_sweep", "qk"), ("sarc_sdpa_qk_coopmat_pk", "qkpk"),
                      ("sarc_sdpa_av_coopmat_sweep", "av"), ("sarc_sdpa_av_coopmat_ml", "avml")):
        y = next(iter(yaml.safe_load(open(g / f"{name}.yaml")).values()))
        for var in y["shader_variants"]:
            v = dict(y["parameter_names_with_default_values"]); v.update(var)
            if fam.startswith(("4w", "8da4w")) and v["IO_STORAGE"] != "texture3d": continue
            r = geom(*geo(v), mma(v))
            if r == "device": continue
            if fam.startswith("4w"):
                f = {k: bool(v.get(k, False)) for k in Q4_FLAGS}
                r = r or ("flags" if not q4_flags_ok(f, fam == "4wbx") else q4(*geo(v), mma(v), f, fam == "4wbx"))
            elif fam.startswith("8da4w"):
                r = r or zpg(*geo(v), mma(v), bool(v["A_MAP_FULL"]), bool(v["A_MULTI_BLOCK"]), v["A_BLOCKS"], fam == "8da4wbt")
            elif fam.startswith("qk"):
                r = r or qk(*geo(v), mma(v), bool(v["NO_MASK_FILL"]), fam == "qkpk")
            else:
                rs = [r or av(*geo(v), mma(v), fam == "avml", h) for h in HEAD_DIM]; r = None if None in rs else rs[0]
            n += 1
            if r: bad += 1; print(f"REJECTED {r}: {v['NAME']}")
    print(f"check: {n} existing 780M-runnable variants replayed, {bad} rejected by the rules")

if __name__ == "__main__":
    if "--check" in sys.argv: check(sys.argv[sys.argv.index("--check") + 1]); sys.exit(0)
    lst = sys.argv[sys.argv.index("--list") + 1] if "--list" in sys.argv else None
    order = ("device", "flags", "geometry", "lds", "shape")
    if not lst: print("family,combinations," + ",".join("pruned_" + o for o in order) + ",survivors,distinct_tile_geometries")
    for fam in ("4w", "8da4w", "qk", "av"):
        if lst and lst != fam: continue
        c = collections.Counter(); geos = set(); n = 0
        for comb, r in space(fam):
            n += 1; c[r] += 1
            if r is None:
                geos.add(comb["g"])
                if lst: print(token(fam, comb))
        if not lst: print(f"{fam},{n}," + ",".join(str(c[o]) for o in order) + f",{c[None]},{len(geos)}")
