#!/usr/bin/env python3
"""sweep.py: the sampled parameter search of the owner decision of 2026-10-04 (CAMPAIGN.md) for the Xe2 kernel
families. A configuration is one compile-time parameter set of one shader family; it becomes one extra shader
variant named <family>_xs<id> in a SWEEP BUILD (build-sweep.sh: an export of a commit plus this overlay). The
variants are never written to the working copy: the committed record is the configuration csv (seeded, so it
can be regenerated) and the result csv.

  sweep.py count                                  legal-space sizes after the analytic pruning
  sweep.py sample <space> <seed> <n> <out.csv>    the first n of a seeded shuffle of the analytically legal list
                                                  (n = 0: the whole list); ids are <space digit><rank>
  sweep.py neighbours <space> <measured.csv> <best.csv> <first rank> <out.csv>
                                                  every legal configuration one parameter away from a best one
                                                  and not in <measured.csv>, in a seeded random order
  sweep.py subset <ids.txt | ids.csv> <out.csv> <cfg.csv>...   the rows of the given ids
  sweep.py precheck-dir <tree> <dir> <cfg.csv>... a glsl directory holding only these variants (compile check)
  sweep.py lds <spv dir> <out.csv> <keep> <cfg.csv>...
                                                  per configuration: compiled or not, exact shared-memory bytes
                                                  read from the SPIR-V (Workgroup variables), legal or not, and
                                                  run = 1 for the first <keep> legal ones in file order (0 = all)
  sweep.py overlay <exported tree> <checked.csv>  add the legal variants, their candidate rows and (SDPA) one
                                                  profile per variant to an EXPORTED tree

Analytic pruning (before any compile): workgroup of at most 1024 invocations; the subgroup tile is a whole
number of MMA tiles (8 x 16 output, the only shapes Xe2 exposes: 8x16x16 fp16, 8x16x32 int8); tile sizes divide
every production shape (M = 2048; N in 512..14336; K in 2048..14336, group size 128; S = 2048; head_dim 64 /
128); the flag exclusions the shader bodies enforce with #error; the staging-integrality rules of each body
that are known (gen_xe2.py); a loose lower bound of the shared memory. Exact pruning (after the compile): the
variant compiles with the build's glslc and its Workgroup variables fit LDS_MAX. A tile over the device limit
can hang the GPU instead of failing pipeline creation (impl/sarc/Select.cpp), so nothing over LDS_MAX is run.
"""
import csv, itertools, os, pathlib, random, re, struct, sys

LDS_MAX = 46000      # bytes; maxComputeSharedMemorySize is 49152 on the B70 (the B580's value was not read)
WG_MAX = 1024
SPACES = {"4w": 1, "8da4w": 2, "qk": 3, "av": 4}
COLS = ["id", "space", "fam", "M", "N", "K", "sx", "sy", "S", "layout", "img_a", "img_w", "drain", "acc", "nf",
        "a_raw", "b_pair", "cia", "du", "bsel"]
PARAMS = {  # the parameters that vary in each space (the importance table is over these)
    "4w": ["fam", "M", "N", "K", "sx", "sy", "S", "layout", "img_a", "img_w", "drain", "acc"],
    "8da4w": ["fam", "M", "N", "K", "sx", "sy", "S", "a_raw", "b_pair", "cia", "du", "bsel"],
    "qk": ["fam", "M", "N", "K", "sx", "sy", "S", "nf"],
    "av": ["fam", "M", "N", "K", "sx", "sy", "S"],
}
FAMILY = {  # (space, fam) -> (yaml / glsl file stem = kernel prefix, Op)
    ("4w", "sweep"): ("sarc_linear_q4gsw_coopmat_sweep", "kQ4gswLinear"),
    ("4w", "xe2s"): ("sarc_dev_linear_q4gsw_coopmat_xe2s", "kQ4gswLinear"),
    ("4w", "xe2bx"): ("sarc_dev_linear_q4gsw_coopmat_xe2bx", "kQ4gswLinear"),
    ("8da4w", "zpg"): ("sarc_linear_dq8ca_coopmat_zpg_sweep", "kDq8caLinear"),
    ("8da4w", "bt"): ("sarc_dev_linear_dq8ca_coopmat_zpg_bt", "kDq8caLinear"),
    ("8da4w", "xe2bt"): ("sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt", "kDq8caLinear"),
    ("8da4w", "zpgtr"): ("sarc_linear_dq8ca_coopmat_zpgtr_sweep", "kDq8caLinear"),
    ("qk", "sweep"): ("sarc_sdpa_qk_coopmat_sweep", "kSdpaQk"),
    ("qk", "pk"): ("sarc_sdpa_qk_coopmat_pk", "kSdpaQk"),
    ("qk", "xe2"): ("sarc_sdpa_qk_coopmat_xe2", "kSdpaQk"),
    ("qk", "xe2c"): ("sarc_sdpa_qk_coopmat_xe2c", "kSdpaQk"),
    ("av", "sweep"): ("sarc_sdpa_av_coopmat_sweep", "kSdpaAv"),
    ("av", "ml"): ("sarc_sdpa_av_coopmat_ml", "kSdpaAv"),
    ("av", "xe2"): ("sarc_sdpa_av_coopmat_xe2", "kSdpaAv"),
}

def geo(Ms, Ns, Ks):
    for m, n, k, sx, sy, S in itertools.product(Ms, Ns, Ks, (1, 2, 4, 8, 16), (1, 2, 4, 8, 16), (16, 32)):
        if sx * sy * S <= WG_MAX and m % sy == 0 and n % sx == 0 and (m // sy) % 8 == 0 and (n // sx) % 16 == 0:
            yield m, n, k, sx, sy, S

def cfg(space, fam, g, **kw):
    c = dict.fromkeys(COLS, "-"); c.update(space=space, fam=fam, M=g[0], N=g[1], K=g[2], sx=g[3], sy=g[4], S=g[5]); c.update(kw)
    return c

def legal_4w():
    for fam in ("sweep", "xe2s", "xe2bx"):
        for g in geo((32, 64, 128, 256), (32, 64, 128, 256), (16, 32, 64, 128)):
            m, n, k, sx, sy, S = g; wg = sx * sy * S
            if wg % (k // 8) or wg % (n // 8): continue                      # a thread stages whole uvec4 columns
            if fam != "xe2s" and m % (wg // (k // 8)): continue              # A passes integral (xe2s rounds up)
            if fam == "sweep" and k % (wg // (n // 8)): continue             # B passes integral
            if fam == "xe2s" and (wg // (n // 8)) % 4: continue              # B rows per pass keep the nibble phase
            if 4 * (m * k + k * n) > LDS_MAX: continue                       # A + B double-buffered, lower bound
            for layout, ia, iw, drain, acc in itertools.product(("frag", "pad", "col", "f16"), (0, 1), (0, 1),
                                                                ("def", "band", "full", "cia", "pool"), ("fp16", "fp32", "grp")):
                if layout == "frag" and drain in ("cia", "pool"): continue   # #error in the body
                if layout == "col" and (drain == "pool" or fam == "xe2s"): continue
                if layout == "f16" and drain == "cia": continue
                if fam == "xe2bx" and drain == "pool": continue
                yield cfg("4w", fam, g, layout=layout, img_a=ia, img_w=iw, drain=drain, acc=acc)

def dq_a(m, k, wg):
    """A staging of the zpg bodies: (A_MAP_FULL, A_BLOCKS) or None."""
    blocks = (m // 4) * (k // 4)
    if blocks % wg == 0: return 1, blocks // wg
    if blocks < wg: return 0, 1
    return None

def legal_8da4w():
    for g in geo((32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128)):
        m, n, k, sx, sy, S = g; wg = sx * sy * S
        if 2 * (k // 32) * (m * 32 + n * 32) > LDS_MAX: continue             # two slices of A and B, lower bound
        if dq_a(m, k, wg):
            if ((k // 4) * n) % wg == 0: yield cfg("8da4w", "zpg", g)
            if (2 * (k // 4) * (n // 8)) % wg == 0: yield cfg("8da4w", "bt", g)
            yield cfg("8da4w", "xe2bt", g)
        for ar, bp, cia, du, bs in itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1, 2)):
            yield cfg("8da4w", "zpgtr", g, a_raw=ar, b_pair=bp, cia=cia, du=du, bsel=bs)

def legal_qk():
    for fam in ("sweep", "pk", "xe2", "xe2c"):
        for g in geo((32, 64, 128, 256), (32, 64, 128, 256), (32,) if fam == "sweep" else (32, 64)):
            if 2 * g[0] * g[1] > LDS_MAX: continue                           # the scaled-result scratch alone
            for nf in (0, 1): yield cfg("qk", fam, g, nf=nf)

def legal_av():
    for fam in ("sweep", "ml", "xe2"):
        for g in geo((32, 64, 128, 256), (32, 64, 128), (32, 64)):
            m, n, k, sx, sy, S = g
            if fam == "sweep" and not (sx * sy * S == 4 * m == k * n // 8): continue   # single-pass staging
            if 2 * (m * k + k * n) > LDS_MAX: continue
            yield cfg("av", fam, g)

LEGAL = {"4w": legal_4w, "8da4w": legal_8da4w, "qk": legal_qk, "av": legal_av}
def key(c, space): return tuple(str(c[p]) for p in PARAMS[space])
def read(paths): return [r for p in paths for r in csv.DictReader(open(p))]
def write(path, rows, cols=COLS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def kernel(c):
    return f'{FAMILY[(c["space"], c["fam"])][0]}_xs{c["id"]}' + ("nf" if str(c["nf"]) == "1" else "")
def token(c): return f'xs{c["id"]}' + ("nf" if str(c["nf"]) == "1" else "")

def yaml_variant(c):
    s = c["space"]; m, n, k, sx, sy, S = (int(c[x]) for x in ("M", "N", "K", "sx", "sy", "S")); t = lambda b: "true" if b else "false"
    tile = f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {S}\n      MMA_M: 8\n"
    if s in ("qk", "av"):
        return f"    - NAME: {kernel(c)}\n{tile}" + (f"      NO_MASK_FILL: {t(str(c['nf']) == '1')}\n" if s == "qk" else "")
    y = f"    - NAME: {kernel(c)}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n{tile}"
    if s == "4w":
        fl = {"FRAG_LAYOUT": c["layout"] == "frag", "B_COLMAJOR": c["layout"] == "col", "SH_F16V4": c["layout"] == "f16",
              "IMG_A": str(c["img_a"]) == "1", "IMG_W": str(c["img_w"]) == "1", "CSH_BAND": c["drain"] == "band",
              "CSH_FULL": c["drain"] in ("full", "pool"), "CSH_POOL": c["drain"] == "pool", "CSH_IN_ASH": c["drain"] == "cia",
              "ACC_FP32": c["acc"] == "fp32", "ACC_GROUP_FP32": c["acc"] == "grp"}
        return y + "".join(f"      {f}: {t(v)}\n" for f, v in fl.items())
    y += "      MMA_K: 32\n"
    if c["fam"] == "zpgtr":
        return y + (f"      A_RAW: {t(str(c['a_raw']) == '1')}\n      B_PAIR: {t(str(c['b_pair']) == '1')}\n      CSH_IN_ASH: {t(str(c['cia']) == '1')}\n"
                    f"      DRAIN_UNROLL: {t(str(c['du']) == '1')}\n      B_SEL_EARLY_N: {c['bsel']}\n")
    full, ab = dq_a(m, k, sx * sy * S)
    return y + f"      A_MAP_FULL: {t(full)}\n      A_MULTI_BLOCK: {t(full)}\n      A_BLOCKS: {ab}\n"

def row_cpp(c):
    s = c["space"]; _, op = FAMILY[(s, c["fam"])]
    cia = (s == "4w" and c["drain"] == "cia") or (c["fam"] == "zpgtr" and str(c["cia"]) == "1")
    full = s == "4w" and c["drain"] in ("full", "pool"); pool = s == "4w" and c["drain"] == "pool"; b = lambda v: "true" if v else "false"
    st = "kBufBuf" if s in ("qk", "av") else "kTex3dTex2d"
    return (f'    {{"", nullptr, Op::{op}, "{kernel(c)}",\n     {{{c["M"]}, {c["N"]}, {c["K"]}, {c["sx"]}, {c["sy"]}, {c["S"]}, 8, {b(cia)}, {b(full)}, {b(pool)}}},\n'
            f'     {st}, nullptr, Status::kUnverified, {b(c["fam"] == "zpgtr")}}},\n')

def variants_block(tree, cfgs):
    """{yaml stem: (header up to and including shader_variants:, variant text)} for the families of cfgs."""
    g = pathlib.Path(tree) / "backends/vulkan/runtime/graph/ops/glsl/sarc_dev"; out = {}
    for c in cfgs:
        stem = FAMILY[(c["space"], c["fam"])][0]
        if stem not in out:
            y = (g / f"{stem}.yaml").read_text(); out[stem] = [y[:y.index("  shader_variants:\n") + len("  shader_variants:\n")], ""]
        out[stem][1] += yaml_variant(c)
    return g, out

def spv_shared_bytes(path):
    """Sum of the sizes of the Workgroup-storage variables of a SPIR-V module, or None if it is not one."""
    d = open(path, "rb").read()
    if len(d) < 20 or struct.unpack_from("<I", d)[0] != 0x07230203: return None
    w = struct.unpack(f"<{len(d) // 4}I", d[:len(d) // 4 * 4]); i = 5; size = {}; const = {}; ptr = {}; arr = {}; total = 0
    while i < len(w):
        op, wc = w[i] & 0xFFFF, w[i] >> 16; a = w[i + 1:i + wc]; i += wc
        if op in (21, 22): size[a[0]] = a[1] // 8                             # OpTypeInt, OpTypeFloat
        elif op == 20: size[a[0]] = 4                                         # OpTypeBool
        elif op == 23: size[a[0]] = size[a[1]] * a[2]                         # OpTypeVector
        elif op == 28: arr[a[0]] = (a[1], a[2])                               # OpTypeArray (length id)
        elif op == 32: ptr[a[0]] = (a[1], a[2])                               # OpTypePointer (storage class, type)
        elif op in (43, 50): const[a[1]] = a[2]                               # OpConstant, OpSpecConstant
        elif op == 59 and a[2] == 4:                                          # OpVariable, Workgroup
            t = ptr[a[0]][1]; n = 1
            while t in arr: n *= const[arr[t][1]]; t = arr[t][0]
            total += n * size[t]
    return total

def main():
    cmd = sys.argv[1]
    if cmd == "count":
        for s, f in LEGAL.items():
            L = list(f()); fams = {}
            for c in L: fams[c["fam"]] = fams.get(c["fam"], 0) + 1
            print(f"{s}: {len(L)} analytically legal configurations, {len({(c['M'], c['N'], c['K'], c['sx'], c['sy'], c['S']) for c in L})} tile geometries; by family {fams}")
    elif cmd == "sample":
        space, seed, n, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        L = list(LEGAL[space]()); random.Random(seed).shuffle(L); L = L[:n] if n else L
        for i, c in enumerate(L): c["id"] = f"{SPACES[space]}{i:05d}"
        write(out, L); print(f"{space}: {len(L)} configurations, seed {seed} -> {out}")
    elif cmd == "neighbours":
        space, measured, best, first, out = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
        have = {key(c, space) for c in read([measured])}; bk = [key(c, space) for c in read([best])]; L = []; seen = set()
        for c in LEGAL[space]():
            k = key(c, space)
            if k in have or k in seen: continue
            if any(sum(a != b for a, b in zip(k, b_)) == 1 for b_ in bk): seen.add(k); L.append(c)
        random.Random(20261005).shuffle(L)                                    # build-sweep.sh may keep only the first ones
        for i, c in enumerate(L): c["id"] = f"{SPACES[space]}{first + i:05d}"
        write(out, L); print(f"{space}: {len(L)} unmeasured one-parameter neighbours of {len(bk)} configurations -> {out}")
    elif cmd == "subset":
        ids = [l.split(",")[0].strip() for l in open(sys.argv[2]) if l.strip()]; rows = {c["id"]: c for c in read(sys.argv[4:])}
        keep = [rows[i] for i in dict.fromkeys(ids) if i in rows]
        write(sys.argv[3], keep, list(keep[0].keys()) if keep else COLS); print(f"{len(keep)} of {len(set(ids))} ids -> {sys.argv[3]}")
    elif cmd == "precheck-dir":
        tree, d = sys.argv[2], pathlib.Path(sys.argv[3]); cfgs = read(sys.argv[4:]); d.mkdir(parents=True)
        g, blocks = variants_block(tree, cfgs)
        for src in (g, g.parent / "sarc", g.parent):
            for f in src.glob("*.glslh"): (d / f.name).write_bytes(f.read_bytes())
            for f in src.glob("*.h"): (d / f.name).write_bytes(f.read_bytes())
        for stem, (head, body) in blocks.items():
            (d / f"{stem}.glsl").write_bytes((g / f"{stem}.glsl").read_bytes()); (d / f"{stem}.yaml").write_text(head + body)
        print(f"{len(cfgs)} variants of {len(blocks)} families in {d}")
    elif cmd == "lds":
        spv, out, keep = pathlib.Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]); cfgs = read(sys.argv[5:]); n = 0
        files = {re.search(r"_xs(\d{6})", f.name).group(1): f for f in spv.rglob("*.spv") if re.search(r"_xs\d{6}", f.name)}
        for c in cfgs:
            b = spv_shared_bytes(files[c["id"]]) if c["id"] in files else None
            c["compiled"] = int(b is not None); c["shared_bytes"] = "" if b is None else b; c["legal"] = int(b is not None and b <= LDS_MAX)
            c["run"] = int(c["legal"] and (keep == 0 or n < keep)); n += c["legal"]
        write(out, cfgs, COLS + ["compiled", "shared_bytes", "legal", "run"])
        print(f"{len(cfgs)} checked: {sum(c['compiled'] for c in cfgs)} compile, {n} legal (shared memory <= {LDS_MAX} bytes), {sum(c['run'] for c in cfgs)} to run -> {out}")
    elif cmd == "overlay":
        tree = sys.argv[2]; cfgs = [c for c in read(sys.argv[3:]) if c.get("run", "1") == "1"]
        g, blocks = variants_block(tree, cfgs)
        for stem, (_, body) in blocks.items():
            p = g / f"{stem}.yaml"; p.write_text(p.read_text().rstrip("\n") + "\n# xe2 sweep overlay (tools/sweep.py, not committed)\n" + body)
        impl = pathlib.Path(tree) / "backends/vulkan/runtime/graph/ops/impl/sarc_dev"
        (impl / "Xe2SweepRows.cpp").write_text(
            "// xe2 sweep overlay (openspec/changes/sarc-1.5-xe2-prefill-refine/tools/sweep.py): candidate rows of a sweep build, not committed.\n"
            "#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>\nnamespace vkcompute {\nnamespace sarc {\nnamespace {\n"
            "const Row kXe2SweepCandidates[] = {\n" + "".join(row_cpp(c) for c in cfgs) + "};\nstruct Registrar {\n  Registrar() {\n"
            "    register_candidates(kXe2SweepCandidates, sizeof(kXe2SweepCandidates) / sizeof(kXe2SweepCandidates[0]));\n  }\n} registrar;\n"
            "} // namespace\n} // namespace sarc\n} // namespace vkcompute\n")
        sd = [c for c in cfgs if c["space"] in ("qk", "av")]
        if sd:                                                                # one profile per SDPA variant: xe2-xs<id>
            p = impl / "Overrides.cpp"; t = p.read_text(); a = t.index("// xe2 end\n"); b = t.rindex("    // xe2 end\n")
            prefs = "".join(f'const Preference kXe2Xs{c["id"]}[] = {{{{Op::{FAMILY[(c["space"], c["fam"])][1]}, "{token(c)}", nullptr}}}};\n' for c in sd)
            profs = "".join(f'    {{"xe2-xs{c["id"]}", kXe2Xs{c["id"]}, 1}},\n' for c in sd)
            p.write_text(t[:a] + prefs + t[a:b] + profs + t[b:])
        print(f"overlay: {len(cfgs)} variants, {len(sd)} SDPA profiles")
    else:
        print(__doc__); sys.exit(2)

if __name__ == "__main__":
    main()
