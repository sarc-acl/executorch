#!/usr/bin/env python3
"""neighbours.py <order 1|2> <token>...: the surviving 4w configurations that differ from a token in exactly one
parameter (order 1) or in one or two parameters (order 2). A parameter is one of: tile M, N, K, grid X, grid Y,
subgroup size, accumulator mode (fp16 / fp32 / group / group in registers), drain mode (default / in Ash / full /
full pooled / band), FRAG_LAYOUT, IMG_A, IMG_W, B_COLMAJOR, SH_F16V4, texel-wise staging. Prints
"<neighbour> <centre> <changed parameters>", each neighbour once (first centre that reaches it)."""
import itertools, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import enum_space as es, gen_space_names as gn

ACC = {"fp16": (), "fp32": ("ACC_FP32",), "group": ("ACC_GROUP_FP32",), "group_reg": ("ACC_GROUP_FP32_REG",)}
CSH = {"default": (), "in_ash": ("CSH_IN_ASH",), "full": ("CSH_FULL",), "full_pool": ("CSH_FULL", "CSH_POOL"), "band": ("CSH_BAND",)}
BOOL = ("FRAG_LAYOUT", "IMG_A", "IMG_W", "B_COLMAJOR", "SH_F16V4")
def moves(c):
    """Yields (parameter name, changed copy) for every single-parameter change."""
    for i, (name, vals) in enumerate((("M", es.TILE), ("N", es.TILE), ("K", es.KS), ("X", es.GRID), ("Y", es.GRID), ("S", es.DEV_SG))):
        for v in vals:
            if v != c["g"][i]:
                g = list(c["g"]); g[i] = v; yield name, dict(c, g=tuple(g))
    for name, table in (("ACC", ACC), ("CSH", CSH)):
        keys = {k for ks in table.values() for k in ks}
        for mode, on in table.items():
            f = dict(c["f"]); f.update({k: k in on for k in keys})
            if f != c["f"]: yield name, dict(c, f=f)
    for k in BOOL:
        f = dict(c["f"]); f[k] = not f[k]; yield k, dict(c, f=f)
    yield "TEXEL_STAGING", dict(c, bx=not c["bx"])
def neighbours(token, order):
    c0 = gn.parse_4w(token); out = {}
    for n1, c1 in moves(c0):
        if gn.survives_4w(es, c1) is None: out.setdefault(es.token("4w", c1), n1)
        if order == 2:
            for n2, c2 in moves(c1):
                if n2 != n1 and gn.survives_4w(es, c2) is None: out.setdefault(es.token("4w", c2), f"{n1}+{n2}")
    out.pop(token, None)
    return out
if __name__ == "__main__":
    order = int(sys.argv[1]); seen = set(sys.argv[2:])
    for t in sys.argv[2:]:
        for n, how in neighbours(t, order).items():
            if n not in seen: seen.add(n); print(n, t, how)
