"""Kernel base names and tile geometry of enum_space.py tokens (shared by gen_space.py and plan_space.py)."""
import re
PREFIX = {"4w": ("bx_", "sarc_dev_780m_x_linear_q4gsw_coopmat"), "8da4w": ("bt_", "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg"),
          "qk": ("pk_", "sarc_sdpa_qk_coopmat_x780m"), "av": ("ml_", "sarc_sdpa_av_coopmat_x780m")}
def kernel_base(fam, token):
    px, base = PREFIX[fam]
    return f"{base}_{token}"   # "<base>_bx_<tile>" is the twin template "<base>_bx" plus the tile
def geometry(token):
    m = re.search(r"t(\d+)x(\d+)k(\d+)g(\d)(\d)s(\d+)", token)
    return tuple(int(x) for x in m.groups())

Q4_SUFFIX = (("ACC_FP32", "f32"), ("ACC_GROUP_FP32", "ga"), ("ACC_GROUP_FP32_REG", "gr"), ("CSH_IN_ASH", "c"), ("CSH_FULL", "x"),
             ("CSH_POOL", "p"), ("CSH_BAND", "b"), ("FRAG_LAYOUT", "fl"), ("IMG_A", "i"), ("IMG_W", "iw"), ("B_COLMAJOR", "bt"),
             ("SH_F16V4", "h"))
_Q4_RE = re.compile(r"^(bx_)?t(\d+)x(\d+)k(\d+)g(\d)(\d)s(\d+)(f32|ga|gr)?(c)?(x)?(p)?(b(?!t))?(fl)?(i(?!w))?(iw)?(bt)?(h)?$")
def parse_4w(token):
    """Inverse of enum_space.token for the 4w family (MMA 16x16x16, the only shape this device exposes)."""
    m = _Q4_RE.match(token); assert m, token
    g = m.groups(); acc = g[7]
    f = {k: False for k, _ in Q4_SUFFIX}
    if acc: f[{"f32": "ACC_FP32", "ga": "ACC_GROUP_FP32", "gr": "ACC_GROUP_FP32_REG"}[acc]] = True
    for (k, _), v in zip(Q4_SUFFIX[3:], g[8:]): f[k] = v is not None
    return dict(g=tuple(int(x) for x in g[1:7]), mma=(16, 16, 16), f=f, bx=g[0] is not None)
def survives_4w(es, c):
    """The static rules of enum_space.py for one 4w configuration; None if it survives, else the first failed rule."""
    r0 = es.geom(*c["g"], c["mma"])
    if r0 == "device": return r0
    if not es.q4_flags_ok(c["f"], c["bx"]): return "flags"
    return r0 or es.q4(*c["g"], c["mma"], c["f"], c["bx"])
