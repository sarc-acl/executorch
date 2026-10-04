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
