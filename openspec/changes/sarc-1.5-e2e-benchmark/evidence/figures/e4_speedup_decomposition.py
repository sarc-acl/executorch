"""e4: GEMM kernel speedup = hardware headroom (roof ratio) x efficiency gain (%-of-roof ratio), 8B."""
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (GEMM, GPUS, GPU_LABEL, INK, MUTED, SCHEMES, WIDTH, apply_style, eff_row,
                    efficiency, families, family_ms, save)

apply_style()
E = efficiency()
F = families()

HEAD = "#9AA8B6"
GAIN_UP = "#0072B2"
GAIN_DOWN = "#CC79A7"

rows = []
print("gpu,scheme,stock_roof,sarc_roof,headroom,stock_pct,sarc_pct,eff_gain,product,rate_ratio,trace_gemm_ratio,rel_err_pct")
for gpu in GPUS:
    for scheme in SCHEMES:
        st, sa = eff_row(E, gpu, scheme, "stock"), eff_row(E, gpu, scheme, "sarc")
        h = sa.roof_value / st.roof_value
        g = sa.pct_of_roof / st.pct_of_roof
        prod = h * g
        meas = sa.rate / st.rate
        tr = family_ms(F, gpu, "8b", scheme, "stock")[GEMM] / family_ms(F, gpu, "8b", scheme, "sarc")[GEMM]
        err = 100 * (prod / meas - 1)
        assert abs(err) < 1.0, (gpu, scheme, prod, meas)
        rows.append((gpu, scheme, st, sa, h, g, prod, meas, tr))
        print(f"{gpu},{scheme},{st.roof},{sa.roof},{h:.3f},{st.pct_of_roof},{sa.pct_of_roof},{g:.3f},"
              f"{prod:.3f},{meas:.3f},{tr:.3f},{err:+.2f}")

fig, ax = plt.subplots(figsize=(WIDTH, 5.0))
yt, yl = [], []
y = 0.0
for i, (gpu, scheme, st, sa, h, g, prod, meas, tr) in enumerate(rows):
    if i and scheme == SCHEMES[0]:
        y -= 0.45
    ax.barh(y, h - 1, left=1, height=0.62, color=HEAD, edgecolor="none")
    lo, hi = sorted((h, prod))
    ax.barh(y, hi - lo, left=lo, height=0.30, color=GAIN_UP if g >= 1 else GAIN_DOWN, edgecolor="none",
            zorder=3)
    ax.plot([meas, meas], [y - 0.38, y + 0.38], color=INK, lw=1.8, zorder=4)
    yt.append(y)
    yl.append(f"{GPU_LABEL[gpu]}  {scheme}")
    cols = [f"{h:.2f}", f"{g:.2f}", f"{prod:.2f}", f"{meas:.2f}"]
    for k, t in enumerate(cols):
        ax.text(1.17 + 0.25 * k, y, t, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9, color=INK if k != 1 else (GAIN_UP if g >= 1 else "#9C3D7A"),
                fontweight="bold" if k == 3 else "normal")
    y -= 1.0

for k, t in enumerate(["headroom\nH", "eff. gain\nE", "\nH x E", "measured\nratio"]):
    ax.text(1.17 + 0.25 * k, 0.75, t, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=9, color=MUTED)

ax.set_xscale("log")
ax.set_xlim(0.9, 13)
ax.set_xticks([1, 1.5, 2, 3, 5, 8, 12])
ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8", "12"])
ax.minorticks_off()
ax.set_yticks(yt)
ax.set_yticklabels(yl)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_ylim(y + 0.4, 1.35)
ax.axvline(1, color=MUTED, lw=0.8)
ax.grid(axis="x", color="#E6E6E6", lw=0.6)
ax.set_xlabel("factor, SARC / stock (x, log scale)")

handles = [
    Patch(color=HEAD, label="hardware headroom = matched matrix roof / matched stock roof"),
    Patch(color=GAIN_UP, label="efficiency gain > 1 (SARC % of roof above stock's)"),
    Patch(color=GAIN_DOWN, label="efficiency gain < 1 (bar shrinks back from headroom)"),
    Line2D([], [], color=INK, lw=1.8, label="measured GEMM rate ratio, SARC / stock"),
]
fig.legend(handles=handles, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0), handlelength=1.6)
fig.subplots_adjust(left=0.235, right=0.60, top=0.78, bottom=0.1)
save(fig, "e4_speedup_decomposition")
