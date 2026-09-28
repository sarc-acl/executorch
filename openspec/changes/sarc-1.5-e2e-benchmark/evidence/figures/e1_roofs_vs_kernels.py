"""e1: measured roofs (bars) vs achieved 8B prefill GEMM rates (markers), one panel per GPU."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (GPUS, GPU_LABEL, INK, MUTED, ROOF_LABEL, SARC, SARC_EDGE, STOCK, STOCK_EDGE,
                    STOCK_ROOF, WIDTH, apply_style, eff_row, efficiency, roofs, sarc_roof, save)

apply_style()
R = roofs()
E = efficiency()

XMIN, XMAX = 0.06, 1500
BAR_FILL = "#DCE3EA"
BAR_EDGE = "#7A8896"
MATRIX_FILL = "#C3CFDB"

fig, axes = plt.subplots(len(GPUS), 1, figsize=(WIDTH, 7.6), sharex=True)
print("gpu,row,roof,roof_value,marker,rate,pct_of_roof")
for ax, gpu in zip(axes, GPUS):
    rows = [  # (roof key, marker scheme, build)
        (STOCK_ROOF["4w"], "4w", "stock"),
        (STOCK_ROOF["8da4w"], "8da4w", "stock"),
        (sarc_roof(gpu, "4w"), "4w", "sarc"),
        (sarc_roof(gpu, "8da4w"), "8da4w", "sarc"),
    ]
    ylabels = []
    for i, (roof, scheme, build) in enumerate(rows):
        y = -i
        v = R[gpu][roof]
        ax.barh(y, v - XMIN, left=XMIN, height=0.72,
                color=MATRIX_FILL if roof.startswith("matrix") else BAR_FILL,
                edgecolor=BAR_EDGE, linewidth=0.6)
        ax.text(v * 1.08, y, f"{v:.3g}", va="center", ha="left", fontsize=9, color=INK)
        ylabels.append(ROOF_LABEL[roof])

        r = eff_row(E, gpu, scheme, build)
        assert r.roof == roof, (gpu, scheme, build, r.roof, roof)
        assert abs(r.roof_value - v) < 1e-6, (gpu, roof, r.roof_value, v)
        mk = "o" if scheme == "4w" else "^"
        fc, ec = (STOCK, STOCK_EDGE) if build == "stock" else (SARC, SARC_EDGE)
        ax.plot(r.rate, y, marker=mk, ms=8.5, mfc=fc, mec=ec, mew=0.9, ls="none", zorder=4)
        ax.text(r.rate / 1.17, y, f"{r.rate:.3g} ({r.pct_of_roof:.0f}%)", va="center", ha="right",
                fontsize=9, color=INK, zorder=5)
        print(f"{gpu},{i},{roof},{v},{build} {scheme},{r.rate},{r.pct_of_roof}")

    ax.set_yticks([0, -1, -2, -3])
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-3.55, 0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.grid(axis="x", which="major", color="#E6E6E6", lw=0.6)
    ax.set_title(GPU_LABEL[gpu], loc="left", fontweight="bold", pad=3)
    # Divider between scalar and matrix roofs.
    ax.axhline(-1.5, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)))

    if gpu == "780m":
        ratio = R[gpu]["matrix_int8"] / R[gpu]["matrix_fp16_fp32"]
        ax.text(32, -2.45,
                f"int8 matrix roof = {ratio:.2f}x\nfp16 matrix roof: no int8\nthroughput advantage",
                ha="left", va="center", fontsize=9, color=MUTED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

axes[-1].set_xlabel("tera-ops/s, fp16 FLOP or int8 OP (log scale)")
axes[-1].set_xticks([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
axes[-1].set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100", "300", "1000"])

handles = [
    Patch(facecolor=BAR_FILL, edgecolor=BAR_EDGE, label="measured scalar roof"),
    Patch(facecolor=MATRIX_FILL, edgecolor=BAR_EDGE, label="measured matrix roof"),
    Line2D([], [], marker="o", ls="none", mfc=STOCK, mec=STOCK_EDGE, ms=8, label="stock 4w"),
    Line2D([], [], marker="^", ls="none", mfc=STOCK, mec=STOCK_EDGE, ms=8, label="stock 8da4w"),
    Line2D([], [], marker="o", ls="none", mfc=SARC, mec=SARC_EDGE, ms=8, label="SARC 4w"),
    Line2D([], [], marker="^", ls="none", mfc=SARC, mec=SARC_EDGE, ms=8, label="SARC 8da4w"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985),
           columnspacing=1.4, handletextpad=0.4)
fig.suptitle("Achieved 8B prefill GEMM rate on its matched roof (label: rate and % of that roof)",
             y=1.0, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.6)
save(fig, "e1_roofs_vs_kernels")
