"""s3_scaling: speedup vs model size, one line per GPU, two schemes side by side.
The only slide that colours by GPU (Okabe-Ito subset + marker shapes + direct names)."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import slide_style as S

NAME = {  # direct end labels; shortened only where the full name does not fit
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "4070 Ti SUPER",
    "orin": "Orin Nano",
}
YLIM = (0.8, 5.9)
LABEL_GAP = 0.40  # min vertical spacing of end labels (data units)
X_LABEL = 2.28    # x of end labels (points sit at 0, 1, 2)


def panel(ax, df, scheme, show_ylabel):
    xs = [0, 1, 2]
    ax.axhline(1.0, color=S.MUTED, lw=2, ls=(0, (5, 4)), zorder=1)
    ends = {}
    for g in S.GPUS:
        rows = [S.cell(df, g, m, scheme) for m in S.MODELS]
        ys = [r.speedup for r in rows]
        lo = [r.speedup - r.speedup_ci_lo for r in rows]
        hi = [r.speedup_ci_hi - r.speedup for r in rows]
        c = S.GPU_COLOR[g]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=c, elinewidth=2,
                    capsize=6, capthick=2, alpha=0.55, zorder=2)
        ax.plot(xs, ys, color=c, lw=4, marker=S.GPU_MARKER[g], ms=14,
                mec="white", mew=2, zorder=3)
        ends[g] = ys[-1]

    gs = list(ends)
    ly = S.repel([ends[g] for g in gs], LABEL_GAP)
    for g, y_lab in zip(gs, ly):
        ax.plot([2.13, X_LABEL - 0.04], [ends[g], y_lab], color=S.GPU_COLOR[g],
                lw=2, zorder=2)
        ax.text(X_LABEL, y_lab, NAME[g], ha="left", va="center",
                fontsize=S.FS_VALUE, color=S.INK)

    ax.set_xlim(-0.25, 3.75)
    ax.set_ylim(*YLIM)
    ax.set_xticks(xs)
    ax.set_xticklabels([S.MODEL_SHORT[m] for m in S.MODELS])
    ax.spines["bottom"].set_bounds(-0.25, 2.25)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    ax.set_yticklabels([f"{t}×" for t in [1, 2, 3, 4, 5]])
    ax.grid(True, axis="y", color=S.GRID, lw=1.2)
    ax.set_xlabel("Llama model size", labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Speedup vs ExecuTorch 1.5", labelpad=10)
    ax.set_title(S.SCHEME_LONG[scheme], loc="left", fontsize=S.FS_TICK,
                 color=S.INK, pad=18, fontweight="semibold")


def main():
    S.apply_style()
    df = S.load_cells()
    fig = S.new_fig()
    ax1 = fig.add_axes([0.095, 0.2, 0.40, 0.66])
    ax2 = fig.add_axes([0.575, 0.2, 0.40, 0.66], sharey=ax1)
    panel(ax1, df, "4w", True)
    panel(ax2, df, "8da4w", False)
    ax2.tick_params(labelleft=True)
    S.footnote(fig, " · error bars: ~95 % paired bootstrap CI")
    S.save(fig, "s3_scaling")
    plt.close(fig)


if __name__ == "__main__":
    main()
