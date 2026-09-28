"""s1_hero_8b_4w / s1b_hero_8b_8da4w: dumbbell of prefill tok/s, Llama 3.1 8B,
stock -> tuned, one row per GPU grouped by device class, shared log x range."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import slide_style as S

XLIM = (18, 22000)  # shared by both schemes so the two slides can be flipped
XTICKS = [30, 100, 300, 1000, 3000, 10000]
DOT = 20           # marker diameter (pt)
ARROW = "#F3A67C"  # light orange shaft: direction toward "ours"


def draw(df, scheme, stem):
    S.apply_style()
    fig = S.new_fig()
    ax = fig.add_axes([0.235, 0.22 if S.PRERELEASE else 0.20, 0.735,
                       0.695 if S.PRERELEASE else 0.70])

    ypos, groups = S.grouped_rows(gap=0.75)
    for g in S.GPUS:
        r = S.cell(df, g, "8b", scheme)
        y, a, b = ypos[g], r.stock_median, r.sarc_median
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=5,
                                    mutation_scale=30, shrinkA=DOT / 2 + 2,
                                    shrinkB=DOT / 2 + 1), zorder=2)
        ax.plot([a], [y], "o", ms=DOT, color=S.STOCK, mec="white", mew=2.5, zorder=3)
        if S.is_pre(g):  # hollow orange = pre-release rows
            ax.plot([b], [y], "o", ms=DOT - 2, mfc="white", mec=S.OURS, mew=4, zorder=3)
        else:
            ax.plot([b], [y], "o", ms=DOT, color=S.OURS, mec="white", mew=2.5, zorder=3)
        for x, col in ((a, S.MUTED), (b, S.INK)):
            ax.annotate(S.fmt_toks(x), (x, y), xytext=(0, 13), textcoords="offset points",
                        ha="center", va="bottom", fontsize=S.FS_VALUE, color=col)
        ax.annotate(S.fmt_x(r.speedup), (b, y), xytext=(DOT / 2 + 12, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=S.FS_BIG + 2, fontweight="bold", color=S.OURS)

    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:,}" for t in XTICKS])
    ax.set_xlabel("Prefill throughput (tokens/s, log scale)", labelpad=10)
    ax.grid(True, axis="x", color=S.GRID, lw=1.2)

    ax.set_yticks([ypos[g] for g in S.GPUS])
    ax.set_yticklabels([S.GPU_LABEL[g] for g in S.GPUS])
    ax.tick_params(axis="y", length=0, pad=14)
    ax.spines["left"].set_visible(False)
    last = ypos[S.GPUS[-1]]
    ax.set_ylim(last + 0.45, -0.72)
    S.draw_group_labels(ax, groups, dy=-0.60)

    # Direct build labels under the top row's dots instead of a legend.
    top = S.cell(df, S.GPUS[0], "8b", scheme)
    y0 = ypos[S.GPUS[0]]
    ax.annotate(S.STOCK_LABEL, (top.stock_median, y0), xytext=(DOT / 2, -18),
                textcoords="offset points", ha="right", va="top",
                fontsize=S.FS_ANNOT, color=S.MUTED)
    ax.annotate(S.OURS_LABEL, (top.sarc_median, y0), xytext=(-DOT / 2, -18),
                textcoords="offset points", ha="left", va="top",
                fontsize=S.FS_ANNOT, color=S.OURS, fontweight="semibold")
    fig.text(0.012, 0.955, f"Llama 3.1 8B · {S.SCHEME_LONG[scheme]}", ha="left",
             va="center", fontsize=S.FS_TICK, color=S.MUTED)

    S.footnote(fig)
    S.save(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    df = S.load_cells()
    lo = df[df.model == "8b"][["stock_median", "sarc_median"]].min().min()
    hi = df[df.model == "8b"][["stock_median", "sarc_median"]].max().max()
    assert XLIM[0] < lo and hi < XLIM[1] / 3, (lo, hi)
    draw(df, "4w", "s1_hero_8b_4w")
    draw(df, "8da4w", "s1b_hero_8b_8da4w")
