"""s2_ttft_8b_4w: time to first token (2048-token prompt), Llama 3.1 8B, 4-bit
weights. Each row is normalised to its own stock TTFT (grey = full width); the
orange bar is the tuned build's fraction of it. Right column: time saved."""
import matplotlib.pyplot as plt

import slide_style as S

PROMPT = 2048
BAR_H = 0.62
X_SAVED = 1.44  # data x of the "time saved" column (bars span 0..1)


def main():
    S.apply_style()
    df = S.load_cells()
    fig = S.new_fig()
    ax = fig.add_axes([0.235, 0.17, 0.735, 0.69])

    ypos, groups = S.grouped_rows(gap=0.6)
    for g in S.GPUS:
        r = S.cell(df, g, "8b", "4w")
        t_stock = PROMPT / r.stock_median
        t_ours = PROMPT / r.sarc_median
        frac = t_ours / t_stock
        saved = 1 - frac
        y = ypos[g]
        ax.barh(y, 1.0, height=BAR_H, color=S.STOCK, lw=0, zorder=1)
        ax.barh(y, frac, height=BAR_H, color=S.OURS, lw=0, zorder=2)
        ax.plot([frac, frac], [y - BAR_H / 2, y + BAR_H / 2], color="white", lw=3,
                solid_capstyle="butt", zorder=2.5)
        ax.text(1.0 + 0.015, y, S.fmt_s(t_stock), ha="left", va="center",
                fontsize=S.FS_BIG - 2, color=S.MUTED)
        ax.text(frac + 0.02, y, S.fmt_s(t_ours), ha="left", va="center",
                fontsize=S.FS_BIG - 2, color=S.INK, fontweight="bold", zorder=3)
        ax.text(X_SAVED, y, f"−{saved * 100:.0f} %", ha="right", va="center",
                fontsize=S.FS_BIG + 4, color=S.OURS, fontweight="bold")

    top = ypos[S.GPUS[0]] - BAR_H / 2 - 0.08
    ax.text(0.0, top, S.OURS_LABEL, ha="left", va="bottom", fontsize=S.FS_ANNOT,
            color=S.OURS, fontweight="semibold")
    ax.text(1.0, top, S.STOCK_LABEL, ha="right", va="bottom", fontsize=S.FS_ANNOT,
            color=S.MUTED)
    ax.text(X_SAVED, top, "time saved", ha="right", va="bottom",
            fontsize=S.FS_ANNOT, color=S.MUTED)

    ax.set_xlim(0, X_SAVED + 0.01)
    last = ypos[S.GPUS[-1]]
    ax.set_ylim(last + 0.5, -0.95)
    ax.set_yticks([ypos[g] for g in S.GPUS])
    ax.set_yticklabels([S.GPU_LABEL[g] for g in S.GPUS])
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    S.draw_group_labels(ax, groups, dy=-0.58)

    fig.text(0.012, 0.955, "Time to first token · Llama 3.1 8B · 4-bit weights",
             ha="left", va="center", fontsize=S.FS_TICK, color=S.MUTED)
    fig.text(0.235, 0.105, "Each row scaled to its own ExecuTorch 1.5 time",
             ha="left", va="center", fontsize=S.FS_ANNOT, color=S.MUTED)
    S.footnote(fig)
    S.save(fig, "s2_ttft_8b_4w")
    plt.close(fig)


if __name__ == "__main__":
    main()
