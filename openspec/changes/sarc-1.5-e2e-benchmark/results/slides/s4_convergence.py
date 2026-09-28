"""s4_convergence: why 8-bit activations gain less on AMD/Intel. Llama 3.1 8B,
one slope-chart panel per GPU (stock -> tuned), 4-bit weights solid, int8 act.
dashed, linear tok/s per panel from 0. Row 1: 780M/B580/B70 (stock int8 already
ahead, both converge; the 6-GPU variant adds the pre-release RX 7900 XTX here, as
it follows the same pattern). Row 2: 4070 Ti/Orin (stock equal, both rise) + key."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import slide_style as S

LINE = {"4w": dict(color="#2B3036", ls="-"),
        "8da4w": dict(color="#6B7075", ls=(0, (2.2, 1.4)))}
LABEL_STYLE = {"4w": dict(color=S.INK, fontweight="semibold"),
               "8da4w": dict(color=S.MUTED, fontstyle="italic")}
LW = 4.5
MS = 15
GAP_FRAC = 0.20       # min label spacing as a fraction of the panel's y range
XL, XR = -0.13, 1.13  # label anchors (points at x = 0 and 1)
if "7900xtx" in S.GPUS:
    ROWS = [["780m", "b580", "b70", "7900xtx"], ["4070ti", "orin", None]]
    PANEL_W, PANEL_H = 0.19, 0.25
    LEFTS = [0.05, 0.29, 0.53, 0.77]
    BOTTOMS = [0.56, 0.14]
    KEY_RECT = [0.535, 0.12, 0.45, 0.30]  # spans the last two row-2 slots
    XLIM = (-1.0, 2.0)
else:
    ROWS = [["780m", "b580", "b70"], ["4070ti", "orin", None]]
    PANEL_W, PANEL_H = 0.25, 0.27
    LEFTS = [0.055, 0.385, 0.715]
    BOTTOMS = [0.545, 0.105]
    KEY_RECT = [0.725, 0.105, 0.27, 0.33]
    XLIM = (-0.85, 1.85)


def panel(ax, df, g):
    rows = {s: S.cell(df, g, "8b", s) for s in S.SCHEMES}
    ymax = max(max(r.stock_median, r.sarc_median) for r in rows.values()) * 1.12
    gap = GAP_FRAC * ymax
    for s, r in rows.items():
        ax.plot([0, 1], [r.stock_median, r.sarc_median], lw=LW, zorder=2,
                solid_capstyle="round", dash_capstyle="butt", **LINE[s])
        ax.plot([0], [r.stock_median], "o", ms=MS, color=S.STOCK, mec="white",
                mew=2, zorder=3)
        if S.is_pre(g):  # hollow orange = pre-release rows
            ax.plot([1], [r.sarc_median], "o", ms=MS - 1, mfc="white", mec=S.OURS,
                    mew=3.5, zorder=3)
        else:
            ax.plot([1], [r.sarc_median], "o", ms=MS, color=S.OURS, mec="white",
                    mew=2, zorder=3)
    for side, x, ha, col in ((0, XL, "right", "stock_median"),
                             (1, XR, "left", "sarc_median")):
        vals = [getattr(rows[s], col) for s in S.SCHEMES]
        ys = S.repel(vals, gap)
        lift = max(0.0, 0.08 * ymax - min(ys))  # keep labels clear of the baseline
        ys = [y + lift for y in ys]
        for s, v, y in zip(S.SCHEMES, vals, ys):
            ax.text(x, y, S.fmt_toks(v), ha=ha, va="center",
                    fontsize=S.FS_VALUE, **LABEL_STYLE[s])
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.06 * ymax, ymax)
    ax.plot([-0.15, 1.15], [0, 0], color=S.FAINT, lw=1.5, zorder=1)  # zero baseline
    ax.axis("off")
    ax.text(0.5, 1.30, S.GPU_CLASS[g].upper(), transform=ax.transAxes, ha="center",
            va="bottom", fontsize=S.FS_GROUP, color=S.MUTED, fontweight="semibold")
    ax.text(0.5, 1.11, S.GPU_LABEL[g], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=S.FS_TICK, color=S.INK, fontweight="semibold")


def key(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    dot = dict(ls="", marker="o", ms=MS, mec="white", mew=2)
    items = [
        (Line2D([], [], lw=LW, **LINE["4w"]), "4-bit weights", LABEL_STYLE["4w"]),
        (Line2D([], [], lw=LW, **LINE["8da4w"]), "int8 act.", LABEL_STYLE["8da4w"]),
        (Line2D([], [], color=S.STOCK, **dot), S.STOCK_LABEL, {"color": S.INK}),
        (Line2D([], [], color=S.OURS, **dot), "+ tuned kernels", {"color": S.INK}),
    ]
    if S.PRERELEASE:
        items.append((Line2D([], [], ls="", marker="o", ms=MS - 1, mfc="white",
                             mec=S.OURS, mew=3.5), "pre-release †", {"color": S.INK}))
    leg = ax.legend([h for h, _, _ in items], [t for _, t, _ in items],
                    loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
                    fontsize=S.FS_VALUE, handlelength=1.9 if S.PRERELEASE else 2.4, labelspacing=0.55,
                    borderaxespad=0, ncol=2 if S.PRERELEASE else 1,
                    columnspacing=1.0 if S.PRERELEASE else 1.5,
                    handletextpad=0.6)
    for txt, (_, _, st) in zip(leg.get_texts(), items):
        txt.set_color(st["color"])
        txt.set_fontstyle(st.get("fontstyle", "normal"))
        txt.set_fontweight(st.get("fontweight", "normal"))
    ax.text(0.0, 0.0, "Prefill tok/s · Llama 3.1 8B\nlinear scale from 0 per panel",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=S.FS_ANNOT,
            color=S.MUTED, linespacing=1.3)


def main():
    S.apply_style()
    df = S.load_cells()
    fig = S.new_fig()
    for row, bottom in zip(ROWS, BOTTOMS):
        for g, left in zip(row, LEFTS):
            if g is None:
                continue
            panel(fig.add_axes([left, bottom, PANEL_W, PANEL_H]), df, g)
    key(fig, KEY_RECT)
    S.footnote(fig)
    S.save(fig, "s4_convergence")
    plt.close(fig)


if __name__ == "__main__":
    main()
