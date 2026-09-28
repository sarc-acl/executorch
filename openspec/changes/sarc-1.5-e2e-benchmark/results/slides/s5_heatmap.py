"""s5_heatmap_backup: all 30 speedups (5 GPUs x 3 models x 2 schemes)."""
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

import slide_style as S

# Single-hue sequential ramp (light -> dark orange), CVD-safe by construction
# (monotone lightness); starts at 1x.
CMAP = LinearSegmentedColormap.from_list(
    "ours_seq", ["#FFF4EC", "#FDD0B1", "#F8A06B", "#E8590C", "#A33A04"])
VMIN, VMAX = 1.0, 5.5


def _lum(rgba):
    """WCAG relative luminance of an RGBA tuple."""
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgba[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def main():
    S.apply_style()
    df = S.load_cells()
    fig = S.new_fig()
    ax = fig.add_axes([0.235, 0.14 if S.PRERELEASE else 0.12, 0.735,
                       0.64 if S.PRERELEASE else 0.66])

    cols = [(s, m) for s in S.SCHEMES for m in S.MODELS]
    # Visual gap between the two scheme blocks and between device classes.
    xs = [0, 1, 2, 3.25, 4.25, 5.25]
    ypos, groups = S.grouped_rows(gap=0.3)
    norm = Normalize(VMIN, VMAX)
    for g in S.GPUS:
        for (s, m), x in zip(cols, xs):
            v = S.cell(df, g, m, s).speedup
            y = ypos[g]
            if S.is_pre(g):  # outlined (hollow) cell = pre-release rows
                ax.add_patch(plt.Rectangle((x - 0.44, y - 0.40), 0.88, 0.8,
                                           facecolor="white", edgecolor=CMAP(norm(v)),
                                           lw=5))
                dark = False
            else:
                ax.add_patch(plt.Rectangle((x - 0.47, y - 0.45), 0.94, 0.9,
                                           color=CMAP(norm(v)), lw=0))
                dark = _lum(CMAP(norm(v))) < 0.18  # white text only where it has contrast
            ax.text(x, y, S.fmt_x(v), ha="center", va="center",
                    fontsize=S.FS_BIG, fontweight="semibold",
                    color="white" if dark else S.INK)

    ax.set_xlim(-0.55, 5.8)
    last = ypos[S.GPUS[-1]]
    ax.set_ylim(last + 0.55, -0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([S.MODEL_SHORT[m] for _, m in cols])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_yticks([ypos[g] for g in S.GPUS])
    ax.set_yticklabels([S.GPU_LABEL[g] for g in S.GPUS])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for name, y in groups:
        ax.text(-0.02, y - 0.47, name.upper(), transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=S.FS_GROUP, color=S.MUTED,
                fontweight="semibold")

    # Scheme headers sit a fixed distance above the axes top (clear of the top ticks).
    top = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    line_tf = transforms.offset_copy(top, fig=fig, y=46, units="points")
    text_tf = transforms.offset_copy(top, fig=fig, y=52, units="points")
    for s, x0 in (("4w", 1.0), ("8da4w", 4.25)):
        ax.text(x0, 1.0, S.SCHEME_LONG[s], ha="center", va="bottom",
                fontsize=S.FS_TICK, fontweight="semibold", color=S.INK,
                transform=text_tf, clip_on=False)
        ax.plot([x0 - 1.45, x0 + 1.45], [1.0, 1.0], color=S.FAINT, lw=2,
                transform=line_tf, clip_on=False)

    fig.text(0.97, 0.03, "Speedup over ExecuTorch 1.5 (prefill tok/s)", ha="right",
             va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)
    S.footnote(fig)
    S.save(fig, "s5_heatmap_backup")
    plt.close(fig)
    arr = np.array([[S.cell(df, g, m, s).speedup for (s, m) in cols] for g in S.GPUS])
    print("range", arr.min().round(2), arr.max().round(2))


if __name__ == "__main__":
    main()
