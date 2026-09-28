"""r4: absolute end-to-end prefill throughput (tok/s, 2048-token real-text prompt) before -> after the
re-tuning, per GPU x scheme, one panel per model, log axis. Source: ../../../refine/cells.csv
(median of 5 interleaved repeats per build)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import common as C


def fmt(v):
    return f"{v/1000:.1f}k" if v >= 10000 else f"{v:.0f}"


def main():
    C.setup_style(9)
    cells = C.load_cells()
    rows = [(g, s) for g in C.GPUS for s in C.SCHEMES]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.3), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.80, bottom=0.195, wspace=0.2)
    n = len(rows)
    for ax, mdl in zip(axes, C.MODELS):
        sub = cells[cells.M == mdl]
        for ri, (gpu, sch) in enumerate(rows):
            y = n - 1 - ri
            r = sub[(sub.G == gpu) & (sub.scheme == sch)].iloc[0]
            b, a = r.stock_median, r.sarc_median
            col = C.GPU_COLOR[gpu]
            ax.plot([b, a], [y, y], color=col, lw=2.0, zorder=2, solid_capstyle="butt")
            ax.plot(b, y, "o", ms=5, mfc=C.BEFORE_GREY, mec=C.INK2, mew=0.7, zorder=3)
            ax.plot(a, y, "o", ms=5.5, mfc=col, mec="white", mew=0.7, zorder=4)
            ax.text(max(a, b) * 1.3, y, f"{fmt(b)}→{fmt(a)}", ha="left", va="center", fontsize=9, color=C.INK2)
        ax.set_xscale("log")
        ax.set_xlim(60, 4e5)
        ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["100", "1k", "10k"])
        ax.grid(axis="x", color=C.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title({"llama-3.2-1b": "Llama 3.2 1B", "llama-3.2-3b": "Llama 3.2 3B",
                      "llama-3.1-8b": "Llama 3.1 8B"}[mdl], fontweight="bold", fontsize=9)
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", length=0)
        for k in range(1, len(C.GPUS)):
            ax.axhline(n - 2 * k - 0.5, color=C.GRID, lw=0.6, zorder=0)
    axes[0].set_yticks([n - 1 - i for i in range(n)])
    axes[0].set_yticklabels([f"{C.GPU_SHORT[g]} {s}" for g, s in rows])
    axes[1].set_xlabel("Prefill throughput (tokens/s, log scale)")
    fig.text(0.015, 0.99, "Prefill throughput before → after the re-tuning (measured, 2048-token prompt)",
             fontsize=10, fontweight="bold", va="top")
    handles = [
        Line2D([], [], ls="", marker="o", mfc=C.BEFORE_GREY, mec=C.INK2, ms=6, label="previous best WMMA"),
        Line2D([], [], ls="", marker="o", mfc="#6b6b6b", mec="white", ms=6, label="re-tuned (colour = GPU)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    fig.text(0.015, 0.008,
             "Median of 5 interleaved repeats per build, real-text prompt. 4070 Ti 4w is flat or lower (accuracy fix);\n"
             "780M 8da4w is unchanged. Previous kernels on B580, B70, 4070 Ti and Orin were opt-in.",
             fontsize=9, color=C.INK2, va="bottom")
    C.save(fig, "r4_tok_s_before_after")


if __name__ == "__main__":
    main()
