"""r2_e2e_measured: MEASURED end-to-end prefill speedup, re-tuned vs previous best WMMA kernels,
per GPU x scheme x model (real-text 2048-token prompt, 5 interleaved repeats per build).
Bars = sarc_median / stock_median tok/s; error bars = paired bootstrap 95 % CI;
hollow markers = the earlier Amdahl projection from ../refinement.csv.
Writes r2_e2e_measured.{pdf,png} and r2_e2e_measured_slide.{pdf,png}."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C


def build(slide: bool):
    base = 18 if slide else 9
    C.setup_style(base)
    df = C.load()
    cells = C.load_cells()
    if slide:
        fig, axes = plt.subplots(2, 1, figsize=(13.333, 7.5), sharey=True)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.175, hspace=0.38)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.7), sharey=True)
        fig.subplots_adjust(left=0.085, right=0.99, top=0.845, bottom=0.165, hspace=0.62)
    w = 0.26
    stats = []
    for ax, sch in zip(axes, C.SCHEMES):
        for gi, gpu in enumerate(C.GPUS):
            col = C.GPU_COLOR[gpu]
            pr = C.projections(df, gpu, sch)
            tops = []
            for mi, mdl in enumerate(C.MODELS):
                x = gi + (mi - 1) * (w + 0.03)
                r = cells[(cells.G == gpu) & (cells.scheme == sch) & (cells.M == mdl)].iloc[0]
                v, lo, hi = r.speedup, r.speedup_ci_lo, r.speedup_ci_hi
                p = pr.get(mdl)
                stats.append((gpu, sch, mdl, v, lo, hi, p, r.stock_median, r.sarc_median))
                regress = gpu == "4070TiS" and sch == "4w" and v < 1.0
                ax.bar(x, min(v, 1.0), w, color=C.BEFORE_GREY, edgecolor="white", lw=0.6, zorder=2)
                if v > 1.005:
                    ax.bar(x, v - 1.0, w, bottom=1.0, color=col, edgecolor="white", lw=0.6, zorder=2)
                if regress:
                    ax.bar(x, 1.0 - v, w, bottom=v, facecolor="white", edgecolor=col, hatch="////",
                           lw=0.9, zorder=2)
                ax.plot([x, x], [lo, hi], color=C.INK, lw=1.8 if slide else 1.0, zorder=4,
                        solid_capstyle="butt")
                capw = w * 0.35
                for yy in (lo, hi):
                    ax.plot([x - capw / 2, x + capw / 2], [yy, yy], color=C.INK,
                            lw=1.8 if slide else 1.0, zorder=4)
                if p is not None:
                    ax.plot(x + w * 0.3, p, marker="o", ms=9 if slide else 4.5, mfc="white", mec=C.INK,
                            mew=1.6 if slide else 0.9, zorder=5)
                ytxt = max(v, hi, p or 0, 1.0) + (0.05 if slide else 0.035)
                ax.text(x, ytxt, f"{v:.2f}", ha="center", va="bottom", fontsize=base, color=C.INK,
                        fontweight="bold" if slide else "normal", zorder=6)
                tops.append(ytxt)
                ax.text(x, -0.04, C.MODEL_SHORT[mdl], ha="center", va="top", fontsize=base,
                        color=C.INK2, transform=ax.get_xaxis_transform())
            if gpu == "780M" and sch == "8da4w":
                ax.text(gi, max(tops) + (0.30 if slide else 0.22), "unchanged kernel", ha="center",
                        va="bottom", fontsize=base, color=C.INK2)
            if gpu == "4070TiS" and sch == "4w":
                ax.text(gi, max(tops) + (0.30 if slide else 0.22), "accuracy fix", ha="center",
                        va="bottom", fontsize=base, color=C.INK2)
            name = (C.GPU_SHORT[gpu] if not slide else C.GPU_SHORT[gpu]) + (" †" if gpu in C.OPT_IN else "")
            if not slide or sch == C.SCHEMES[-1]:
                ax.text(gi, -0.17 if not slide else -0.20, name, ha="center", va="top", fontsize=base,
                        fontweight="bold", transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color=C.INK, lw=1.4 if slide else 0.9, ls=(0, (4, 3)), zorder=3)
        ax.set_xlim(-0.5, len(C.GPUS) - 0.5)
        ax.set_ylim(0, 2.15)
        ax.set_yticks([0, 1.0, 2.0] if slide else [0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(["0", "1×", "2×"] if slide else ["0", "0.5×", "1×", "1.5×", "2×"])
        ax.set_xticks([])
        ax.grid(axis="y", color=C.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("speedup (×)" if slide else "e2e prefill speedup (×)")
        ax.set_title(sch, loc="left", fontweight="bold", fontsize=base + 1, pad=4)
        ax.spines["bottom"].set_visible(False)

    mk = 10 if slide else 5
    handles = [
        Patch(facecolor=C.BEFORE_GREY, label="previous best WMMA (= 1×)"),
        Patch(facecolor="#6b6b6b", label="measured gain (colour = GPU)"),
        Line2D([], [], color=C.INK, lw=1.8 if slide else 1.0, label="paired 95 % CI"),
        Line2D([], [], ls="", marker="o", mfc="white", mec=C.INK, ms=mk, label="projection"),
    ]
    if slide:
        fig.text(0.07, 0.975, "Measured end-to-end: re-tuned kernels speed up real prefill by up to 1.8×",
                 fontsize=24, fontweight="bold", va="top")
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.915), ncol=4,
                   frameon=False, handlelength=1.1, columnspacing=1.0, handletextpad=0.4)
        fig.text(0.01, 0.005, "2048-token real-text prompt, 5 interleaved repeats per build.  "
                 "† previous kernels were opt-in (default ran tiled).",
                 fontsize=base, color=C.INK2, va="bottom")
    else:
        fig.text(0.085, 0.99, "Measured end-to-end prefill speedup from the re-tuning (2048-token real-text prompt)",
                 fontsize=10, fontweight="bold", va="top")
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.955), ncol=4,
                   frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
        fig.text(0.01, 0.006,
                 "Bars: median tok/s re-tuned / previous, 5 interleaved repeats per build; CI = paired bootstrap.\n"
                 "Hollow markers: earlier Amdahl projection from kernel times (none for Orin 8B); within 3.1 % everywhere.\n"
                 "† previous WMMA kernels were opt-in (default path ran tiled); measured here with their opt-in env.",
                 fontsize=9, color=C.INK2, va="bottom")
    C.save(fig, "r2_e2e_measured_slide" if slide else "r2_e2e_measured")
    return stats


if __name__ == "__main__":
    st = build(False)
    build(True)
    worst = 0
    for gpu, sch, mdl, v, lo, hi, p, b, a in st:
        d = None if p is None else 100 * (p / v - 1)
        worst = max(worst, abs(d or 0))
        print(f"{gpu:8s} {sch:6s} {C.MODEL_SHORT[mdl]}  {b:9.1f} -> {a:9.1f} tok/s  "
              f"{v:.3f} [{lo:.3f},{hi:.3f}]  proj {p if p is None else round(p, 3)}  "
              f"diff {'' if d is None else f'{d:+.2f}%'}")
    print("max |projection - measured| =", round(worst, 2), "%")
