"""r1: additional prefill-GEMM speedup from the two-day re-tuning over the previous best WMMA kernels.

Bars: 12-shape geomean (texture3d = model path, M = 2048, 3 repeats per cell).
Markers: per-model time-weighted speedup (per-layer call counts: 2*wq_wo + 2*wk_wv + 2*w1_w3 + w2).
Whiskers: per-shape min-max. Writes r1_kernel_gain.{pdf,png} and r1_kernel_gain_slide.{pdf,png}.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import common as C


def build(slide: bool):
    base = 18 if slide else 9
    C.setup_style(base)
    df = C.load()
    if slide:
        fig, ax = plt.subplots(figsize=(13.333, 7.5))
        fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.24)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.20)

    w = 0.36
    gap = 1.0
    ms = 7 if slide else 3.6
    lw_whisk = 2.0 if slide else 1.0
    offs = {"llama-3.2-1b": -0.09, "llama-3.2-3b": 0.0, "llama-3.1-8b": 0.09}
    xt, xl = [], []
    summary = {}
    for gi, gpu in enumerate(C.GPUS):
        col = C.GPU_COLOR[gpu]
        for si, sch in enumerate(C.SCHEMES):
            x = gi * gap * 1.0 + (si - 0.5) * (w + 0.06)
            g, lo, hi, n = C.kernel_geomean(df, gpu, sch)
            pm = C.per_model(df, gpu, sch)
            summary[(gpu, sch)] = (g, lo, hi, n, pm)
            regress = gpu == "4070TiS" and sch == "4w"
            unchanged = gpu == "780M" and sch == "8da4w"
            # grey = previous best (normalised to 1x); colour = additional gain
            ax.bar(x, min(g, 1.0), w, color=C.BEFORE_GREY, edgecolor="white", linewidth=0.8, zorder=2)
            if g > 1.0 and not unchanged:
                ax.bar(x, g - 1.0, w, bottom=1.0, color=col, edgecolor="white", linewidth=0.8, zorder=2)
            if regress:
                ax.bar(x, 1.0 - g, w, bottom=g, facecolor="white", edgecolor=col, hatch="////",
                       linewidth=1.0, zorder=2)
            # per-shape min-max whisker (thin) at bar centre
            ax.plot([x, x], [lo, hi], color=C.INK, lw=lw_whisk, zorder=3, solid_capstyle="butt")
            for mdl in C.MODELS:
                if mdl in pm:
                    ax.plot(x + offs[mdl] * (w / 0.36), pm[mdl], C.MODEL_MARKER[mdl], ms=ms,
                            mfc="white", mec=C.INK, mew=0.9 if not slide else 1.6, zorder=4)
            top = max(hi, g)
            lab = f"{g:.2f}×"
            ax.text(x, top + (0.07 if not slide else 0.08), lab, ha="center", va="bottom",
                    fontsize=base, fontweight="bold", color=C.INK, zorder=5)
            xt.append(x)
            xl.append(sch)
            if regress:
                ax.annotate("accuracy fix:\nold kernel failed\n8B check (tiles\nalone: 1.04×)" if not slide
                            else "accuracy fix:\nold kernel failed\n8B check",
                            xy=(x, top + (0.30 if not slide else 0.40)), xytext=(x - (0.02 if not slide else 0.0), top + (0.95 if not slide else 1.05)),
                            ha="center", va="bottom", fontsize=base, color=C.INK2,
                            arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8), zorder=5)
            if unchanged:
                ax.annotate("unchanged:\nno int8 headroom\non RDNA3",
                            xy=(x, top + (0.30 if not slide else 0.40)), xytext=(x + 0.02, top + (0.95 if not slide else 1.05)),
                            ha="center", va="bottom", fontsize=base, color=C.INK2,
                            arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8), zorder=5)
        # GPU label under the pair
        name = C.GPU_SHORT[gpu] + (" †" if gpu in C.OPT_IN else "")
        if gpu == "B580":
            name += "‡"
        ax.text(gi * gap, -0.085 if not slide else -0.10, name, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=base, fontweight="bold", color=C.INK)
        # colour key under the name
    ax.axhline(1.0, color=C.INK, lw=0.9 if not slide else 1.5, ls=(0, (4, 3)), zorder=4)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(C.GPUS) - 0.45)
    ymax = 4.9 if not slide else 5.2
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["0", "1×", "2×", "3×", "4×"])
    ax.set_ylabel("Kernel speedup, tuned / previous (×)")
    ax.grid(axis="y", color=C.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # legend
    mk = 8 if slide else 5
    handles = [
        Patch(facecolor=C.BEFORE_GREY, label="previous best WMMA (= 1×)"),
        Patch(facecolor="#6b6b6b", label="gain from re-tuning"),
        Line2D([], [], color=C.INK, lw=lw_whisk, label="per-shape range"),
    ] + [
        Line2D([], [], ls="", marker=C.MODEL_MARKER[m], mfc="white", mec=C.INK, ms=mk,
               label=C.MODEL_SHORT[m]) for m in C.MODELS
    ]
    title = ("Additional prefill-GEMM speedup from the two-day re-tuning,\n"
             "over the previous best WMMA kernels")
    if slide:
        fig.text(0.075, 0.975, "Two-day re-tuning added up to 2.5× over the previous best WMMA kernels", fontsize=24, fontweight="bold", va="top")
        fig.text(0.075, 0.905, "Prefill GEMMs of Llama 1B / 3B / 8B, M = 2048, texture3d (model path); bars = geomean of 12 shapes", fontsize=base, color=C.INK2, va="top")
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.865), ncol=6,
                         frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.5)
        foot = ("† previous WMMA kernel was opt-in; the default path ran tiled.   ‡ B580 before = B70 branch's Xe2 tiles.\n"
                "3 repeats per cell; Orin 4w bar = 11 shapes (8B w2 original numerically invalid).")
        fig.text(0.01, 0.015, foot, fontsize=base - 2 if False else base, color=C.INK2, va="bottom", wrap=True)
    else:
        fig.text(0.085, 0.985, title, fontsize=base + 1, fontweight="bold", va="top")
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.905), ncol=6,
                         frameon=False, handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
        foot = ("† previous WMMA kernel was opt-in (default path ran tiled).  ‡ B580 before = B70 branch's Xe2 tiles.\n"
                "Bars: geomean of 12 shapes (Orin 4w: 11). M = 2048, texture3d, 3 repeats per cell. Markers: per-model time-weighted.")
        fig.text(0.01, 0.012, foot, fontsize=base - 1 if False else base, color=C.INK2, va="bottom")
    C.save(fig, "r1_kernel_gain_slide" if slide else "r1_kernel_gain")
    return summary


if __name__ == "__main__":
    s = build(False)
    build(True)
    for k, (g, lo, hi, n, pm) in s.items():
        print(k, f"geomean {g:.3f} [{lo:.3f}-{hi:.3f}] n={n}",
              " ".join(f"{C.MODEL_SHORT[m]}={pm[m]:.3f}" for m in C.MODELS if m in pm))
