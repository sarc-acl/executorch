"""r3: the work. Per GPU, each profiler-/roofline-guided change from the lessons docs'
"What changed" tables, with its evidence and its measured effect, ending in the confirmed net.

Step numbers are quoted from igpu-roofline/docs/*-WMMA-LESSONS.md (line numbers in CAPTIONS.md
and in the `cite` fields below). Net rows come from ../refinement.csv (12-shape geomean,
texture3d, 3 repeats). Kinds:
  chain  - measured on the same base; the chained bars multiply to the net (780M 4w, 4070 Ti 4w)
  alone  - factor measured on its own base (screen / control / single shape); drawn from 1x
  range  - doc gives a range only; drawn as a floating range bar from 1x
  text   - doc gives no speedup number for the step
  net    - confirmed net from the CSV
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import common as C

D780 = "780M-WMMA-LESSONS.md"
DXE = "XE2-WMMA-LESSONS.md"
D40 = "4070TI-WMMA-LESSONS.md"
DJ = "JETSON-WMMA-LESSONS.md"

# (scheme, label, evidence, kind, value(s), note, cite)
STEPS = {
    "780M": [
        ("4w", "ACC_FP32 (fp32 accumulator)", "ISA + roofline", "chain", 1.09,
         "×1.09; VALU 637→330", f"{D780} L69"),
        ("4w", "CSH_IN_ASH (drain in dead LDS)", "occupancy arithmetic", "chain", 1.19,
         "×1.19; 4→6 waves/SIMD", f"{D780} L70"),
        ("4w", "one tile for all shapes", "sweep", "text", None,
         "3B: 8.2 vs 10.8 TFLOP/s", f"{D780} L71"),
        ("4w", "net, confirmed", "", "net", None, "", f"{D780} L26; CSV"),
        ("8da4w", "analysed, not changed", "roofline", "text", None,
         "int8 roof only 1.23× dot roof", f"{D780} L107-109"),
        ("8da4w", "net (identical kernel)", "", "net", None, "", f"{D780} L28; CSV"),
    ],
    "B580": [
        ("4w", "subgroup 16 + 32×32 tiles", "ISA spills; roofline feed", "alone", 36.9 / 19.2,
         "1B 19.2→36.9 TFLOP/s", f"{DXE} L48-49"),
        ("4w", "IMG_A (A via imageLoad)", "storage A/B", "range", (1.06, 1.29),
         "+6–29 %, tile dependent", f"{DXE} L52"),
        ("4w", "FRAG_LAYOUT (no LDS padding)", "OA: 0 bank conflicts", "alone", 1.12,
         "+12 %, buffer only", f"{DXE} L51"),
        ("4w", "net, confirmed", "", "net", None, "", "CSV (doc states no ratio)"),
        ("8da4w", "WG_TILE_M = 256", "sweep", "alone", 52.7 / 32.4,
         "1B 32.4→52.7 TOP/s", f"{DXE} L50"),
        ("8da4w", "net, confirmed", "", "net", None, "", "CSV (doc states no ratio)"),
        ("both", "made default, buffer added", "dispatch query", "text", None,
         "model path runs WMMA", f"{DXE} L53-54"),
    ],
    "B70": [
        ("4w", "B580 tiles re-screened", "1-run screen", "alone", 62.5 / 36.1,
         "1B 36.1→62.5 TFLOP/s", f"{DXE} L103-106"),
        ("4w", "FRAG_LAYOUT", "OA: SLM 42 % of roof", "range", (1.00, 1.04),
         "+0–4 % (B580: +12 %)", f"{DXE} L126-128"),
        ("4w", "net, confirmed", "", "net", None, "", f"{DXE} L115; CSV"),
        ("8da4w", "B580 tiles re-screened", "1-run screen", "alone", 87.7 / 53.1,
         "1B 53.1→87.7 TOP/s", f"{DXE} L103-106"),
        ("8da4w", "G31 8B-only tile removed", "3-repeat A/B", "range", (1.56, 1.73),
         "1.56–1.73×, 8B shapes", f"{DXE} L106-108"),
        ("8da4w", "net, confirmed", "", "net", None, "", f"{DXE} L116; CSV"),
    ],
    "4070TiS": [
        ("8da4w", "WG_TILE_K 32→64", "ablation + LDS-fed roof", "alone", 1.10,
         "×1.10 (buffer ×1.85)", f"{D40} L84"),
        ("8da4w", "A_RAW (raw 16-B A staging)", "ablation: A store 27 %", "alone", 303 / 252,
         "3B wq_wo 303→252 µs", f"{D40} L71-75, L85"),
        ("8da4w", "B_PAIR + CSH_IN_ASH", "ablation; LDS budget", "text", None,
         "½ B fetches; fits 48 KiB", f"{D40} L86-87"),
        ("8da4w", "net, confirmed", "", "net", None, "", f"{D40} L38; CSV"),
        ("4w", "per-shape tiles", "12-tile screen + nsys", "chain", 1.036,
         "×1.036", f"{D40} L40, L109-115; CSV step 1"),
        ("4w", "ACC_GROUP_FP32 (accuracy)", "8B w2 err 1.49 > 0.5", "chain", 0.924,
         "×0.924, correctness", f"{D40} L121-135; CSV step 2"),
        ("4w", "net, confirmed", "", "net", None, "", f"{D40} L40; CSV"),
    ],
    "Orin": [
        ("4w", "M256/g22 tile where aligned", "pipeline stats + screen", "alone", 1.06,
         "≈1.06× (M256/g42: 0.88×)", f"{DJ} L164-168"),
        ("4w", "fp32 tile for K > 8192", "8B w2 failed check", "alone", 40.784 / 45.881,
         "8B w2 only; excluded", f"{DJ} L133-138, L197-201"),
        ("4w", "net (11 shapes)", "", "net", None, "", f"{DJ} L194; CSV"),
        ("8da4w", "raw A + paired B + LDS reuse", "Nsight: Tensor 9.9 %", "alone", 2.31,
         "2.31× (128×64 control)", f"{DJ} L95-98, L158-160"),
        ("8da4w", "K32→K64", "screen control", "alone", 1.076,
         "×1.076 more", f"{DJ} L160-161"),
        ("8da4w", "net, confirmed", "", "net", None, "", f"{DJ} L195; CSV"),
        ("8da4w", "after: Tensor Active 20.9 %", "Nsight", "text", None,
         "staging overhead cut", f"{DJ} L100-101"),
    ],
}

TITLE_NOTE = {
    "780M": "before = previous default kernel",
    "B580": "before = B70 branch's Xe2 tiles, opt-in",
    "B70": "before = previous B70 tiles, opt-in",
    "4070TiS": "before = previous branch kernels, opt-in",
    "Orin": "before = 4070 Ti branch kernels, explicitly enabled",
}

XMIN, XMAX = 0.8, 3.0


def main():
    from matplotlib import transforms
    C.setup_style(9)
    df = C.load()
    nets = {(g, s): C.kernel_geomean(df, g, s)[0] for g in C.GPUS for s in C.SCHEMES}
    heights = [len(STEPS[g]) for g in C.GPUS]
    fig = plt.figure(figsize=(7.2, 10.0))
    X_EV, X_NOTE = 0.56, 0.765
    gs = fig.add_gridspec(len(C.GPUS), 1, height_ratios=heights, hspace=0.42,
                          left=0.325, right=0.545, top=0.885, bottom=0.10)
    bh = 0.64
    axes = []
    for gi, gpu in enumerate(C.GPUS):
        ax = fig.add_subplot(gs[gi], sharex=axes[0] if axes else None)
        axes.append(ax)
        tr = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        col = C.GPU_COLOR[gpu]
        steps = STEPS[gpu]
        n = len(steps)
        cum = {"4w": 1.0, "8da4w": 1.0}
        for ri, (sch, label, ev, kind, val, note, cite) in enumerate(steps):
            y = n - 1 - ri
            if kind == "chain":
                a, b = cum[sch], cum[sch] * val
                cum[sch] = b
                lo, hi = min(a, b), max(a, b)
                if b < a:
                    ax.add_patch(Rectangle((lo, y - bh / 2), hi - lo, bh, facecolor="white",
                                           edgecolor=col, hatch="////", lw=1.0, zorder=3))
                else:
                    ax.add_patch(Rectangle((lo, y - bh / 2), hi - lo, bh, facecolor=col,
                                           alpha=0.45, lw=0, zorder=3))
                    ax.add_patch(Rectangle((lo, y - bh / 2), hi - lo, bh, facecolor="none",
                                           edgecolor=col, lw=1.0, zorder=3))
            elif kind == "alone":
                lo, hi = min(1.0, val), max(1.0, val)
                hatch = "////" if val < 1 else None
                ax.add_patch(Rectangle((lo, y - bh / 2), hi - lo, bh, facecolor="white",
                                       edgecolor=col, hatch=hatch, lw=1.1, zorder=3))
            elif kind == "range":
                lo, hi = val
                ax.add_patch(Rectangle((lo, y - bh / 2), max(hi - lo, 0.004), bh, facecolor="white",
                                       edgecolor=col, lw=1.1, ls=(0, (2, 1.2)), zorder=3))
            elif kind == "net":
                v = nets[(gpu, sch)]
                lo, hi = min(v, 1.0), max(v, 1.0)
                ax.add_patch(Rectangle((lo, y - bh / 2), max(hi - lo, 0.004), bh, facecolor=col,
                                       lw=0, zorder=3))
                ax.text(hi * 1.02, y, f"{v:.2f}×", ha="left", va="center", fontsize=9,
                        fontweight="bold", color=C.INK, zorder=5)
            sch_lab = "" if sch == "both" else f"{sch} · "
            ax.text(0.315, y, f"{sch_lab}{label}", transform=tr, ha="right", va="center",
                    fontsize=9, color=C.INK, fontweight="bold" if kind == "net" else "normal")
            if ev:
                ax.text(X_EV, y, ev, transform=tr, ha="left", va="center", fontsize=9,
                        color=C.MUTED, style="italic")
            if note:
                ax.text(X_NOTE, y, note, transform=tr, ha="left", va="center", fontsize=9,
                        color=C.INK2)
        ax.axvline(1.0, color=C.INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ticks = [0.8, 1.0, 1.5, 2.0, 3.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}×" for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(axis="x", color=C.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if gi < len(C.GPUS) - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)
            ax.spines["bottom"].set_visible(False)
        name = C.GPU_NAME[gpu] + (" †" if gpu in C.OPT_IN else "")
        ytop = n - 0.45
        ax.add_patch(Rectangle((0.015, ytop + 0.25), 0.012, 0.75, transform=tr, facecolor=col,
                               clip_on=False, lw=0))
        ax.text(0.034, ytop + 0.28, name, transform=tr, ha="left", va="bottom", fontsize=10,
                fontweight="bold", color=C.INK)
        ax.text(0.325, ytop + 0.28, TITLE_NOTE[gpu], transform=tr, ha="left", va="bottom",
                fontsize=9, color=C.INK2)
    axes[-1].set_xlabel("Speedup over the previous kernel (×, log scale)")
    fig.text(0.015, 0.993, "The work: each profiler- or roofline-guided change, its evidence and effect",
             fontsize=10, fontweight="bold", va="top")
    handles = [
        Patch(facecolor=C.MUTED, alpha=0.45, edgecolor=C.INK2, label="step, chained (multiply to net)"),
        Patch(facecolor="white", edgecolor=C.INK2, label="step on its own base (screen/control)"),
        Patch(facecolor="white", edgecolor=C.INK2, ls=(0, (2, 1.2)), label="doc gives a range only"),
        Patch(facecolor="white", edgecolor=C.INK2, hatch="////", label="accuracy fix (slower, correct)"),
        Patch(facecolor=C.INK2, label="net, confirmed (geomean of shapes)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.972), ncol=3, frameon=False,
               handlelength=1.4, columnspacing=0.8, handletextpad=0.4, fontsize=9)
    hy = 0.905
    fig.text(0.315, hy, "change", ha="right", va="bottom", fontsize=9, fontweight="bold", color=C.INK2)
    fig.text(0.435, hy, "effect (×)", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C.INK2)
    fig.text(X_EV, hy, "evidence", ha="left", va="bottom", fontsize=9, fontweight="bold", color=C.INK2)
    fig.text(X_NOTE, hy, "numbers from the doc", ha="left", va="bottom", fontsize=9, fontweight="bold", color=C.INK2)
    fig.text(0.015, 0.006,
             "Steps: per-GPU lessons docs (line numbers in CAPTIONS.md); screens are single 1B runs, so steps\n"
             "on their own base do not multiply to the net. Net: refinement.csv, texture3d, M = 2048, 3 repeats.\n"
             "† previous WMMA kernel was opt-in (default path ran tiled).",
             fontsize=9, color=C.INK2, va="bottom")
    C.save(fig, "r3_steps")
    return nets


if __name__ == "__main__":
    nets = main()
    for g in C.GPUS:
        for s, label, ev, kind, val, note, cite in STEPS[g]:
            v = nets.get((g, s)) if kind == "net" else val
            print(g, s, kind, label, v if not isinstance(v, float) else round(v, 3), "|", cite)
