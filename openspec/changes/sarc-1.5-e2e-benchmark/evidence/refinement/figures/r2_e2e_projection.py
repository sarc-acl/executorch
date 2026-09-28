"""r2_e2e_projection (superseded by r2_e2e_measured): Amdahl-projected end-to-end prefill speedup from the re-tuning,
with Orin's measured e2e (2048-token prompt, 3 repeats) overlaid to show the projection's accuracy."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C


def main():
    C.setup_style(9)
    df = C.load()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.855, bottom=0.155, hspace=0.62)
    w = 0.26
    out = {}
    for ax, sch in zip(axes, C.SCHEMES):
        for gi, gpu in enumerate(C.GPUS):
            col = C.GPU_COLOR[gpu]
            pr = C.projections(df, gpu, sch)
            me = C.measured_e2e(df, gpu, sch)
            for mi, mdl in enumerate(C.MODELS):
                x = gi + (mi - 1) * (w + 0.03)
                ax.text(x, -0.04, C.MODEL_SHORT[mdl], ha="center", va="top", fontsize=9,
                        color=C.INK2, transform=ax.get_xaxis_transform())
                if mdl not in pr:
                    lines = ["Orin: measured / proj."] + [
                        f"{C.MODEL_SHORT[k]}  {me[k]:.3f} / {pr[k]:.3f}" for k in C.MODELS if k in me]
                    ax.text(x - w / 2, 0.30, "\n".join(lines),
                            ha="left", va="bottom", fontsize=9, color=C.INK, linespacing=1.15)
                    ax.text(x - w / 2, 0.08, "8B: not run", ha="left",
                            va="bottom", fontsize=9, color=C.MUTED)
                    continue
                v = pr[mdl]
                out[(gpu, sch, mdl)] = (v, me.get(mdl))
                regress = v < 1.0 and gpu == "4070TiS"
                ax.bar(x, min(v, 1.0), w, color=C.BEFORE_GREY, edgecolor="white", lw=0.6, zorder=2)
                if v > 1.0 and abs(v - 1) >= 0.005:
                    ax.bar(x, v - 1.0, w, bottom=1.0, color=col, edgecolor="white", lw=0.6, zorder=2)
                if regress:
                    ax.bar(x, 1.0 - v, w, bottom=v, facecolor="white", edgecolor=col, hatch="////",
                           lw=0.9, zorder=2)
                lab = f"{v:.2f}"
                ytxt = max(v, 1.0) + 0.03
                if mdl in me:
                    m = me[mdl]
                    ytxt = max(v, m, 1.0) + 0.07
                    ax.plot(x, m, marker="D", ms=5.5, mfc=C.INK, mec="white", mew=0.8, zorder=5)
                    ax.text(x, ytxt, lab, ha="center", va="bottom", fontsize=9, color=C.INK)
                else:
                    ax.text(x, ytxt, lab, ha="center", va="bottom", fontsize=9, color=C.INK)
            name = C.GPU_SHORT[gpu] + (" †" if gpu in C.OPT_IN else "")
            ax.text(gi, -0.16, name, ha="center", va="top", fontsize=9, fontweight="bold",
                    transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color=C.INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.5, len(C.GPUS) + 0.55)
        ax.set_ylim(0, 2.1)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(["0", "0.5×", "1×", "1.5×", "2×"])
        ax.set_xticks([])
        ax.grid(axis="y", color=C.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("e2e prefill speedup (×)")
        ax.set_title(f"{sch}", loc="left", fontweight="bold", fontsize=10, pad=4)
        ax.spines["bottom"].set_visible(False)

    fig.text(0.085, 0.985, "Projected end-to-end prefill speedup from the re-tuning (2048-token prompt)",
             fontsize=10, fontweight="bold", va="top")
    handles = [
        Patch(facecolor=C.BEFORE_GREY, label="previous best WMMA (= 1×)"),
        Patch(facecolor="#6b6b6b", label="projected gain (Amdahl; bar colour = GPU)"),
        Line2D([], [], ls="", marker="D", mfc=C.INK, mec="white", ms=6, label="measured e2e (Orin only)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.955), ncol=3, frameon=False,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    fig.text(0.01, 0.008,
             "Projection = measured tuned e2e + layers × Σ calls × (before − after) kernel time; release-1.4 branches, 3 e2e repeats.\n"
             "† previous WMMA kernel was opt-in (default path ran tiled). Orin 8B: e2e stopped by the memory guard.",
             fontsize=9, color=C.INK2, va="bottom")
    C.save(fig, "r2_e2e_projection")
    return out


if __name__ == "__main__":
    for k, (p, m) in main().items():
        print(k, f"projected {p:.3f}", "" if m is None else f"measured {m:.3f} diff {100*(p/m-1):+.2f}%")
