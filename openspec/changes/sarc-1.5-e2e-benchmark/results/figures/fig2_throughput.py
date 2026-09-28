"""Fig. 2: absolute prefill throughput, stock vs SARC (median, min-max whiskers, runs)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S

S.apply_style()
cells = S.load_cells().set_index(["gpu", "model", "scheme"])
runs = S.load_timed_runs()
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 3, figsize=(S.DOUBLE_COL, 5.0), sharey=True, sharex=True)
x = np.arange(len(S.GPUS))
width = 0.38

for r, scheme in enumerate(S.SCHEMES):
    for c, model in enumerate(S.MODELS):
        ax = axes[r, c]
        rows = cells.loc[[(g, model, scheme) for g in S.GPUS]]
        for j, build in enumerate(S.BUILDS):
            med = rows[f"{build}_median"].to_numpy()
            mn = rows[f"{build}_min"].to_numpy()
            mx = rows[f"{build}_max"].to_numpy()
            xs = x + (j - 0.5) * width
            bars = ax.bar(xs, med, width * 0.9, color=S.BUILD_COLOR[build], zorder=2)
            S.hatch_bars(bars, S.GPUS)
            ax.errorbar(xs, med, yerr=[med - mn, mx - med], fmt="none",
                        ecolor=S.INK, elinewidth=1.2, capsize=3.5, capthick=1.2,
                        zorder=4)
            for k, g in enumerate(S.GPUS):
                v = runs[(runs.gpu == g) & (runs.model == model)
                         & (runs.scheme == scheme) & (runs.build == build)].tok_s
                jit = rng.uniform(-0.3, 0.3, len(v)) * width
                ax.scatter(xs[k] + jit, v, s=4, facecolor="none",
                           edgecolor=S.BUILD_EDGE[build], linewidth=0.5, zorder=3)
        for k, g in enumerate(S.GPUS):
            if g in S.PRERELEASE:
                S.mark_prerelease_band(ax, k, g)
        ax.set_yscale("log")
        ax.set_xticks(x, [S.GPU_TICK1[g] for g in S.GPUS], rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.grid(True, which="major", axis="y")
        if r == 0:
            ax.set_title(S.MODEL_LABEL[model])
        if c == 0:
            ax.set_ylabel(f"{scheme}\nprefill throughput\n(tokens/s, log scale)")

axes[0, 0].set_ylim(10, 60000)
handles = [Patch(color=S.BUILD_COLOR[b], label=S.BUILD_LABEL[b]) for b in S.BUILDS]
handles += [Line2D([], [], color=S.INK, marker="_", markersize=6, lw=1.0,
                   label="min–max of n=5"),
            Line2D([], [], ls="none", marker="o", markersize=3, markerfacecolor="white",
                   markeredgecolor=S.MUTED, label="individual runs"),
            Patch(facecolor=S.PRERELEASE_COLOR, alpha=0.35, hatch=S.PRERELEASE_HATCH,
                  edgecolor="white", label="† pre-release")]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, -0.005,
         "2048-token prompt, ExecuTorch Vulkan (llama_main). Bars: median; "
         "whiskers: min–max of n=5 runs; dots: individual runs.\n"
         + S.PRERELEASE_NOTE_WIDE,
         ha="center", va="top", fontsize=8, color=S.MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95))
S.save(fig, "fig2_throughput")
