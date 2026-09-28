"""Fig. 1: SARC speedup over stock ExecuTorch 1.5 (median ratio, approx. 95 % paired bootstrap CI)."""
import matplotlib.pyplot as plt
import numpy as np

import style as S

S.apply_style()
cells = S.load_cells().set_index(["gpu", "model", "scheme"])

fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE_COL, 3.0), sharey=True)
x = np.arange(len(S.GPUS))
width = 0.26
ymax = cells["speedup_ci_hi"].max()

for ax, scheme in zip(axes, S.SCHEMES):
    for i, model in enumerate(S.MODELS):
        rows = cells.loc[[(g, model, scheme) for g in S.GPUS]]
        s = rows["speedup"].to_numpy()
        lo = s - rows["speedup_ci_lo"].to_numpy()
        hi = rows["speedup_ci_hi"].to_numpy() - s
        xs = x + (i - 1) * width
        bars = ax.bar(xs, s, width * 0.92, color=S.MODEL_COLOR[model],
                      label=S.MODEL_LABEL[model], zorder=2)
        S.hatch_bars(bars, S.GPUS)
        ax.errorbar(xs, s, yerr=[lo, hi], fmt="none", ecolor=S.INK,
                    elinewidth=0.9, capsize=2, capthick=0.9, zorder=3)
        for xx, v, top in zip(xs, s, s + hi):
            ax.text(xx, top + 0.08, f"{v:.1f}×", ha="center", va="bottom",
                    rotation=90, fontsize=8, color=S.INK)
    for k, g in enumerate(S.GPUS):
        if g in S.PRERELEASE:
            S.mark_prerelease_band(ax, k)
    ax.axhline(1.0, color=S.INK, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x, [S.GPU_TICK[g] for g in S.GPUS])
    ax.tick_params(axis="x", length=0)
    ax.set_title(S.SCHEME_LABEL[scheme], loc="left")
    ax.set_xlim(-0.55, len(S.GPUS) - 0.45)

axes[0].set_ylabel("prefill speedup, SARC / stock (×)")
axes[0].set_ylim(0, np.ceil(ymax + 0.9))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, -0.01,
         "2048-token prompt, ExecuTorch Vulkan (llama_main). Bars: ratio of medians of n=5 runs;\n"
         "error bars: approx. 95 % paired bootstrap CI; dashed line: 1.0× (stock).\n"
         + S.PRERELEASE_NOTE.replace("; different", ";\ndifferent"),
         ha="center", va="top", fontsize=8, color=S.MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.93))
S.save(fig, "fig1_speedup")
