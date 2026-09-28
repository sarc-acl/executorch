"""Fig. 1: SARC speedup over stock ExecuTorch 1.5 (median ratio, approx. 95 % paired bootstrap CI).

Speedup-only figure: includes the speedup-only devices in S.SPEEDUP_GPUS. Cells whose
SARC output is incorrect are not plotted; their slot is labelled instead.
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S

S.apply_style()
cells = S.load_speedups().set_index(["gpu", "model", "scheme"])
GPUS = S.SPEEDUP_GPUS

# Two rows (4w above 8da4w) with shared x and y: eight GPU groups do not fit legibly side by side.
fig, axes = plt.subplots(2, 1, figsize=(S.DOUBLE_COL, 5.6), sharey=True, sharex=True)
x = np.arange(len(GPUS))
width = 0.27
ymax = cells["speedup_ci_hi"].max()
ytop = np.ceil(ymax + 0.9)

for ax, scheme in zip(axes, S.SCHEMES):
    for i, model in enumerate(S.MODELS):
        rows = cells.loc[[(g, model, scheme) for g in GPUS]]
        ok = rows["correct"].to_numpy()
        s = rows["speedup"].to_numpy()
        lo = s - rows["speedup_ci_lo"].to_numpy()
        hi = rows["speedup_ci_hi"].to_numpy() - s
        xs = x + (i - 1) * width
        bars = ax.bar(xs[ok], s[ok], width * 0.92, color=S.MODEL_COLOR[model],
                      label=S.MODEL_LABEL[model], zorder=2)
        S.hatch_bars(bars, [g for g, k in zip(GPUS, ok) if k])
        ax.errorbar(xs[ok], s[ok], yerr=[lo[ok], hi[ok]], fmt="none", ecolor=S.INK,
                    elinewidth=0.9, capsize=2, capthick=0.9, zorder=3)
        for xx, v, top in zip(xs[ok], s[ok], (s + hi)[ok]):
            ax.text(xx, top + 0.08, f"{v:.1f}×", ha="center", va="bottom",
                    rotation=90, fontsize=8, color=S.INK)
    for k, g in enumerate(GPUS):
        if g in S.PRERELEASE:
            S.mark_prerelease_band(ax, k, g)
        if not cells.loc[[(g, m, scheme) for m in S.MODELS], "correct"].any():
            ax.text(k, (1.0 + ytop) / 2 + 0.3, f"{scheme}: re-measuring,\nnot shown", rotation=90,
                    ha="center", va="center", fontsize=8, color=S.INK)
    ax.axhline(1.0, color=S.INK, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x, [S.GPU_TICK[g] for g in GPUS])
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="x", length=0)
    ax.set_title(S.SCHEME_LABEL[scheme], loc="left")
    ax.set_xlim(-0.55, len(GPUS) - 0.45)

for ax in axes:
    ax.set_ylabel("prefill speedup,\nSARC / stock (×)")
axes[0].set_ylim(0, ytop)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, -0.01,
         "2048-token prompt, ExecuTorch Vulkan (llama_main). Bars: ratio of medians of n=5 runs;\n"
         "error bars: approx. 95 % paired bootstrap CI; dashed line: 1.0× (stock).\n"
         + S.PRERELEASE_NOTE_WIDE + "\n"
         + S.M51_NOTE + "\n" + S.INCORRECT_NOTE,
         ha="center", va="top", fontsize=8, color=S.MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.2)
S.save(fig, "fig1_speedup")
