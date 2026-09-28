"""Fig. 3: speedup heatmap (GPU x scheme/model)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style as S

S.apply_style()
cells = S.load_cells().set_index(["gpu", "model", "scheme"])
cols = [(s, m) for s in S.SCHEMES for m in S.MODELS]
Z = np.array([[cells.loc[(g, m, s), "speedup"] for s, m in cols] for g in S.GPUS])

fig, ax = plt.subplots(figsize=(4.6, 3.1))
cmap = plt.get_cmap("cividis")
vmin, vmax = 1.0, np.ceil(Z.max() * 2) / 2
im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        r, g, b, _ = cmap((Z[i, j] - vmin) / (vmax - vmin))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="black" if lum > 0.5 else "white", zorder=4,
                bbox=None)
ax.set_xticks(range(len(cols)), [S.MODEL_SHORT[m] for _, m in cols])
ax.set_yticks(range(len(S.GPUS)), [S.GPU_SHORT[g] for g in S.GPUS])
ax.tick_params(length=0)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.axvline(2.5, color="white", lw=2)
for i, g in enumerate(S.GPUS):
    if g in S.PRERELEASE:
        # White hatching over the row, then a pre-release-colour outline.
        ax.add_patch(Rectangle((-0.5, i - 0.5), Z.shape[1], 1, fill=False,
                               hatch=S.PRERELEASE_HATCH, edgecolor="white", lw=0,
                               alpha=0.35, zorder=2))
        ax.add_patch(Rectangle((-0.5 + 0.03, i - 0.5 + 0.04), Z.shape[1] - 0.06, 0.92,
                               fill=False, edgecolor=S.PRERELEASE_COLOR, lw=2.0, zorder=3))
sec = ax.secondary_xaxis("top")
sec.set_xticks([1, 4], S.SCHEMES)
sec.tick_params(length=0)
sec.spines["top"].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
cb.set_label("speedup (×)")
cb.outline.set_visible(False)
fig.text(0.01, 0.0, "† RX 7900 XTX: pre-release. SARC rows unverified\n"
         "(ET_VK_SARC_UNVERIFIED=1); different .pte export, so absolute\n"
         "tok/s are not strictly comparable; AMDVLK driver.", ha="left", va="top", fontsize=8,
         color=S.MUTED)
fig.tight_layout()
S.save(fig, "fig3_heatmap")
