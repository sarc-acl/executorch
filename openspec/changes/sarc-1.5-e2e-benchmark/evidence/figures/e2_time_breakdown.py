"""e2: 8B prefill GPU time by kernel family, stock vs SARC, normalised to stock = 100 %."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import (PRERELEASE_NOTE, ATTN, GEMM, GPUS, GPU_LABEL, INK, MUTED, QUANT, SCHEMES, WIDTH, apply_style,
                    SARC_ATTN, families, family_ms, save)

apply_style()
F = families()
MODEL = "8b"

SEGS = [  # (label, families, colour, text colour)
    ("GEMM (linear layers)", [GEMM], "#2B5C8A", "white"),
    ("8-bit act. quantize", [QUANT], "#9ECAE1", INK),
    ("attn QK^T", ["attention: QK^T"], "#54278F", "white"),
    ("attn softmax", ["attention: softmax"], "#8C85C2", "white"),
    ("attn AV + KV update", ["attention: AV", "attention: KV update"], "#C9C6E3", INK),
    ("everything else", None, "#DADADA", INK),
]


def split(ms):
    total = sum(ms.values())
    named = set(sum((s[1] for s in SEGS if s[1]), []))
    out = []
    for _, fams, _, _ in SEGS:
        out.append(sum(ms.get(f, 0.0) for f in fams) if fams else
                   sum(v for k, v in ms.items() if k not in named))
    assert abs(sum(out) - total) < 1e-6
    return out, total


rows = []  # (y, gpu, scheme, build, segs, total, stock_total)
y = 0.0
yticks, yticklabels, group_y = [], [], []
for gpu in GPUS:
    y -= 0.45  # room for the GPU header
    top = y
    for scheme in SCHEMES:
        st_segs, st_total = split(family_ms(F, gpu, MODEL, scheme, "stock"))
        sa_segs, sa_total = split(family_ms(F, gpu, MODEL, scheme, "sarc"))
        for build, segs, total in (("stock", st_segs, st_total), ("sarc", sa_segs, sa_total)):
            rows.append((y, gpu, scheme, build, segs, total, st_total))
            yticks.append(y)
            yticklabels.append(f"{scheme} {'stock' if build == 'stock' else 'SARC'}")
            y -= 1.0
        y -= 0.35
    group_y.append((top, y + 0.35, gpu))
    y -= 0.55

fig, ax = plt.subplots(figsize=(WIDTH, 11.8))
print("gpu,scheme,build,total_ms,gemm_pct_of_stock,attn_pct_of_own,quant_pct_of_own")
for yy, gpu, scheme, build, segs, total, st_total in rows:
    left = 0.0
    for (label, fams, col, tcol), v in zip(SEGS, segs):
        w = 100.0 * v / st_total
        if w <= 0:
            continue
        ax.barh(yy, w, left=left, height=0.78, color=col, edgecolor="white", linewidth=0.8)
        if label.startswith("GEMM") and w >= 7:
            ax.text(left + w / 2, yy, f"{w:.0f}%", ha="center", va="center", color=tcol, fontsize=9)
        left += w
    attn = sum(v for (lab, *_), v in zip(SEGS, segs) if lab.startswith("attn")) / total * 100
    quant = segs[1] / total * 100
    txt = f"{total:,.0f}" if build == "stock" else f"{total:,.0f} ({st_total / total:.2f}x)"
    ax.text(103, yy, txt, ha="left", va="center", fontsize=9,
            color=INK if build == "stock" else MUTED)
    ax.text(147, yy, f"{attn:.0f}%", ha="right", va="center", fontsize=9,
            color=INK, fontweight="bold" if build == "sarc" else "normal")
    print(f"{gpu},{scheme},{build},{total:.1f},{100 * segs[0] / st_total:.1f},{attn:.1f},{quant:.1f}")

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.tick_params(axis="y", length=0)
for top, bottom, gpu in group_y:
    ax.text(0, top + 0.62, GPU_LABEL[gpu], ha="left", va="center", fontsize=9.5, fontweight="bold")
    if gpu in SARC_ATTN:
        sp = []
        for scheme in SCHEMES:
            st = family_ms(F, gpu, MODEL, scheme, "stock")
            sa = family_ms(F, gpu, MODEL, scheme, "sarc")
            sp.append(sum(st[f] for f in ATTN) / sum(sa[f] for f in ATTN))
        ax.text(98, top - 1.0, f"SARC attention kernels:\nattention {sp[0]:.1f}x (4w), {sp[1]:.1f}x (8da4w)",
                ha="right", va="center", fontsize=9, color=INK, linespacing=1.15)
ax.set_xlim(0, 100)
ax.set_ylim(y + 0.3, 0.2)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("prefill GPU time, % of the stock build's time (same GPU and scheme)")
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color="#E6E6E6", lw=0.6)
ax.axvline(100, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(103, 0.25, "GPU time, ms\n(SARC: speedup)", ha="left", va="bottom", fontsize=9,
        color=MUTED, clip_on=False)
ax.text(147, 0.25, "attention\nshare", ha="right", va="bottom", fontsize=9, color=MUTED,
        clip_on=False)

handles = [Patch(facecolor=c, edgecolor="white", label=l) for l, _, c, _ in SEGS]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.47, 1.0),
           columnspacing=1.2, handletextpad=0.4)
fig.text(0.0, 0.0, PRERELEASE_NOTE, fontsize=9,
         color=MUTED, ha="left", va="top")
fig.subplots_adjust(left=0.17, right=0.70, top=0.915, bottom=0.05)
save(fig, "e2_time_breakdown")
