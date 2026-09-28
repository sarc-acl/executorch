"""e3: GEMM kernel speedup, GEMM share of stock time and end-to-end speedup vs model size."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from common import (ATTN, GEMM, GPUS, GPU_LABEL, INK, MODELS, MODEL_LABEL, MUTED, SCHEME_COLOR,
                    SCHEMES, WIDTH, apply_style, cells, families, family_ms, save)

apply_style()
F = families()
C = cells()

data = {}
print("gpu,scheme,model,gemm_speedup,gemm_share_stock,e2e_speedup_cells,e2e_trace,attn_speedup,attn_share_stock")
for gpu in GPUS:
    for scheme in SCHEMES:
        for m in MODELS:
            st = family_ms(F, gpu, m, scheme, "stock")
            sa = family_ms(F, gpu, m, scheme, "sarc")
            st_tot, sa_tot = sum(st.values()), sum(sa.values())
            st_att = sum(st[f] for f in ATTN)
            sa_att = sum(sa[f] for f in ATTN)
            e2e = C[(C.gpu == gpu) & (C.scheme == scheme) & (C.model == m)].speedup
            assert len(e2e) == 1
            d = dict(gemm=st[GEMM] / sa[GEMM], share=100 * st[GEMM] / st_tot, e2e=float(e2e.iloc[0]),
                     e2e_trace=st_tot / sa_tot, attn=st_att / sa_att, attn_share=100 * st_att / st_tot)
            data[gpu, scheme, m] = d
            print(f"{gpu},{scheme},{m},{d['gemm']:.2f},{d['share']:.1f},{d['e2e']:.2f},"
                  f"{d['e2e_trace']:.2f},{d['attn']:.2f},{d['attn_share']:.1f}")

x = [0, 1, 2]
fig, axes = plt.subplots(2, len(GPUS), figsize=(WIDTH, 4.9), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.25, 1]))
for j, gpu in enumerate(GPUS):
    top, bot = axes[0, j], axes[1, j]
    for scheme in SCHEMES:
        c = SCHEME_COLOR[scheme]
        g = [data[gpu, scheme, m]["gemm"] for m in MODELS]
        e = [data[gpu, scheme, m]["e2e"] for m in MODELS]
        s = [data[gpu, scheme, m]["share"] for m in MODELS]
        top.plot(x, g, color=c, lw=1.4, ls=(0, (3, 1.5)), marker="o", ms=5.5, mfc="white", mec=c, mew=1.3)
        top.plot(x, e, color=c, lw=2.0, marker="o", ms=5.5, mfc=c, mec="white", mew=0.8)
        bot.plot(x, s, color=c, lw=2.0, marker="s", ms=5, mfc=c, mec="white", mew=0.8)
        if gpu == "780m":
            a = [data[gpu, scheme, m]["attn"] for m in MODELS]
            sh = [data[gpu, scheme, m]["attn_share"] for m in MODELS]
            top.plot(x, a, color=c, lw=1.2, ls=":", marker="D", ms=4.5, mfc="white", mec=c, mew=1.1)
            bot.plot(x, sh, color=c, lw=1.2, ls=":", marker="D", ms=4.5, mfc="white", mec=c, mew=1.1)
        # Direct labels at 8B for the end-to-end speedup.
        e_other = data[gpu, SCHEMES[1 - SCHEMES.index(scheme)], "8b"]["e2e"]
        dy = 0
        if abs(e[-1] / e_other - 1) < 0.08:  # nudge apart labels that would collide
            dy = 6 if e[-1] >= e_other else -6
        top.annotate(f"{e[-1]:.2f}", (2, e[-1]), xytext=(4, dy), textcoords="offset points",
                     ha="left", va="center", fontsize=8.5, color=c, fontweight="bold")
    if gpu == "780m":
        top.text(0.75, 5.0, "attention", fontsize=9, color=MUTED, ha="left", va="center")
        bot.text(0.75, 14, "attention", fontsize=9, color=MUTED, ha="left", va="center")
    top.set_yscale("log")
    top.set_ylim(1.0, 13)
    top.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 8, 12]))
    top.yaxis.set_minor_locator(NullLocator())
    top.set_yticklabels(["1", "1.5", "2", "3", "5", "8", "12"] if j == 0 else [])
    bot.set_ylim(0, 100)
    bot.set_yticks([0, 25, 50, 75, 100])
    if j:
        bot.set_yticklabels([])
    for ax in (top, bot):
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_xlim(-0.3, 2.75)
    bot.set_xticks(x)
    bot.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    top.set_title(GPU_LABEL[gpu].replace("RTX 4070 Ti SUPER", "RTX 4070 Ti S.")
                  .replace("Jetson Orin Nano", "Orin Nano"), fontsize=9.5, fontweight="bold")

axes[0, 0].set_ylabel("speedup, stock / SARC\n(x, log scale)")
axes[1, 0].set_ylabel("share of stock\nGPU time (%)")
fig.supxlabel("model size (Llama 3.2 1B / 3B, Llama 3.1 8B), 2048-token prefill", fontsize=9, y=0.02)

handles = [
    Line2D([], [], color=SCHEME_COLOR["4w"], lw=2, label="4w"),
    Line2D([], [], color=SCHEME_COLOR["8da4w"], lw=2, label="8da4w"),
    Line2D([], [], color=INK, lw=1.4, ls=(0, (3, 1.5)), marker="o", mfc="white", mec=INK,
           label="GEMM kernel speedup"),
    Line2D([], [], color=INK, lw=2.0, marker="o", mfc=INK, mec="white", label="end-to-end speedup"),
    Line2D([], [], color=INK, lw=2.0, marker="s", mfc=INK, mec="white", label="GEMM share of stock time"),
    Line2D([], [], color=INK, lw=1.2, ls=":", marker="D", mfc="white", mec=INK,
           label="attention speedup / share (780M only)"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handletextpad=0.5, fontsize=9)
fig.tight_layout(rect=(0, 0.035, 1, 0.87), w_pad=0.4, h_pad=0.6)
save(fig, "e3_amdahl_model_size")
