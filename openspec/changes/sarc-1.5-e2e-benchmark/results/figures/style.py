"""Shared order, labels, colours and matplotlib style for the e2e prefill figures."""
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
# Six-GPU combined data (the five original GPUs are identical to ../cells.csv rows
# apart from the paired-bootstrap speedup CI; the 7900 XTX is added).
CELLS_CSV = REPORT / "raw6" / "cells.csv"
RUNS_CSV = REPORT / "raw6" / "runs_all.csv"
# Xclipse (M51): internal device, relative speedups only (no absolute tok/s anywhere).
M51_CSV = Path("/home/doremy/Desktop/sarc-acl/dev/1.5/executorch/openspec/changes/"
               "sarc-1.5-e2e-benchmark/contrib/m51/speedups.csv")
M51_PROMPT = "the_x2048"  # same prompt as the other GPUs

# GPU order: integrated / low power first, then discrete by vendor.
# GPUS: figures with absolute throughput (fig2). SPEEDUP_GPUS: speedup-only figures
# (fig1, fig3), which may also show speedup-only devices.
GPUS = ["780m", "b580", "b70", "4070ti", "orin", "7900xtx"]
SPEEDUP_GPUS = GPUS + ["m51"]
GPU_LABEL = {
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
    "7900xtx": "RX 7900 XTX†",
    "m51": "Xclipse (M51)‡",
}
# Two-line tick labels for narrow facets.
GPU_TICK = {
    "780m": "Radeon\n780M",
    "b580": "Arc\nB580",
    "b70": "Arc Pro\nB70",
    "4070ti": "RTX 4070\nTi S",
    "orin": "Jetson\nOrin Nano",
    "7900xtx": "RX 7900\nXTX†",
    "m51": "Xclipse\n(M51)‡",
}
# Three-line tick labels for the 7-group speedup bar chart.
GPU_TICK3 = {
    "780m": "Radeon\n780M",
    "b580": "Arc\nB580",
    "b70": "Arc Pro\nB70",
    "4070ti": "RTX\n4070\nTi S",
    "orin": "Jetson\nOrin\nNano",
    "7900xtx": "RX\n7900\nXTX†",
    "m51": "Xclipse\n(M51)‡",
}

# Compact single-line labels for the heatmap rows.
GPU_SHORT = {
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
    "7900xtx": "RX 7900 XTX†",
    "m51": "Xclipse (M51)‡",
}

# Pre-release GPUs: SARC rows unverified (ET_VK_SARC_UNVERIFIED=1), different .pte
# export (absolute tok/s not strictly comparable), AMDVLK driver. Marked everywhere by
# a tinted background band / outline in PRERELEASE_COLOR, white hatching and a dagger.
PRERELEASE = {"7900xtx", "m51"}
# Okabe-Ito reddish purple: distinct from the model and build colours (CVD dE >= 7.6
# over all pairs); always paired with hatching and the dagger as secondary encoding.
PRERELEASE_COLOR = "#CC79A7"
PRERELEASE_HATCH = "////"
PRERELEASE_NOTE = ("† RX 7900 XTX: pre-release. SARC rows unverified (ET_VK_SARC_UNVERIFIED=1); "
                   "different .pte export, so absolute tok/s are not strictly comparable; "
                   "AMDVLK driver.")
M51_NOTE = "‡ Xclipse (M51): internal device, relative speedups only; pre-release rows."
INCORRECT_NOTE = "4w on Xclipse (M51): not shown (kernel in development)."
NA_COLOR = "#D9D9D9"

MODELS = ["1b", "3b", "8b"]
MODEL_LABEL = {"1b": "Llama 3.2 1B", "3b": "Llama 3.2 3B", "8b": "Llama 3.1 8B"}
MODEL_SHORT = {"1b": "1B", "3b": "3B", "8b": "8B"}
# Okabe-Ito subset; validated for CVD separation (worst adjacent dE 11.4).
MODEL_COLOR = {"1b": "#E69F00", "3b": "#009E73", "8b": "#0072B2"}

SCHEMES = ["4w", "8da4w"]
SCHEME_LABEL = {
    "4w": "4w (int4 weights, fp16 activations)",
    "8da4w": "8da4w (int8 dyn. activations, int4 weights)",
}

BUILDS = ["stock", "sarc"]
BUILD_LABEL = {"stock": "Stock ExecuTorch 1.5", "sarc": "SARC 1.5-r2"}
BUILD_COLOR = {"stock": "#A6A6A6", "sarc": "#D55E00"}
BUILD_EDGE = {"stock": "#5E5E5E", "sarc": "#8A3C00"}

INK = "#222222"
MUTED = "#666666"
GRID = "#E3E3E3"

DOUBLE_COL = 7.2
SINGLE_COL = 3.5


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "hatch.linewidth": 0.8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType, editable text
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def mark_prerelease_band(ax, x_index, half_width=0.48):
    """Tinted background band behind a pre-release GPU group."""
    ax.axvspan(x_index - half_width, x_index + half_width, color=PRERELEASE_COLOR,
               alpha=0.13, lw=0, zorder=0)


def hatch_bars(bars, which):
    """White hatching on the bars whose GPU is pre-release (which: list of GPU ids)."""
    for bar, g in zip(bars, which):
        if g in PRERELEASE:
            bar.set_hatch(PRERELEASE_HATCH)
            bar.set_edgecolor("white")
            bar.set_linewidth(0)


def load_cells():
    return pd.read_csv(CELLS_CSV)


def load_timed_runs():
    runs = pd.read_csv(RUNS_CSV)
    return runs[runs["log"].str.startswith("logs/prefill")].copy()


def load_speedups():
    """Speedup table for SPEEDUP_GPUS: raw6 cells plus M51 speedup rows only.

    Cells whose SARC output is incorrect get speedup = NaN (never plotted); the column
    `correct` is False for them.
    """
    cols = ["gpu", "model", "scheme", "speedup", "speedup_ci_lo", "speedup_ci_hi"]
    base = load_cells()[cols].copy()
    base["correct"] = True
    m = pd.read_csv(M51_CSV)
    m = m[m["prompt"] == M51_PROMPT].copy()
    m["correct"] = m["sarc_output_correct"].str.strip().str.lower() == "yes"
    for c in ["speedup", "speedup_ci_lo", "speedup_ci_hi"]:
        m.loc[~m["correct"], c] = float("nan")
    out = pd.concat([base, m[cols + ["correct"]]], ignore_index=True)
    assert len(out) == len(SPEEDUP_GPUS) * len(MODELS) * len(SCHEMES), len(out)
    return out


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}")
    print(f"wrote {stem}.pdf / {stem}.png")
