"""Shared projection-first style, labels, colours and helpers for the talk slides.

Canvas is a fixed 13.333 x 7.5 in (16:9); figures are never saved with a tight
bounding box, so every export keeps the exact slide aspect ratio.
"""
import os
from math import floor, log10
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
# Variant: "6gpu" (default; adds the pre-release RX 7900 XTX from raw6/) or "5gpu"
# (the original five GPUs from ../cells.csv). The *_5gpu.* files in this folder are the
# archived renders of the earlier 5-GPU revision; a 5gpu run writes *_5gpu_regen.*.
VARIANT = os.environ.get("SLIDES_VARIANT", "6gpu")
assert VARIANT in ("5gpu", "6gpu"), VARIANT
if VARIANT == "6gpu":
    CELLS_CSV = HERE.parent / "raw6" / "cells.csv"
    SUFFIX = ""
else:
    CELLS_CSV = HERE.parent / "cells.csv"
    SUFFIX = "_5gpu_regen"  # never overwrites the archived *_5gpu.* files

W, H = 13.333, 7.5

# Build colours (identical on every slide).
STOCK = "#9A9FA5"
OURS = "#E8590C"
STOCK_LABEL = "ExecuTorch 1.5"
OURS_LABEL = "+ tuned WMMA kernels"

INK = "#1F2328"
MUTED = "#6B7075"
FAINT = "#A4A9AE"
GRID = "#E6E8EA"

# Font sizes (pt). Floors from the brief: axis/tick/value >= 20, annotations >= 18.
FS_TICK = 22
FS_VALUE = 20
FS_BIG = 28
FS_ANNOT = 18
FS_GROUP = 18
FS_FOOT = 14

FOOTNOTE = "2048-token prompt · ExecuTorch Vulkan · median of 5 runs"

if VARIANT == "6gpu":
    GPUS = ["780m", "b580", "b70", "4070ti", "7900xtx", "orin"]
else:
    GPUS = ["780m", "b580", "b70", "4070ti", "orin"]
# Pre-release rows: unverified kernel rows and a different model export. Drawn with
# hollow / outlined orange marks and a dagger on the GPU name.
PRERELEASE = {"7900xtx"} & set(GPUS)
DAGGER_NOTE = "† RX 7900 XTX: pre-release kernel rows, different model export"
GPU_LABEL = {
    "7900xtx": "RX 7900 XTX †",
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
}
GPU_CLASS = {
    "780m": "Laptop iGPU",
    "b580": "Desktop GPU",
    "b70": "Desktop GPU",
    "4070ti": "Desktop GPU",
    "7900xtx": "Desktop GPU",
    "orin": "Edge",
}
# Okabe-Ito subset, only for s3 (the one figure that needs GPU identity by colour).
# validate_palette.js (light): all checks pass; worst adjacent CVD dE 7.6, so every
# line also carries a distinct marker shape and a direct name label.
GPU_COLOR = {
    "780m": "#D55E00",
    "b580": "#0072B2",
    "b70": "#56B4E9",
    "4070ti": "#009E73",
    "7900xtx": "#E69F00",
    "orin": "#CC79A7",
}
# With 7900xtx (#E69F00) added, --pairs all passes; worst CVD dE 7.6 (pink/green).
GPU_MARKER = {"780m": "o", "b580": "s", "b70": "D", "4070ti": "^", "7900xtx": "P",
              "orin": "v"}

MODELS = ["1b", "3b", "8b"]
MODEL_SHORT = {"1b": "1B", "3b": "3B", "8b": "8B"}
SCHEMES = ["4w", "8da4w"]
SCHEME_LONG = {"4w": "4-bit weights", "8da4w": "8-bit activations + 4-bit weights"}
SCHEME_SHORT = {"4w": "4-bit weights", "8da4w": "int8 act."}


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "DejaVu Sans", "Arial"],
        "font.size": FS_TICK,
        "axes.labelsize": FS_TICK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.edgecolor": FAINT,
        "axes.linewidth": 1.5,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 1.2,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "axes.unicode_minus": True,
    })


def load_cells():
    return pd.read_csv(CELLS_CSV)


def cell(df, gpu, model, scheme):
    r = df[(df.gpu == gpu) & (df.model == model) & (df.scheme == scheme)]
    assert len(r) == 1, (gpu, model, scheme)
    return r.iloc[0]


def new_fig():
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(W, H))


def footnote(fig, extra=""):
    """Bottom-left footnote; the pre-release dagger note sits on the line above."""
    fig.text(0.012, 0.018, FOOTNOTE + extra, fontsize=FS_FOOT, color=MUTED,
             ha="left", va="bottom")
    if PRERELEASE:
        fig.text(0.012, 0.052, DAGGER_NOTE, fontsize=FS_FOOT, color=MUTED,
                 ha="left", va="bottom")


def is_pre(g):
    return g in PRERELEASE


def save(fig, stem):
    stem = stem + SUFFIX
    for ext in ("svg", "pdf"):
        fig.savefig(HERE / f"{stem}.{ext}")
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    print(f"wrote {stem}.svg / .pdf / .png")


# ---- number formatting -------------------------------------------------------

def fmt_x(v):
    return f"{v:.1f}×"


def fmt_sig2(v):
    """Two significant figures, no trailing exponent: 57.7 -> 58, 0.456 -> 0.46."""
    if v == 0:
        return "0"
    digits = 2 - 1 - floor(log10(abs(v)))
    r = round(v, digits)
    return f"{r:.{max(digits, 0)}f}"


def fmt_s(v):
    return f"{fmt_sig2(v)} s"


def fmt_toks(v):
    """tok/s label: 3 significant figures below 1000, thousands separator above."""
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


# ---- row layout with device-class groups ------------------------------------

def grouped_rows(gpus=GPUS, row=1.0, gap=0.55):
    """y position per GPU (top to bottom) with an extra gap between device classes.

    Returns (ypos dict, list of (class, y_top_row)) with y increasing downward.
    """
    ypos, groups = {}, []
    y, prev = 0.0, None
    for g in gpus:
        c = GPU_CLASS[g]
        if prev is not None:
            y += row + (gap if c != prev else 0.0)
        if c != prev:
            groups.append((c, y))
        ypos[g] = y
        prev = c
    return ypos, groups


def draw_group_labels(ax, groups, x_axes=-0.02, dy=-0.52):
    """Subtle device-class header above the first row of each group, right-aligned
    with the GPU tick labels (y axis must be inverted, data units)."""
    for name, y in groups:
        ax.text(x_axes, y + dy, name.upper(), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FS_GROUP, color=MUTED,
                fontweight="semibold")


def repel(values, min_gap):
    """Spread 1-D label positions so neighbours are at least min_gap apart,
    keeping the group centred on the original values. Returns list in input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            d = pos[k] - pos[k - 1]
            if d < min_gap - 1e-9:
                push = (min_gap - d) / 2
                pos[k - 1] -= push
                pos[k] += push
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out
