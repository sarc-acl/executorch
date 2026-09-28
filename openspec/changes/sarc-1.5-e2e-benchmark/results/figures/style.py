"""Shared order, labels, colours and matplotlib style for the e2e prefill figures."""
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
CELLS_CSV = REPORT / "cells.csv"
RUNS_CSV = REPORT / "runs_all.csv"

# GPU order: integrated / low power first, then discrete by vendor.
GPUS = ["780m", "b580", "b70", "4070ti", "orin"]
GPU_LABEL = {
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
}
# Two-line tick labels for narrow facets.
GPU_TICK = {
    "780m": "Radeon\n780M",
    "b580": "Arc\nB580",
    "b70": "Arc Pro\nB70",
    "4070ti": "RTX 4070\nTi SUPER",
    "orin": "Jetson\nOrin Nano",
}

# Compact single-line labels for the heatmap rows.
GPU_SHORT = {
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
}

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


def load_cells():
    return pd.read_csv(CELLS_CSV)


def load_timed_runs():
    runs = pd.read_csv(RUNS_CSV)
    return runs[runs["log"].str.startswith("logs/prefill")].copy()


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}")
    print(f"wrote {stem}.pdf / {stem}.png")
