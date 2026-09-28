"""Shared data access and style for the WMMA refinement figures.

Reads ../refinement.csv only (read-only). All plotted kernel and e2e numbers come from
that CSV; the per-step numbers in r3 are quoted from the per-GPU lessons docs (see r3_steps.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "refinement.csv"

GPUS = ["780M", "B580", "B70", "4070TiS", "Orin"]
GPU_NAME = {
    "780M": "Radeon 780M",
    "B580": "Arc B580",
    "B70": "Arc Pro B70",
    "4070TiS": "RTX 4070 Ti SUPER",
    "Orin": "Jetson Orin Nano",
}
GPU_SHORT = {
    "780M": "Radeon 780M",
    "B580": "Arc B580",
    "B70": "Arc Pro B70",
    "4070TiS": "RTX 4070 Ti S",
    "Orin": "Jetson Orin",
}
# Colour-blind-safe categorical set, validated with the dataviz validator
# (--pairs all, light surface): worst CVD dE 9.1, worst normal-vision dE 16.3.
GPU_COLOR = {
    "780M": "#D55E00",
    "B580": "#2A78D6",
    "B70": "#4A3AA7",
    "4070TiS": "#1BAF7A",
    "Orin": "#EDA100",
}
# GPUs whose previous WMMA kernel was opt-in (env flag / explicit enabling); default ran tiled.
OPT_IN = {"B580", "B70", "4070TiS", "Orin"}
SCHEMES = ["4w", "8da4w"]
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
MODEL_SHORT = {"llama-3.2-1b": "1B", "llama-3.2-3b": "3B", "llama-3.1-8b": "8B"}
MODEL_MARKER = {"llama-3.2-1b": "o", "llama-3.2-3b": "s", "llama-3.1-8b": "^"}

INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BEFORE_GREY = "#c9c8c3"


def setup_style(base: float = 9.0) -> None:
    fam = "Liberation Sans"
    try:
        font_manager.findfont(fam, fallback_to_default=False)
    except Exception:  # noqa: BLE001
        fam = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": fam,
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "hatch.linewidth": 0.8,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)


def load() -> pd.DataFrame:
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def _f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def kernel_geomean(df, gpu, scheme, storage="texture3d"):
    """12-shape equal-weight geomean + per-shape min/max from the derived row, cross-checked
    against the per-shape rows."""
    row = df[
        (df.gpu == gpu) & (df.scheme == scheme) & (df.storage == storage)
        & (df.data_kind == "derived") & df["shape"].str.match(r"geomean over \d+ shapes \(equal weight per shape\)$")
    ]
    assert len(row) == 1, (gpu, scheme, row)
    row = row.iloc[0]
    g = _f(row.speedup_after_over_before)
    mm = re.search(r"min ([\d.]+) max ([\d.]+)", row.notes)
    lo, hi = float(mm.group(1)), float(mm.group(2))
    # cross-check with per-shape rows
    ps = per_shape(df, gpu, scheme, storage)
    gchk = float(np.exp(np.mean(np.log(ps))))
    assert abs(gchk - g) < 0.002, (gpu, scheme, g, gchk)
    assert abs(min(ps) - lo) < 0.002 and abs(max(ps) - hi) < 0.002, (gpu, scheme, lo, hi, ps)
    return g, lo, hi, len(ps)


def per_shape(df, gpu, scheme, storage="texture3d"):
    r = df[
        (df.gpu == gpu) & (df.scheme == scheme) & (df.storage == storage)
        & (df.data_kind == "microbench kernel")
    ]
    out = []
    for _, x in r.iterrows():
        if x.before_value == "" or x.speedup_after_over_before == "":
            continue  # Orin 8B 4w w2: invalid original, excluded
        b, a = _f(x.before_value), _f(x.after_value)
        out.append(b / a)
    return out


def per_model(df, gpu, scheme, storage="texture3d"):
    """Per-layer-call-count-weighted speedup per model (the e2e weighting)."""
    r = df[
        (df.gpu == gpu) & (df.scheme == scheme) & (df.storage == storage)
        & (df.data_kind == "derived") & df["shape"].str.contains("per-layer call counts")
    ]
    return {x.model: _f(x.speedup_after_over_before) for _, x in r.iterrows()}


def projections(df, gpu, scheme):
    r = df[(df.gpu == gpu) & (df.scheme == scheme) & (df.data_kind == "derived (e2e projection)")]
    return {x.model: _f(x.speedup_after_over_before) for _, x in r.iterrows()}


def measured_e2e(df, gpu, scheme, tokens=2048):
    r = df[
        (df.gpu == gpu) & (df.scheme == scheme) & (df.data_kind == "e2e")
        & df["shape"].str.contains(f"{tokens}-token")
    ]
    return {x.model: _f(x.speedup_after_over_before) for _, x in r.iterrows()}


# ---- measured end-to-end before/after (refine/cells.csv) ----
CELLS = HERE.parents[2] / "refine" / "cells.csv"
CELL_GPU = {"780m": "780M", "b580": "B580", "b70": "B70", "4070ti": "4070TiS", "orin": "Orin"}
CELL_MODEL = {"1b": "llama-3.2-1b", "3b": "llama-3.2-3b", "8b": "llama-3.1-8b"}


def load_cells() -> pd.DataFrame:
    """One row per (gpu, model, scheme): stock_* = previous best WMMA kernels, sarc_* = re-tuned.
    speedup = sarc_median / stock_median (tok/s); speedup_ci_lo/hi = paired bootstrap 95 % CI."""
    c = pd.read_csv(CELLS)
    c["G"] = c.gpu.map(CELL_GPU)
    c["M"] = c.model.map(CELL_MODEL)
    assert c.G.notna().all() and c.M.notna().all()
    assert len(c) == 30
    return c
