"""Shared data loading, labels, colours and style for the evidence figures (e1-e4).

All inputs are read-only files under report/ (see CAPTIONS.md)."""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
EVID = HERE.parent
REPORT = EVID.parent
ROOFLINE_JSON = EVID / "roofline.json"
EFFICIENCY_CSV = EVID / "efficiency.csv"
FAMILIES_CSV = EVID / "trace" / "families.csv"
CELLS_CSV = REPORT / "cells.csv"

GPUS = ["780m", "b580", "b70", "4070ti", "orin"]
GPU_LABEL = {
    "780m": "Radeon 780M",
    "b580": "Arc B580",
    "b70": "Arc Pro B70",
    "4070ti": "RTX 4070 Ti SUPER",
    "orin": "Jetson Orin Nano",
}
SCHEMES = ["4w", "8da4w"]
MODELS = ["1b", "3b", "8b"]
MODEL_LABEL = {"1b": "1B", "3b": "3B", "8b": "8B"}

STOCK = "#8F8F8F"
STOCK_EDGE = "#4D4D4D"
SARC = "#E8590C"
SARC_EDGE = "#8A3000"
# Scheme colours for e3 (Okabe-Ito blue / bluish green; not orange, which means SARC).
SCHEME_COLOR = {"4w": "#0072B2", "8da4w": "#009E73"}

INK = "#222222"
MUTED = "#5E5E5E"
GRID = "#E6E6E6"
WIDTH = 7.0

# Roof matched with each (scheme, build); SARC 4w on the 780M uses the fp32-accumulate kernel.
STOCK_ROOF = {"4w": "alu_fp16", "8da4w": "dot_int8"}


def sarc_roof(gpu, scheme):
    if scheme == "8da4w":
        return "matrix_int8"
    return "matrix_fp16_fp32" if gpu == "780m" else "matrix_fp16"


ROOF_LABEL = {
    "alu_fp16": "fp16 FMA",
    "dot_int8": "int8 dot",
    "matrix_fp16": "fp16 matrix (fp16 acc)",
    "matrix_fp16_fp32": "fp16 matrix (fp32 acc)",
    "matrix_int8": "int8 matrix",
}

ATTN = ["attention: QK^T", "attention: softmax", "attention: AV", "attention: KV update"]
GEMM = "prefill GEMM (linear layers)"
QUANT = "8-bit activation quantize"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def roofs():
    d = json.loads(ROOFLINE_JSON.read_text())
    return {g: {k: v["value"] for k, v in d["gpus"][g]["roofs"].items()} for g in GPUS}


def efficiency():
    return pd.read_csv(EFFICIENCY_CSV)


def eff_row(eff, gpu, scheme, build, model="8b"):
    r = eff[(eff.gpu == gpu) & (eff.scheme == scheme) & (eff.build == build) & (eff.model == model)]
    assert len(r) == 1, (gpu, scheme, build, model)
    return r.iloc[0]


def families():
    return pd.read_csv(FAMILIES_CSV)


def family_ms(fam, gpu, model, scheme, build):
    """Return {family: ms} for one trace cell."""
    f = fam[(fam.gpu == gpu) & (fam.model == model) & (fam.scheme == scheme) & (fam.build == build)]
    assert len(f) > 0, (gpu, model, scheme, build)
    return dict(zip(f.family, f.ms))


def cells():
    return pd.read_csv(CELLS_CSV)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=200)
    print(f"wrote {stem}.pdf / {stem}.png", file=sys.stderr)
