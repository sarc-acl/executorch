#!/usr/bin/env python3
"""
Parse microbench logs from microbench_runner.py and produce the LLaMA-shape
comparison table + target ratios.

The output format from execute_test_cases is per-test "performance row":

    "kernel_name": "<shader>", "operat..     <global_wg>     <local_wg>   <test_name>   [<shape>]   <avg_us> μs   <gflops> GFLOP/s   PASSED

We extract: (test_name, kernel_name, avg_us) — match test_name against LLaMA
tags. Each LLaMA shape may produce one or two rows (variant×shape).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_LOGDIR = Path(
    "/home/doremy/sarc-acl/executorch/main/executorch/yanwen/artifacts/int8_microbench"
)

LLAMA_TAGS = ["llama_ffn_gateup", "llama_ffn_down", "llama_qo", "llama_kv"]

SHAPES = {
    "llama_ffn_gateup": (128, 4096, 14336),
    "llama_ffn_down": (128, 14336, 4096),
    "llama_qo": (128, 4096, 4096),
    "llama_kv": (128, 4096, 1024),
}

# Kernel name capture: contiguous non-whitespace, non-quote, non-dot (kernel names
# may be truncated mid-string with "..", so the closing quote is unreliable).
_KN = r"\S+?"  # lazy match of non-whitespace; kernel name has no spaces

# Match performance row: ... <test_name with llama> [<shape>] <us> μs <gflops> GFLOP/s
ROW_RE = re.compile(
    r'"kernel_name":\s*"('
    + _KN
    + r')(?:\.\.)?(?:"|\b).*?\s+([\w_]*llama[\w_]*)\s+\[(\d+x\d+)\]\s+([\d.]+)\s*μs\s+([\d.]+)\s*GFLOP/s\s+(\w+)'
)

# Alternate format used by q8csw_linear for et_vk.* operator name (no JSON):
ROW_RE_OP = re.compile(
    r"(et_vk\.\S+)\s+(performance_\d+_\d+_\d+(?:_no_bias)?(?:_llama_[\w]+)?_(?:Texture3D|Buffer)_(?:Float|Half))\s+\[(\d+x\d+)\]\s+([\d.]+)\s*μs\s+([\d.]+)\s*GFLOP/s\s+(\w+)"
)

# For q8csw_linear's kernel rows. Kernel name may be truncated (no closing quote)
ROW_RE_KERN = re.compile(
    r'"kernel_name":\s*"('
    + _KN
    + r')(?:\.\.)?(?:"|\b).*?\s+(performance_\d+_\d+_\d+(?:_no_bias)?(?:_llama_\w+)?_(?:Texture3D|Buffer)_(?:Float|Half))\s+\[(\d+x\d+)\]\s+([\d.]+)\s*μs\s+([\d.]+)\s*GFLOP/s\s+(\w+)'
)


def shape_from_name_or_label(test_name, fallback_label):
    """For q8csw_linear, the test name encodes M_K_N. Map to llama tag."""
    # performance_128_4096_14336 → match against SHAPES
    m = re.search(r"performance_(\d+)_(\d+)_(\d+)", test_name)
    if not m:
        return None
    mm, kk, nn = int(m.group(1)), int(m.group(2)), int(m.group(3))
    for tag, (M, K, N) in SHAPES.items():
        if (mm, kk, nn) == (M, K, N):
            return tag
    return None


def parse_log(path: Path, binary_name: str):
    """Returns list of dicts: {tag, variant, dtype, kernel, avg_us, raw_test_name}."""
    rows = []
    if not path.exists():
        return rows
    text = path.read_text()

    for line in text.splitlines():
        # JSON-kernel rows (linear_coopmat_bench, khr_cm_gemm_int8, q8csw_linear kernel rows)
        m = ROW_RE.search(line)
        if m:
            kernel, test_name, shape_str, us, gflops, passing = m.groups()
            tag = None
            for t in LLAMA_TAGS:
                if t in test_name:
                    tag = t
                    break
            if tag is None:
                continue
            rows.append(
                {
                    "binary": binary_name,
                    "tag": tag,
                    "test_name": test_name,
                    "kernel": kernel,
                    "avg_us": float(us),
                    "gflops": float(gflops),
                    "passing": passing,
                }
            )
            continue

        # q8csw_linear kernel rows (matches performance_M_K_N_*)
        m = ROW_RE_KERN.search(line)
        if m:
            kernel, test_name, shape_str, us, gflops, passing = m.groups()
            tag = shape_from_name_or_label(test_name, "")
            if tag is None:
                continue
            rows.append(
                {
                    "binary": binary_name,
                    "tag": tag,
                    "test_name": test_name,
                    "kernel": kernel,
                    "avg_us": float(us),
                    "gflops": float(gflops),
                    "passing": passing,
                }
            )
            continue

        # q8csw_linear op-name rows (et_vk.linear_q8ta_q8csw.default etc.)
        m = ROW_RE_OP.search(line)
        if m:
            opname, test_name, shape_str, us, gflops, passing = m.groups()
            tag = shape_from_name_or_label(test_name, "")
            if tag is None:
                continue
            rows.append(
                {
                    "binary": binary_name,
                    "tag": tag,
                    "test_name": test_name,
                    "kernel": opname,  # use op name as proxy for kernel here
                    "avg_us": float(us),
                    "gflops": float(gflops),
                    "passing": passing,
                }
            )
    return rows


def classify_row(row):
    """Classify a row into (variant, dtype) based on test_name and kernel."""
    name = row["test_name"]
    kernel = row["kernel"]
    binary = row["binary"]

    if binary == "khr_cm_gemm_int8":
        # khr_cm_gemm_int8.cpp emits both orig (impl=3, broken on wave64) and
        # wave64 (impl=4, correct) variants. Test names look like
        # "khr_cm_int8_orig_llama_*" vs "khr_cm_int8_wave64_llama_*". Keep
        # them in distinct buckets so the min() aggregation can't silently
        # pick the artificially-fast orig timings.
        if "_orig_" in name or kernel == "matmul_khr_cm_int8":
            return ("coopmat", "int8_orig")
        if "_wave64_" in name or kernel == "matmul_khr_cm_int8_wave64":
            return ("coopmat", "int8_wave64")
        # Older logs (pre-c331b9cf5) had no orig/wave64 prefix; those tested
        # the broken shader only.
        return ("coopmat", "int8_orig")

    if binary == "linear_coopmat_bench":
        if name.startswith("vec_tex_"):
            return ("noncoopmat", "fp32")
        if name.startswith("cm_fp32_"):
            return ("coopmat", "fp32")
        if name.startswith("vec_fp16_"):
            return ("noncoopmat", "fp16")
        if name.startswith("cm_fp16_"):
            return ("coopmat", "fp16")

    if binary == "q8csw_linear":
        if "q8ta" in kernel.lower():
            return ("q8ta", "int8")
        if "q8csw" in kernel.lower():
            return ("noncoopmat", "int8")
    return ("other", "?")


def fmt_us(x):
    if x is None:
        return "—"
    if x < 1000:
        return f"{x:.0f}"
    return f"{x/1000:.2f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    args = ap.parse_args()

    all_rows = []
    for binary in ["khr_cm_gemm_int8", "linear_coopmat_bench", "q8csw_linear"]:
        candidates = sorted(args.logdir.glob(f"{binary}_*.log"))
        if not candidates:
            print(f"[summary] WARNING no log for {binary} in {args.logdir}")
            continue
        latest = candidates[-1]
        rows = parse_log(latest, binary)
        print(f"[summary] {binary}: {len(rows)} LLaMA rows from {latest.name}")
        for r in rows:
            r["variant"], r["dtype"] = classify_row(r)
        all_rows.extend(rows)

    # Index by (tag, variant, dtype) — take MIN avg_us when multiple (e.g. texture3d vs buffer storage)
    by_cell = defaultdict(list)
    for r in all_rows:
        by_cell[(r["tag"], r["variant"], r["dtype"])].append(r["avg_us"])

    def get(tag, variant, dtype):
        vals = by_cell.get((tag, variant, dtype), [])
        return min(vals) if vals else None

    print()
    print("=== LLaMA-shape microbench comparison (best avg μs across runs/storage) ===")
    header = (
        f"{'component':<20}  {'shape':<22}  "
        f"{'fp32_vec':>10}  {'fp32_cm':>10}  "
        f"{'fp16_vec':>10}  {'fp16_cm':>10}  "
        f"{'int8_q8csw':>11}  {'int8_q8ta':>10}  "
        f"{'int8_cm_orig':>13}  {'int8_cm_w64':>12}"
    )
    print(header)
    print("-" * len(header))

    for tag in LLAMA_TAGS:
        M, K, N = SHAPES[tag]
        shape_str = f"M={M},K={K},N={N}"
        vals = [
            get(tag, "noncoopmat", "fp32"),
            get(tag, "coopmat", "fp32"),
            get(tag, "noncoopmat", "fp16"),
            get(tag, "coopmat", "fp16"),
            get(tag, "noncoopmat", "int8"),
            get(tag, "q8ta", "int8"),
            get(tag, "coopmat", "int8_orig"),
            get(tag, "coopmat", "int8_wave64"),
        ]
        cells = [fmt_us(v) for v in vals]
        print(
            f"{tag:<20}  {shape_str:<22}  "
            f"{cells[0]:>10}  {cells[1]:>10}  "
            f"{cells[2]:>10}  {cells[3]:>10}  "
            f"{cells[4]:>11}  {cells[5]:>10}  "
            f"{cells[6]:>13}  {cells[7]:>12}"
        )

    # Target ratios
    print()
    print("=== Target ratios (per-shape and averaged) ===")
    print()
    print(
        f"{'ratio (smaller=numerator faster)':<35}  {'expected':<10}  {'per-shape values':<35}  {'mean':>8}  {'verdict':<10}"
    )
    print("-" * 105)

    def ratio_block(label, expected_lo, expected_hi, num_v_d, den_v_d, hypothesis_note):
        per_shape = []
        for tag in LLAMA_TAGS:
            n = get(tag, *num_v_d)
            d = get(tag, *den_v_d)
            if n is None or d is None or d == 0:
                per_shape.append((tag, None))
            else:
                per_shape.append((tag, n / d))
        vals = [v for _, v in per_shape if v is not None]
        if not vals:
            print(f"{label:<35}  {hypothesis_note:<10}  {'N/A':<35}  {'-':>8}  no data")
            return
        mean = sum(vals) / len(vals)
        verdict = "OK" if expected_lo <= mean <= expected_hi else "off"
        per_s_str = (
            "["
            + ", ".join(f"{v:.2f}" if v is not None else "—" for _, v in per_shape)
            + "]"
        )
        print(
            f"{label:<35}  {hypothesis_note:<10}  {per_s_str:<35}  {mean:>8.2f}  {verdict:<10}"
        )

    # All headline int8-coopmat ratios use the wave64-correct shader.
    # The orig (impl=3) shader is broken on AMD RDNA3+ (wave64) — see
    # khr_cm_gemm_int8_validate. We expose its timing only for R4c (bug
    # quantification).

    # R1: int8 coopmat / fp16 coopmat — user's "int8 ~2× faster than fp16" hypothesis
    ratio_block(
        "R1 int8_cm_w64/fp16_cm (hyp:0.5x)",
        0.40,
        0.65,
        ("coopmat", "int8_wave64"),
        ("coopmat", "fp16"),
        "~0.50",
    )
    # R1b: vs fp32 coopmat — proxy since fp16 LLaMA microbench crashed.
    ratio_block(
        "R1b int8_cm_w64/fp32_cm",
        0.20,
        0.50,
        ("coopmat", "int8_wave64"),
        ("coopmat", "fp32"),
        "~0.25",
    )
    # R2: fp16 coopmat / fp16 vec — should match E2E 3.08× (i.e., 0.32 ratio)
    ratio_block(
        "R2 fp16_cm/fp16_vec  (hyp: 0.32x)",
        0.25,
        0.45,
        ("coopmat", "fp16"),
        ("noncoopmat", "fp16"),
        "~0.32",
    )
    # R2b: fp32 coopmat / fp32 vec — direct measure of coopmat lift
    ratio_block(
        "R2b fp32_cm/fp32_vec",
        0.20,
        0.45,
        ("coopmat", "fp32"),
        ("noncoopmat", "fp32"),
        "~0.32",
    )
    # R3: int8 q8csw / fp32 vec — what int8 weight-only (W8A16) buys without coopmat
    ratio_block(
        "R3 int8_q8csw/fp32_vec",
        0.0,
        2.0,
        ("noncoopmat", "int8"),
        ("noncoopmat", "fp32"),
        "?",
    )
    # R4a: int8 KHR coopmat (wave64-correct) / W8A16 linear_q8csw_tiled
    # The primary "coopmat lift on int8" answer.
    ratio_block(
        "R4a int8_cm_w64/int8_q8csw (h:0.25)",
        0.10,
        0.40,
        ("coopmat", "int8_wave64"),
        ("noncoopmat", "int8"),
        "~0.25",
    )
    # R4b: int8 KHR coopmat (wave64) / W8A8 linear_q8ta_q8csw
    # q8ta already uses int8-dot-product hardware so this is the harder
    # comparison; we expect the coopmat win to be smaller here.
    ratio_block(
        "R4b int8_cm_w64/int8_q8ta",
        0.0,
        2.0,
        ("coopmat", "int8_wave64"),
        ("q8ta", "int8"),
        "?",
    )
    # R4c: orig (broken) / wave64 (correct) — quantifies the wave32/wave64
    # inflation factor. Should be ~0.65–0.75 at FFN shapes (orig is faster
    # because it only writes half the output tile).
    ratio_block(
        "R4c int8_cm_orig/int8_cm_w64",
        0.50,
        1.0,
        ("coopmat", "int8_orig"),
        ("coopmat", "int8_wave64"),
        "bug-quant",
    )


if __name__ == "__main__":
    main()
