"""Kernel efficiency vs confirmed roofs: warm ETDump GEMM time (trace/gemm.csv) + roofline.json.
Writes efficiency.csv: gpu, scheme, build, model, rate (TFLOP/s or TOP/s), roof name, roof, pct."""
import csv, json, collections
from pathlib import Path
E = Path(__file__).resolve().parent
R = json.load(open(E / "roofline.json"))["gpus"]
def roof(g, name): return R[g]["roofs"][name]["value"]
# which roof each kernel is compared with
ACC4 = {"780m": "matrix_fp16_fp32", "b580": "matrix_fp16", "b70": "matrix_fp16", "4070ti": "matrix_fp16", "orin": "matrix_fp16"}
REF = {("4w", "stock"): lambda g: "alu_fp16", ("8da4w", "stock"): lambda g: "dot_int8",
       ("4w", "sarc"): lambda g: ACC4[g], ("8da4w", "sarc"): lambda g: "matrix_int8"}
A = collections.defaultdict(lambda: [0.0, 0.0])
for r in csv.DictReader(open(E / "trace" / "gemm.csv")):
    k = (r["gpu"], r["scheme"], r["build"], r["model"])
    A[k][0] += 2 * int(r["M"]) * int(r["N"]) * int(r["K"]); A[k][1] += float(r["ms"])
rows = []
for (g, q, b, m), (fl, ms) in sorted(A.items()):
    rate = fl / (ms * 1e-3) / 1e12; rn = REF[(q, b)](g); rv = roof(g, rn)
    rows.append({"gpu": g, "scheme": q, "build": b, "model": m, "rate": round(rate, 3), "roof": rn,
                 "roof_value": rv, "pct_of_roof": round(100 * rate / rv, 1)})
with open(E / "efficiency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["model"] == "8b": print(r)
