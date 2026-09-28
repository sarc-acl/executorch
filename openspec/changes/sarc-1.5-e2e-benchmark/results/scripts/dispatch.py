"""Dispatch evidence from the untimed ETDump runs: raw/<gpu>/trace/<model>-<scheme>-<build>.etdp.

For each (gpu, model, scheme, build) lists the distinct linear and SDPA kernel names that ran
in the real model (last execution in the file), with dispatch counts.
Run with the dev/1.5 venv (executorch.devtools):
  ~/Desktop/sarc-acl/dev/1.5/executorch/.venv/bin/python report/dispatch.py
"""

import collections
import json
import warnings
from pathlib import Path

from executorch.devtools import Inspector

warnings.filterwarnings("ignore")
C = Path(__file__).resolve().parent.parent
GPUS = ["780m", "b580", "b70", "4070ti", "orin"]


def family(k):
    if "sdpa" in k:
        return "sdpa"
    if "linear" in k or "q4gsw" in k or "dq8ca" in k:
        return "linear"
    return None


def kernels(path):
    df = Inspector(etdump_path=str(path)).to_dataframe()
    cnt = collections.Counter()
    for _, r in df.iterrows():
        n = str(r.event_name)
        if not n.startswith("{"):
            continue
        try:
            k = json.loads(n).get("kernel_name", "")
        except Exception:
            continue
        if family(k):
            cnt[k] += 1
    return cnt


def main():
    lines = ["# Dispatched linear and SDPA kernels (ETDump, real model, 2048-token prefill)\n"]
    rows = []
    for g in GPUS:
        d = C / "raw" / g / "trace"
        if not d.exists():
            continue
        lines.append(f"\n## {g}\n")
        for p in sorted(d.glob("*.etdp")):
            m, q, b = p.stem.split("-")
            try:
                cnt = kernels(p)
            except Exception as e:  # keep going; report the failure
                lines.append(f"- {m} {q} {b}: ETDump parse failed: {e}")
                continue
            sarc = sum(v for k, v in cnt.items() if k.startswith("sarc_"))
            lines.append(f"- **{m} {q} {b}** ({sarc}/{sum(cnt.values())} dispatches SARC):")
            for k, v in sorted(cnt.items()):
                lines.append(f"  - `{k}` × {v}")
                rows.append({"gpu": g, "model": m, "scheme": q, "build": b, "kernel": k, "count": v})
    (C / "report" / "dispatch.md").write_text("\n".join(lines) + "\n")
    import csv

    with open(C / "report" / "dispatch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gpu", "model", "scheme", "build", "kernel", "count"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
