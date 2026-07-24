"""Component C: can the microbenchmark rank tile candidates for e2e prefill?

specs/027 found qualitatively that microbench rank disagrees with e2e rank on
M5. This quantifies the question on 780M using the specs/035 round-1 data:
for every gate-passing token, run the bench in normal (perf) mode, join its
per-case GFLOP/s against the cached e2e prefill medians, and report Spearman
rho plus top-5-in-top-10 recall. Verdict thresholds:

  rho >= 0.8 and recall >= 0.8  -> usable as a prefilter on new devices
  0.5 <= rho < 0.8              -> advisory only
  below                         -> microbench is gate-only, never ranking

Bench runs are cached per token (jsonl), so the pass is resumable.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent
REPO = SPEC.parent.parent
BENCH = REPO / "cmake-out-vk/backends/vulkan/test/custom_ops/test_coopmat_linear_bench"

CFG = {
    "q4gsw": {
        "prefix": "4w",
        "env_var": "ET_VK_Q4GSW_COOPMAT_VARIANT",
        "op_prefix": "linear_q4gsw_",
    },
    "dq8ca": {
        "prefix": "8da4w",
        "env_var": "ET_VK_DQ8CA_COOPMAT_VARIANT",
        "op_prefix": "linear_dq8ca_",
    },
}

ROW_RE = re.compile(
    r"(linear_\S+_M(\d+)_K\d+_N\d+_\w+)\s+\[[0-9x]+\]\s+([0-9.]+) μs\s+([0-9.]+) GFLOP/s"
)


def gate_pass_tokens(replay_dir, prefix):
    toks = []
    for line in (Path(replay_dir) / f"{prefix}_gate.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "PASS":
            toks.append(parts[0])
    return toks


def e2e_medians(replay_dir, prefix):
    vals = {}
    for line in (Path(replay_dir) / f"{prefix}_e2e.tsv").read_text().splitlines():
        tok, v = line.split("\t")[:2]
        if v and tok != "CONTROL":
            vals.setdefault(tok, []).append(float(v))
    return {t: statistics.median(v) for t, v in vals.items()}


def bench_gflops(shader, token):
    cfg = CFG[shader]
    env = {**os.environ, cfg["env_var"]: token}
    proc = subprocess.run(
        [str(BENCH)], env=env, capture_output=True, text=True, timeout=1800
    )
    out = proc.stdout + proc.stderr
    # Two row formats. q4gsw is a single-shader dispatch: the case row itself
    # carries the (truncated) tsweep kernel name plus case name + GFLOP/s.
    # dq8ca is multi-shader (activation quantize + linear): the tsweep kernel
    # appears on a timing sub-line immediately BEFORE its case row, which
    # carries the case name + GFLOP/s. The kernel-name column truncates long
    # names ("..") so exact token matching is impossible; with the env var
    # forcing this variant, the only coopmat tsweep kernel dispatched for
    # this op IS the token, so "coopmat_tsweep" identifies its rows.
    cases = {}
    pending = False
    for line in out.splitlines():
        m = ROW_RE.search(line)
        if m and cfg["op_prefix"] in m.group(1):
            if int(m.group(2)) >= 1024 and (pending or "coopmat_tsweep" in line):
                cases[m.group(1)] = float(m.group(4))
            pending = False
        elif "coopmat_tsweep" in line:
            pending = True
    return cases


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for kk in order[i : j + 1]:
                r[kk] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", required=True, choices=tuple(CFG))
    ap.add_argument("--replay-dir", default=str(SPEC / "results/replay-780m"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg = CFG[args.shader]
    cache_path = Path(
        args.cache or SPEC / "results" / f"microbench_780m_{args.shader}.jsonl"
    )
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cached[rec["token"]] = rec["cases"]

    tokens = gate_pass_tokens(args.replay_dir, cfg["prefix"])
    e2e = e2e_medians(args.replay_dir, cfg["prefix"])
    for i, tok in enumerate(tokens):
        if tok in cached:
            continue
        cases = bench_gflops(args.shader, tok)
        cached[tok] = cases
        with open(cache_path, "a") as f:
            f.write(json.dumps({"token": tok, "cases": cases}) + "\n")
        print(f"[{i+1}/{len(tokens)}] {tok}: {len(cases)} perf cases")

    joint = [t for t in tokens if t in e2e and cached.get(t)]
    if len(joint) < 5:
        raise SystemExit(
            f"only {len(joint)} tokens have both microbench and "
            "e2e data -- parsing problem or stale cache?"
        )
    case_names = sorted({c for t in joint for c in cached[t]})
    e2e_vals = [e2e[t] for t in joint]

    lines = [
        f"# Microbench vs e2e rank correlation -- {args.shader} (780M, specs/035 round-1 data)",
        "",
        f"Tokens joined: {len(joint)}",
        "",
        "| ranking signal | spearman rho | e2e-top5 in signal-top10 |",
        "|---|---|---|",
    ]
    results = []
    for name in case_names + ["mean_gflops"]:
        if name == "mean_gflops":
            sig = [statistics.mean(cached[t].values()) for t in joint]
        else:
            if not all(name in cached[t] for t in joint):
                continue
            sig = [cached[t][name] for t in joint]
        rho = spearman(sig, e2e_vals)
        top5 = sorted(joint, key=lambda t: -e2e[t])[:5]
        sig_by = dict(zip(joint, sig))
        top10 = sorted(joint, key=lambda t: -sig_by[t])[:10]
        recall = sum(t in top10 for t in top5) / len(top5)
        results.append((name, rho, recall))
        lines.append(f"| {name} | {rho:.3f} | {recall:.2f} |")

    best = max(results, key=lambda r: r[1]) if results else ("none", 0.0, 0.0)
    if best[1] >= 0.8 and best[2] >= 0.8:
        verdict = (
            "**Usable as prefilter**: rank new-device candidates by "
            f"`{best[0]}` before spending e2e time."
        )
    elif best[1] >= 0.5:
        verdict = (
            "**Advisory only**: microbench rank is weakly informative; "
            "never substitute it for e2e ranking."
        )
    else:
        verdict = (
            "**Gate-only**: microbench correctness gating stays "
            "mandatory, but its performance rank must not drive "
            "candidate selection."
        )
    lines += [
        "",
        f"Best signal: `{best[0]}` (rho={best[1]:.3f}, recall={best[2]:.2f})",
        "",
        f"Verdict: {verdict}",
        "",
    ]

    report_path = Path(
        args.report or SPEC / "results" / f"rank-correlation-780m-{args.shader}.md"
    )
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
