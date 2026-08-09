#!/usr/bin/env python3
"""Rank dbuf4 tile tokens from microbench_sweep.py jsonl, per driver and scheme,
and A/B the two drivers on the tokens they share.

Ranking signal is geomean kernel_gflops across the 12 prefill shapes. Tokens
that failed the correctness gate are excluded from the ranking and counted
separately -- they are the headline number for dq8ca, not a footnote.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 0.0


def load(driver, scheme, tag="", group_size=128):
    tag = f"_{tag}" if tag else ""
    p = RESULTS / f"microbench_{driver}_{scheme}_dbuf4_g{group_size}{tag}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r  # later record wins (resume/re-run)
    return out


def summarize(recs):
    ranked, failed = [], defaultdict(list)
    for tok, r in recs.items():
        if r["status"] == "ok":
            ranked.append(
                (geomean([c["kernel_gflops"] for c in r["cases"].values()]), tok, r)
            )
        else:
            # records written before the sweep script learned to classify
            # crashes land as no_results with a segfault in the captured tail
            st = r["status"]
            if st == "no_results" and any(
                "Segmentation fault" in t for t in (r.get("tail") or [])
            ):
                st = "crash"
            failed[st].append(tok)
    ranked.sort(reverse=True)
    return ranked, failed


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", nargs=2, default=["f14c51b6f8", "e0da99c1d1"])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    A, B = args.drivers

    for scheme in ("q4gsw", "dq8ca"):
        label = {"q4gsw": "4w", "dq8ca": "8da4w"}[scheme]
        print(
            f"\n{'='*78}\n{scheme}  ({label})  — prefill M=2048, geomean over 12 shapes\n{'='*78}"
        )

        per = {}
        for drv in (A, B):
            recs = load(drv, scheme)
            if not recs:
                print(f"  [{drv}] no data")
                continue
            ranked, failed = summarize(recs)
            per[drv] = (ranked, failed, recs)
            nfail = sum(len(v) for v in failed.values())
            # built outside the f-string: a dict comprehension inside one has
            # its ':' parsed as a format spec
            breakdown = {k: len(v) for k, v in failed.items()}
            print(
                f"\n-- driver {drv}: {len(ranked)} ranked, {nfail} excluded "
                f"({breakdown})"
            )
            print(f"   {'rank':>4}  {'geomean GFLOP/s':>16}  token")
            for i, (g, tok, _) in enumerate(ranked[: args.top], 1):
                print(f"   {i:>4}  {g:>16.1f}  {tok}")

        if A in per and B in per:
            ra = {t: g for g, t, _ in per[A][0]}
            rb = {t: g for g, t, _ in per[B][0]}
            shared = sorted(set(ra) & set(rb), key=lambda t: -ra[t])
            print(f"\n-- A/B on {len(shared)} tokens ranked OK under BOTH drivers")
            if shared:
                ratios = [rb[t] / ra[t] for t in shared]
                print(f"   B/A geomean ratio : {geomean(ratios):.4f}")
                print(f"   B/A min / max     : {min(ratios):.4f} / {max(ratios):.4f}")
                print(
                    f"   best token on A   : {shared[0]}  "
                    f"A={ra[shared[0]]:.1f}  B={rb[shared[0]]:.1f}"
                )
                bestb = max(shared, key=lambda t: rb[t])
                print(
                    f"   best token on B   : {bestb}  "
                    f"A={ra[bestb]:.1f}  B={rb[bestb]:.1f}"
                )
                print(f"   same winner?      : {'YES' if shared[0] == bestb else 'NO'}")
                # Spearman on shared tokens
                oa = {t: i for i, t in enumerate(sorted(shared, key=lambda t: -ra[t]))}
                ob = {t: i for i, t in enumerate(sorted(shared, key=lambda t: -rb[t]))}
                n = len(shared)
                if n > 1:
                    d2 = sum((oa[t] - ob[t]) ** 2 for t in shared)
                    print(f"   Spearman rho      : {1 - 6*d2/(n*(n*n-1)):.4f}")
            # correctness-set disagreement is itself a driver finding
            fa = {t for v in per[A][1].values() for t in v}
            fb = {t for v in per[B][1].values() for t in v}
            if fa ^ fb:
                print(
                    f"   gate differs on   : {len(fa ^ fb)} tokens "
                    f"(A-only fail {len(fa-fb)}, B-only fail {len(fb-fa)})"
                )

    # canary drift
    print(f"\n{'='*78}\ncanary (driver A re-run after the B block)\n{'='*78}")
    any_canary = False
    for scheme in ("q4gsw", "dq8ca"):
        first = load(A, scheme)
        again = load(A, scheme, tag="canary")
        common = set(first) & set(again)
        if not common:
            continue
        any_canary = True
        print(f"\n-- {scheme}: {len(common)} canary tokens")
        for tok in sorted(common):
            g1 = geomean([c["kernel_gflops"] for c in first[tok]["cases"].values()])
            g2 = geomean([c["kernel_gflops"] for c in again[tok]["cases"].values()])
            if g1 and g2:
                print(
                    f"   {tok:44s} {g1:8.1f} -> {g2:8.1f}  " f"({100*(g2-g1)/g1:+.2f}%)"
                )
    if not any_canary:
        print("  (none yet)")


if __name__ == "__main__":
    main()
