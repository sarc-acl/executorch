"""Optuna constrained-TPE tile sweep (component B).

One command per shader per device. The sampler proposes tile geometry as
per-dimension categoricals (so TPE can learn cross-dimension structure like
"k32 beats k16 here" -- a single categorical over ~600 legal tokens could
not); illegal or already-settled combinations are rejected at ask time and
cost neither budget nor GPU time. Compilation is batched: each round asks
batch_size candidates, appends their yaml variants, rebuilds once, then
gates and measures each in a fresh process.

Deterministic failures (gate fail, glslc fail, output-hash miscompute) go to
a per-device blocklist so no future run ever re-proposes them. The Optuna
journal (JSON lines) is the resume state: rerunning the same command
continues where it stopped.

Dry-run mode replays the cached specs/035 780M TSVs with no GPU and no
rebuild, to validate loop logic end-to-end.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import optuna

import tile_constraints as tc
import yaml_variants as yv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

SPEC = Path(__file__).resolve().parent.parent

# Seeds are starting points for TPE, not claims that they are good here. The
# wave64 M5 winners are dropped: this device reports min=max=subgroup 32, so
# every s64 candidate is pruned before it ever reaches the GPU.
SEEDS = {
    "q4gsw": [
        "tsweep_t128x128k16g22s32",  # this branch's shipped default
        "tsweep_t128x64k32g22s32",  # 780M winner (specs/035)
        "tsweep_t128x64k16g22s32",  # M5 winner, re-expressed at s32
    ],
    "dq8ca": [
        "tsweep_t128x64k32g42s32",  # this branch's shipped default (as mk32)
        "tsweep_t64x128k32g41s32",  # 780M winner (specs/035)
    ],
}


def default_mma_k(shader):
    """Single source of truth with the yaml the variants are appended to."""
    return yv.SHADER_INFO[shader]["extra_fields"].get("MMA_K", 16)

DIMS = ("m", "n", "k", "gx", "gy", "sub")


def params_to_token(p):
    return tc.token(p["m"], p["n"], p["k"], p["gx"], p["gy"], p["sub"])


def token_to_params(tok):
    t = tc.parse_token(tok)
    return {
        "m": t["wg_tile_m"],
        "n": t["wg_tile_n"],
        "k": t["wg_tile_k"],
        "gx": t["sg_grid_x"],
        "gy": t["sg_grid_y"],
        "sub": t["subgroup_size"],
    }


class Blocklist:
    def __init__(self, path):
        self.path = Path(path)
        self.tokens = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                self.tokens.add(json.loads(line)["token"])

    def add(self, token, reason, detail=""):
        if token in self.tokens:
            return
        self.tokens.add(token)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "token": token,
                        "reason": reason,
                        "detail": detail[:300],
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )
                + "\n"
            )


class ReplaySession:
    """Answers gate/measure from the cached specs/035 TSVs. Tokens the old
    sweep never ran count as gate failures (so the loop's blocklist/prune
    paths get exercised)."""

    TSV_PREFIX = {"q4gsw": "4w", "dq8ca": "8da4w"}

    def __init__(self, shader, replay_dir):
        pre = self.TSV_PREFIX[shader]
        d = Path(replay_dir)
        self.gates = {}
        for line in (d / f"{pre}_gate.tsv").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                self.gates[parts[0]] = parts[1]
        self.e2e = {}
        for line in (d / f"{pre}_e2e.tsv").read_text().splitlines():
            tok, val = line.split("\t")[:2]
            if val:
                self.e2e.setdefault(tok, []).append(float(val))
        ctrl = sorted(self.e2e.get("CONTROL", [0.0]))
        self.baseline_median = ctrl[len(ctrl) // 2]
        self.remeasure = []

    def baseline(self):
        return self.baseline_median, 0.0

    def gate(self, token):
        class G:
            pass

        g = G()
        g.status = "pass" if self.gates.get(token) == "PASS" else "correctness_fail"
        return g

    def measure(self, token, reps=2, stage="screen"):
        class M:
            pass

        m = M()
        vals = self.e2e.get(token, [])
        m.correctness_flag = not vals
        vals = sorted(vals)
        m.median = vals[len(vals) // 2] if vals else 0.0
        return m

    def confirm(self, token, reps=5):
        return self.measure(token, reps, "confirm")


def known_tokens_in_study(study):
    seen = set()
    for t in study.get_trials(deepcopy=False):
        if all(d in t.params for d in DIMS):
            seen.add(params_to_token(t.params))
    return seen


def ask_legal(
    study,
    shader,
    group_size,
    limits,
    blocklist,
    settled,
    batch_size,
    mma_k=None,
    max_rejects=2000,
):
    batch, rejects = [], 0
    while len(batch) < batch_size and rejects < max_rejects:
        trial = study.ask()
        p = {
            "m": trial.suggest_categorical("m", tc.WG_TILE_MN_CHOICES),
            "n": trial.suggest_categorical("n", tc.WG_TILE_MN_CHOICES),
            "k": trial.suggest_categorical("k", tc.WG_TILE_K_CHOICES),
            "gx": trial.suggest_categorical("gx", tc.SG_GRID_CHOICES),
            "gy": trial.suggest_categorical("gy", tc.SG_GRID_CHOICES),
            "sub": trial.suggest_categorical("sub", limits.subgroup_sizes),
        }
        tok = params_to_token(p)
        c = tc.derive(
            shader,
            p["m"],
            p["n"],
            p["k"],
            p["gx"],
            p["gy"],
            p["sub"],
            group_size,
            limits,
            mma_k,
        )
        if not c["valid"] or tok in blocklist.tokens or tok in settled:
            study.tell(trial, state=TrialState.PRUNED)
            rejects += 1
            continue
        trial.set_user_attr("token", tok)
        settled.add(tok)  # constant-liar-adjacent: no dupes within this run
        batch.append((trial, tok))
    return batch


def run_sweep(args):  # noqa: C901
    shader = args.shader
    if args.dry_run:
        session = ReplaySession(shader, args.replay_dir)
        limits = tc.DeviceLimits(65536, 1024, (32, 64))
        slug = "replay-780m"
        fingerprint = {"device_name": "replay", "source": str(args.replay_dir)}
        rebuild_fn = (
            None  # never called: all rebuild call sites are `if not args.dry_run`
        )
    else:
        if args.remote == "android":
            import measure_android as measure_mod
        else:
            import measure as measure_mod

        session = measure_mod.Session(
            shader, pte=args.pte, strict=args.strict, quirks=args.quirk
        )
        limits = session.limits
        slug = session.device_slug
        fingerprint = session.fingerprint
        rebuild_fn = measure_mod.rebuild if args.remote == "android" else yv.rebuild

    blocklist = Blocklist(SPEC / "results" / f"blocklist_{slug}_{shader}.jsonl")
    storage = JournalStorage(
        JournalFileBackend(
            str(SPEC / "results" / f"optuna_journal_{slug}_{shader}.log")
        )
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{shader}_{slug}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, constant_liar=True, seed=36
        ),
    )

    settled = known_tokens_in_study(study) | blocklist.tokens
    done = [
        t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE
    ]
    measured = len(done)
    best = max((t.value for t in done), default=0.0)
    since_best = 0
    pruned_reasons = Counter()

    for tok in SEEDS[shader] + [t.strip() for t in args.seed]:
        if tok not in settled:
            study.enqueue_trial(token_to_params(tok))

    mma_k = args.mma_k or default_mma_k(shader)
    print(f"[{shader}] MMA_K={mma_k}, {len(tc.enumerate_legal(shader, limits, args.group_size, mma_k))} legal candidates")

    baseline_median, noise_cov = session.baseline()
    print(
        f"[{shader}] baseline {baseline_median:.1f} tok/s, noise cov "
        f"{noise_cov*100:.2f}%, resuming with {measured} measured trials"
    )

    while measured < args.budget and since_best < args.early_stop:
        batch = ask_legal(
            study,
            shader,
            args.group_size,
            limits,
            blocklist,
            settled,
            min(args.batch_size, args.budget - measured),
            mma_k,
        )
        if not batch:
            print("legal universe exhausted (or all remaining blocklisted)")
            break

        if not args.dry_run:
            # both_storages: the correctness gate exercises Buffer weight
            # storage as well as texture2d, and a tile that only ships the
            # texture2d variant gates as missing_shader.
            added = yv.ensure_variants(
                shader, [tok for _, tok in batch], both_storages=True
            )
            if added:
                r = rebuild_fn(log_path=SPEC / "results" / "last_build.log")
                if not r.ok:
                    for bad in r.failed_tokens:
                        blocklist.add(bad, "glslc_failure", r.log_excerpt[-300:])
                    yv.remove_variants(shader, r.failed_tokens)
                    r2 = rebuild_fn(log_path=SPEC / "results" / "last_build.log")
                    if not r2.ok:
                        raise RuntimeError(
                            "rebuild failed twice; see "
                            f"{SPEC/'results'/'last_build.log'}"
                        )
                    for trial, tok in batch:
                        if tok in r.failed_tokens:
                            study.tell(trial, state=TrialState.PRUNED)
                            pruned_reasons["glslc_failure"] += 1
                    batch = [
                        (tr, tok) for tr, tok in batch if tok not in r.failed_tokens
                    ]

        for trial, tok in batch:
            # The bench gate only exercises small shapes (M<=128). A dq8ca
            # coopmat tile whose WG_TILE_M exceeds that never dispatches there,
            # so the gate reports alignment_fallback for tiles that are fine at
            # a real 2048-row prefill -- including one already validated e2e on
            # this device. Skipping it leaves e2e output-hash + token checks as
            # the correctness signal, which is the stricter test anyway.
            g = None if args.skip_gate else session.gate(tok)
            if g is not None and g.status != "pass":
                blocklist.add(tok, f"gate_{g.status}")
                study.tell(trial, state=TrialState.PRUNED)
                pruned_reasons[f"gate_{g.status}"] += 1
                print(f"  {tok}\tPRUNED gate:{g.status}")
                continue
            m = session.measure(tok, reps=args.reps)
            if m.correctness_flag:
                blocklist.add(tok, "output_miscompute")
                study.tell(trial, state=TrialState.PRUNED)
                pruned_reasons["output_miscompute"] += 1
                print(f"  {tok}\tPRUNED miscompute")
                continue
            study.tell(trial, m.median)
            measured += 1
            if m.median > best:
                best, since_best = m.median, 0
            else:
                since_best += 1
            print(
                f"  {tok}\t{m.median:.1f}\t(best {best:.1f}, "
                f"{measured}/{args.budget})"
            )

    done = [
        t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE
    ]
    done.sort(key=lambda t: t.value, reverse=True)
    finalists = done[: args.finalists]
    if not args.dry_run and finalists:
        toks = [
            t.user_attrs.get("token") or params_to_token(t.params) for t in finalists
        ]
        if yv.ensure_variants(shader, toks, both_storages=True):
            r = rebuild_fn(log_path=SPEC / "results" / "last_build.log")
            if not r.ok:
                raise RuntimeError("rebuild for finalist confirmation failed")
    confirmed = []
    for t in finalists:
        tok = t.user_attrs.get("token") or params_to_token(t.params)
        m = session.confirm(tok)
        confirmed.append(
            {
                "token": tok,
                "screen_tok_s": t.value,
                "confirm_median_tok_s": m.median,
                "improvement_pct": (
                    100.0 * (m.median - baseline_median) / baseline_median
                    if baseline_median
                    else None
                ),
            }
        )
    confirmed.sort(key=lambda c: c["confirm_median_tok_s"], reverse=True)

    summary = {
        "schema_version": "036.1",
        "shader": shader,
        "device_slug": slug,
        "fingerprint": fingerprint,
        "group_size": args.group_size,
        "budget": args.budget,
        "measured_trials": measured,
        "pruned": dict(pruned_reasons),
        "blocklist_size": len(blocklist.tokens),
        "baseline_median_tok_s": baseline_median,
        "noise_floor_cov": noise_cov,
        "finalists": confirmed,
        "winner": confirmed[0] if confirmed else None,
        "remeasure_pending": list(getattr(session, "remeasure", [])),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = SPEC / "results" / f"sweep_summary_{slug}_{shader}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"winner: {summary['winner']}")
    print(f"summary: {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shader", required=True, choices=tc.SHADERS)
    ap.add_argument(
        "--group-size",
        type=int,
        required=True,
        help="quantization group size of the pte being swept "
        "(this box's buffer ptes: 128)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=60,
        help="measured-trial budget (pruned trials are free)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="candidates per yaml-append + rebuild round",
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=15,
        help="stop after this many consecutive non-improving trials",
    )
    ap.add_argument(
        "--mma-k",
        type=int,
        default=None,
        help="coopmat MMA_K; defaults to the yaml's value for this shader "
        "(dq8ca is 32 on NVIDIA, which enumerates int8 only at 16x16x32)",
    )
    ap.add_argument(
        "--skip-gate",
        action="store_true",
        help="skip the small-shape bench gate; rely on the e2e output-hash and "
        "token check (required for dq8ca, whose tiles do not dispatch at the "
        "gate's shapes)",
    )
    ap.add_argument("--finalists", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--seed", action="append", default=[], help="extra seed token (repeatable)"
    )
    ap.add_argument(
        "--quirk",
        action="append",
        default=[],
        help="device quirk name, e.g. no_int8_wmma_sg32",
    )
    ap.add_argument("--pte", help="override pte path")
    ap.add_argument(
        "--remote",
        default="local",
        choices=("local", "android"),
        help="local = same-machine subprocess (measure.py); "
        "android = adb-tethered device via measure_android.py",
    )
    ap.add_argument(
        "--strict", action="store_true", help="abort on unrecovered control drift"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay-dir", default=str(SPEC / "results/replay-780m"))
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_sweep(args)


if __name__ == "__main__":
    main()
