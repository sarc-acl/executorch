#!/usr/bin/env python3
"""Run one 2048-token-prefill e2e screening measurement per candidate token
(plus baseline) on the M5 EVT1 device, via adb through the ssh host.

Measurement mode (default): drives adb, parses the PyTorchObserver JSON line
from llama_main's stdout, and writes screen_results.json.

--decide-only mode: computes screen_ratio against BASELINE_TOKEN within the
given model_stage and writes escalation_decisions.json. Does not touch the
device.
"""
import argparse
import json
import re
import subprocess
import sys

HOST = "yanwen.xu@sj1-dmckee-d01"
SERIAL = "0000088f8e579c33"
DEVICE_DIR = "/data/local/tmp/llama_vk"
RUNNER = "llama_main_028"
PROMPT_FILE = "p2048_exact.txt"

BASELINE_TOKEN = "(unset — default dispatch)"


def adb_shell(cmd: str, timeout: int = 900) -> str:
    full = f'adb -s {SERIAL} shell "{cmd}"'
    ssh_cmd = ["ssh", HOST, full]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def driver_hash() -> str:
    out = subprocess.run(
        [
            "ssh",
            HOST,
            f"adb -s {SERIAL} shell md5sum /vendor/lib64/hw/vulkan.samsung.so",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.stdout.strip().split()[0] if out.stdout.strip() else "UNKNOWN"


def clocks_pinned() -> bool:
    out = subprocess.run(
        [
            "ssh",
            HOST,
            f"adb -s {SERIAL} shell cat /sys/kernel/gpu/min_freq /sys/kernel/gpu/max_freq",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    vals = out.stdout.split()
    return len(vals) == 2 and vals[0] == vals[1]


def parse_observer(stdout: str):
    m = re.search(r"PyTorchObserver (\{.*\})", stdout)
    if not m:
        return None
    return json.loads(m.group(1))


def _safe_name(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", token)


def run_one(
    token: str,
    model_used: str,
    model_stage: str,
    board: str,
    do_coherence: bool,
    raw_dir: str = None,
    stage: str = "screen",
    run_index: int = 1,
):
    env_prefix = ""
    if token != BASELINE_TOKEN:
        env_prefix = f"ET_VK_Q4GSW_COOPMAT_VARIANT={token} "

    tag = f"{stage}_{_safe_name(token)}_run{run_index}"

    if do_coherence:
        coherence_cmd = (
            f"cd {DEVICE_DIR} && {env_prefix}./{RUNNER} "
            f"--model_path={DEVICE_DIR}/{model_used} --tokenizer_path={DEVICE_DIR}/tokenizer.model "
            f"--prompt='The capital of France is' --seq_len=48 --temperature=0 --warmup=false"
        )
        print(f"[{token}] running coherence check...", file=sys.stderr, flush=True)
        out = adb_shell(coherence_cmd)
        if raw_dir:
            with open(f"{raw_dir}/coherence_{tag}.log", "w") as f:
                f.write(out)
        if "Paris" not in out:
            raise RuntimeError(
                f"Coherence check FAILED for token={token}: {out[-500:]}"
            )

    bench_cmd = (
        f"cd {DEVICE_DIR} && ET_VK_EXECUTE_NODE_THRESHOLD=16 {env_prefix}./{RUNNER} "
        f"--model_path={DEVICE_DIR}/{model_used} --tokenizer_path={DEVICE_DIR}/tokenizer.model "
        f"--prompt_file={DEVICE_DIR}/{PROMPT_FILE} --num_bos=1 --max_new_tokens=1024 "
        f"--ignore_eos --temperature=0 --warmup=true"
    )
    print(
        f"[{token}] running 2048-prefill {stage} (run {run_index})...",
        file=sys.stderr,
        flush=True,
    )
    out = adb_shell(bench_cmd)
    if raw_dir:
        with open(f"{raw_dir}/{tag}.log", "w") as f:
            f.write(out)
    obs = parse_observer(out)
    if obs is None:
        raise RuntimeError(f"No PyTorchObserver line for token={token}: {out[-1000:]}")
    print(
        f"[{token}] prefill_tok_s={obs['prefill_token_per_sec']}",
        file=sys.stderr,
        flush=True,
    )

    return {
        "candidate_token": token,
        "model_stage": model_stage,
        "stage": stage,
        "run_index": run_index,
        "prefill_tok_s": obs["prefill_token_per_sec"],
        "decode_tok_s": obs.get("decode_token_per_sec"),
        "model_used": model_used,
        "driver_hash": driver_hash(),
        "board": board,
        "clocks_pinned": clocks_pinned(),
        "coherence_checked": do_coherence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefilter", required=True)
    ap.add_argument("--port-verification", required=True)
    ap.add_argument("--model-stage", required=True)
    ap.add_argument("--model-used", default="llama3_1_8b_4w_buffer_ctx3072.pte")
    ap.add_argument("--board", default="sj1-dmckee-d01/0000088f8e579c33")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--screen-results", help="required with --decide-only")
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="directory to write raw per-run adb stdout/stderr logs, "
        "and to incrementally persist results after each run",
    )
    args = ap.parse_args()

    prefilter = json.loads(open(args.prefilter).read())
    port_verif = json.loads(open(args.port_verification).read())
    passing_tokens = {
        p["candidate_token"] for p in port_verif if p["correctness_status"] == "pass"
    }
    candidates = [c["token"] for c in prefilter if c["token"] in passing_tokens]

    if args.decide_only:
        screen = json.loads(open(args.screen_results).read())
        by_token = {
            r["candidate_token"]: r
            for r in screen
            if r["model_stage"] == args.model_stage
        }
        baseline = by_token.get(BASELINE_TOKEN)
        if baseline is None:
            print("ERROR: no baseline screen result found", file=sys.stderr)
            sys.exit(1)
        decisions = []
        for tok in candidates:
            r = by_token.get(tok)
            if r is None:
                continue
            ratio = (r["prefill_tok_s"] - baseline["prefill_tok_s"]) / baseline[
                "prefill_tok_s"
            ]
            decisions.append(
                {
                    "candidate_token": tok,
                    "model_stage": args.model_stage,
                    "screen_ratio": round(ratio, 4),
                    "escalated": ratio >= -0.10,
                }
            )
        with open(args.out, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"Wrote {len(decisions)} escalation decisions to {args.out}")
        return

    import os

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)

    results = []
    coherence_done = set()

    def persist():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Baseline first.
    do_coh = args.model_used not in coherence_done
    results.append(
        run_one(
            BASELINE_TOKEN,
            args.model_used,
            args.model_stage,
            args.board,
            do_coh,
            args.raw_dir,
        )
    )
    coherence_done.add(args.model_used)
    persist()
    for tok in candidates:
        do_coh = args.model_used not in coherence_done
        results.append(
            run_one(
                tok, args.model_used, args.model_stage, args.board, do_coh, args.raw_dir
            )
        )
        coherence_done.add(args.model_used)
        persist()

    print(f"Wrote {len(results)} screen results to {args.out}")


if __name__ == "__main__":
    main()
