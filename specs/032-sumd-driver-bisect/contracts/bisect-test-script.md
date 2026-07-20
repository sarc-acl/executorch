# Contract: Bisect Test Procedure

**Feature**: `032-sumd-driver-bisect` | **Date**: 2026-07-16

This is the one "interface" this feature defines: the procedure/script invoked once per commit
under test (`scripts/bisect-test.sh`, authored during implementation), whether it's called
directly by `git bisect run` or driven manually one step at a time (per `research.md` §3, this
study drives it manually — but the contract is identical either way, since `git bisect`'s exit-code
convention is the natural contract regardless of who invokes it).

## Invocation

```bash
scripts/bisect-test.sh <sumd-worktree-dir> [bisect_role]
```

Run with cwd anywhere; `<sumd-worktree-dir>` is the already-checked-out SHA-named SUMD worktree
(e.g. `/local/yanwen.xu/sumd/898709039d/`) for the commit currently under test. This script does
**not** create the worktree or check out the commit — that's the caller's responsibility (`git
worktree add` per `sumd/CLAUDE.md`), keeping this script's job narrowly "build this checkout, flash
it, measure it, verdict it." `bisect_role` is one of `endpoint-old` / `endpoint-new` /
`bisect-step` / `skip-adjacent-probe` (data-model.md) and is recorded verbatim into the appended
trace row; defaults to `bisect-step` if omitted.

## Preconditions

- The target worktree exists and is checked out at the commit under test (detached HEAD).
- M41 (serial `00000a34cdd4abd3`) is reachable via `ssh xgpusw-debug07` and responsive.
- The 1B 4w/8da4w PTEs (ctx supporting 2048-token prefill) are already staged on-device or on NFS.
- Clocks are pinned to M41's own max (980/5333/800 MHz) — this script re-verifies the pin via
  sysfs readback rather than assuming a prior step's pin still holds.

## Side effects

1. Builds the SUMD driver from the worktree's current HEAD (`uv run scripts/run.py --os android
   --build --build-type release`, with `vulkan-sdk` stripped from `LD_LIBRARY_PATH`).
2. Stages the resulting `.so` to NFS (`cmp`-verified) and flashes it to M41.
3. Captures the on-device driver identity (`adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so`
   — never `logcat -d | grep SUMD`, which is documented as unreliable for identifying which build
   is active: it dumps build-ancestry commit hashes, not the active build's own).
4. Re-pins via `pin_freqs.sh` (`S=00000a34cdd4abd3 GPUFREQ=980000 MIFFREQ=5333000 INTFREQ=800000`,
   per research.md §4) before every measurement — cheap and idempotent, so re-running it every
   step (rather than trusting a prior step's pin) removes any risk of drift between builds; checks
   the script's own printed readback matches the requested values.
5. Runs the release/1.3 vanilla `llama_main_rel1.3` runner twice (4w, 8da4w; Llama 3.2 1B;
   2048-token prefill; 1 rep each).
6. Appends one row to the bisect-trace log (per `data-model.md`'s Bisect Step schema) —
   irrespective of outcome, including failures.

**Never performed**: reading, opening, or grepping any SUMD source file under `drivers/`,
`external/`, `test/`, `agents/`, `cmake/`, or any `CMakeLists.txt`/source file in the worktree —
per `/local/yanwen.xu/sumd/CLAUDE.md` Rule 0. A build failure's diagnosis is limited to the
already-documented gotcha (GpuRt "Too many users") or general "unknown build failure"; it never
extends into inspecting *why* at the source level.

## Outputs (exit code contract — matches `git bisect run`'s convention)

| Exit code | Meaning | Condition |
|---|---|---|
| `0` | good | Build succeeded, device stayed responsive, and `prefill_8da4w_tok_s > prefill_4w_tok_s` (strict comparison, no tie-break) |
| `1` | bad | Build succeeded, device stayed responsive, and `prefill_8da4w_tok_s <= prefill_4w_tok_s` |
| `125` | skip | Build failed (even after the documented `LD_LIBRARY_PATH` retry), the driver crashed/hung on-device, or the clock pin could not be verified after a re-pin attempt |

Stdout/stderr: human-readable log of every side-effect step above, plus the final one-line
verdict summary (`commit=<sha> driver_version="<string>" 4w=<x> 8da4w=<y> verdict=<good|bad|skip>
[reason=<...>]`) — this line is what gets appended to the bisect-trace log (`data-model.md`).

## Postconditions

- Exactly one row is appended to the bisect-trace log for this commit, regardless of outcome.
- The SHA-named worktree is left in place afterward (per `sumd/CLAUDE.md`, these are never removed
  without explicit instruction) — it is not cleaned up by this script.
