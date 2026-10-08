# sarc-1.5-orin-fused-port

Second campaign on the Jetson Orin Nano (`duck-naughty`). Branch `topic/orin-fused-port`, forked from
`topic/orin-prefill-refine` at `8973ced76`. It ports one thing on top of the first campaign's final stack: the
fused attention kernel of the Radeon 780M (third structure, with the RX 7600's subgroup barriers). A port, by
owner decision N1: no sampled search, no enumeration, no tile sweep; two candidates at most.

Parent of every comparison: `8973ced76` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack, `sarc-1.5-orin-prefill-refine/proposal.md`, "Outcome"). The pristine state is
the same commit with no environment.

Day-to-day state is in `STATUS.md`. This file is completed as the campaign proceeds.

## Thresholds (fixed before the first measurement)

`tools/thresholds.txt`, committed before any run of this campaign:

- baseline: every cell of the parent within 3 % of `s10-final` (1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
  8B 295.53 / 269.19 tok/s, 4w / 8da4w);
- noise band +-2 %; median of 5 valid interleaved runs per arm and cell;
- clock floor: floor(0.97 x the lowest per-cell median clock of this campaign's own A/A session), device-wide;
- a run needs at least 5 clock samples in its prefill window (0.1 s sampler; the shortest prefill is 1.2 s);
- kernel screen margin 3 % in every round;
- reference-error rule (D3): "the parent" is this campaign's parent, the tuned stack's arithmetic; the stock
  kernels' error is reported beside it, not judged;
- candidate 2 only if the unpacked one-pass form is at least 3 % faster at kernel level in every round.

## Tools: what differs from the sibling (`sarc-1.5-orin-prefill-refine/tools`)

Copied, originals untouched. Not carried over: the `chain*.sh` of the first campaign, its generators
(`gen_orin_*.py`, `orin_refine.py`), the screen, phase-timing, roofline, decode and second-device tools and the
old local hook patch.

| tool | change |
|---|---|
| `common.sh` | device directory `~/hmz-sarc-orin-fused`, change directory `sarc-1.5-orin-fused-port`, workstation root `/mnt/linux-share/hmz-campaigns/jetson-fused` with the artifact directory `.artifacts` itself, `PARENT_COMMIT=8973ced76`, campaign tag `orin-fused-port`; the second Orin is removed (not available) |
| `pull.sh` | the second Orin's mirror removed |
| `build-orin.sh`, `deploy.sh`, `drun.sh` | comments and paths only |
| `thresholds.txt` | new |
