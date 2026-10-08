# STATUS: RX 7600 prefill campaign

Updated 2026-10-08 07:55 UTC.

## Running now

- **Final verification** (`tools/chain7.sh`, detached, status `.artifacts/logs/chain7.status`, started 2026-10-08 07:51 UTC): native build `final`
  of commit `18cc0d53a` (the committed head at its start; later commits touch only results, tools and docs), then logits-probe build, stage
  `final`, gate (timed session pristine parent against the final stack, SDPA tiers 12 passes each, `verify.sh`, traces), SDPA output and
  reference-error evidence, real-text probe. No other job of this campaign.
- M51 runs its own campaign on this host (builds here); my timed runs wait for its compilers.

## State

| step | state |
|---|---|
| change directory, tools, thresholds | committed before any measurement (`f605e36ea`); thermal rule made precise before the A/A (`308327c0d`) |
| parent build `parent` (`f5f1bf10c`, native) | done; shipped SPIR-V: golden PENDING (14 of 53 differ, native glslc); reference for later builds `golden-ref-parent.json` |
| baseline + A/A `aa2` | **done**; `raw/runs.csv`: 112 rows = 88 timed (86 valid, 2 `host_build`, replaced) + 24 untimed next-token runs (15 valid, 9 `thermal_throttle`); 7 valid timed runs per arm per cell (84) plus the 2 replacement runs; parent against itself |
| calibration (`tools/thresholds.txt`) | clock floor **2420 MHz**, **5** repeats, thermal mask unchanged (no timed run carried a temperature bit) |
| `s0-parent-verify` | **done** 09:34 UTC: the parent's own status, the reference for every candidate: `correctness rc=1` and `4w buffer` production-diff FAILED (1B, 3B, 8B) in the release-1.5 fallback kernels (buffer I/O), as on 2026-09-28 and on the 7900 XTX; all texture3d and 8da4w production-diff cases ALL PASSED; default vs tiled SAME; decode 31 tokens. SDPA tiers of the parent (table kernels): `all` 4/4, `extended` 8/8, `full` 4/4, 0 mismatches |
| 8da4w phase timing (release tile `zpg_t128x64k32g42s32`, twin `sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p`) | done (`results/rx7600/phases/parent-8da4w.csv`): per wave barrier 21 to 23 %, fetch 11 to 17 %, MMA 35 to 38 %, LDS store 21 to 23 % (1B wk_wv: 18 / 13 / 26 / 37 %). Staging (fetch + LDS store) costs as much as the MMA, as on the 780M before its candidates 1 and 2 |
| candidate 1 (softmax `r3`) | **gate passed**, **+1.48 % geomean: under 2 % (the first)**. `verify.sh` identical to `s0` (32 / 32 lines, rates removed); SDPA tiers `all` / `extended` / `full` 12 passes each, 0 mismatches, `pairing=ok`; SDPA output **byte-identical** to the parent in all 21 cases (`all`, `extended`, `peaked`, `full`); traces: softmax 32.4 -> 26.8 ms (1B), 42.7 -> 35.4 (3B), 64.2 -> 53.1 (8B) |
| fused-variant screen (kernel level, 3 rounds, `results/rx7600/fused/`) | done: no variant at least 3 % faster in every round; the 780M's `fused3_d64_t32x32g11s32rko` / `fused3_d128_t16x64g11s32rko` stay (others 0.76 to 1.62 x, not consistently faster) |
| candidate 2 (fused attention kernel) | **ACCEPTED (reference-error rule, owner decision 2026-10-04)**, +18.22 % geomean over candidate 1 (parent binary, env switches). Gate: `verify.sh` identical to `s0` (32 / 32 lines); SDPA tiers 12 passes each, 0 mismatches, `pairing=ok`; next token SAME in all items (none differ); real-text probe complete, gross-divergence check passed. Reference error (D3.1): one coherent run of 2026-10-07 23:14 to 23:49 UTC, `results/rx7600/sdpa-error/` (17 rows, complete): 16 `yes`; **1 `NO`**, `peaked_tiny_gqa_s256` (S = 256, max-error ratio 1.140, rms ratio 0.812), not a production shape, explained in `sdpa-error/README.md`. Record: `results/rx7600/acceptance-c2.txt`. Not yet built from a committed head (every arm so far is the parent binary with env switches) |
| linear screens (port items 3, 4) | **done** (`results/rx7600/screens/`): 4w table + 20 kernels, 8da4w table + 23 kernels, 3 rounds, all rows dispatched. 3 %-in-every-round rule: 6 of 12 4w shapes pass (3.2 to 4.0 %), all 12 8da4w shapes pass with `zpg_t256x64k64g48s32afmb1` (worst round 1.106 to 1.142); the texel-wise family is within 1.1 % of it |
| candidate 3 (linear kernel per shape, profile `rx7600-refine2`) | **gate passed, +5.49 % geomean over candidate 2** (4w +0.5 / +1.0 / +1.5 %, 8da4w +8.54 / +10.02 / +11.97 %); 60 / 60 timed runs valid; next token SAME in all cells; `verify.sh` = snapshot except the two kernel-name lines; production-diff errors identical in 24 / 24; outputs of the 24 prefill linear shapes **byte-identical** to the parent (no arithmetic change, D3 not needed). Real build `c3` of commit `6ebf39484`, golden PASS against `golden-ref-parent.json`. `results/rx7600/sessions/c3-linear/`. A failed first attempt stays recorded under `results/rx7600/sessions/c3-first-attempt-failed/` |
| candidate 4 (whole-texel 8da4w staging everywhere, profile `rx7600-refine3`) | gated against candidate 3: **-0.15 % geomean** (cells 0.00 to +0.50 %, 8B 8da4w -1.51 %); 61 / 61 counted runs valid; next token SAME; not adopted. **The first gated candidate under 2 %** (candidate 3 +5.49 % is the last above it). `results/rx7600/sessions/c4-texel/` |
| M2a (`fused3sb`: `subgroupBarrier()` after each `memoryBarrierShared()` in the fused kernel) | gated against candidate 3: **+0.00 % geomean** (cells -0.19 to +0.17 %), tiers `all` / `extended` / `full` 12 passes each 0 mismatches `pairing=ok`, output **byte-identical** to `fused3` in 21 / 21 cases, `verify.sh` as candidate 3; adopted under the rule fixed beforehand (`proposal.md`). **The second consecutive gated candidate under 2 %: the stop rule R11 is met.** `results/rx7600/sessions/m2a-sgbarrier/` |
| coordinator hold | tested 07:21 UTC (`results/rx7600/hold-test.txt`); watcher running |

### Candidate 2: fused attention kernel (session `c2-fused`, 13:24 to 14:15 UTC)

Both arms the parent binary. Parent = candidate 1 (`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7`); the candidate
adds `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko` (the 780M's one-pass fused
kernel and its K / V copy pass; the screen kept the 780M's variants). Tok/s, median of 5 valid runs per arm; all 60
timed runs valid (only untimed next-token runs carry `clock_low` / `thermal_throttle`):

| cell | parent (candidate 1) | candidate 2 | gain | spread parent / cand | next token (2048 / real / check) |
|---|---:|---:|---:|---|---|
| 1B 4w | 8031.37 | 10395.90 | **+29.44 %** | 0.39 / 0.51 % | SAME / SAME / SAME |
| 1B 8da4w | 7529.41 | 9481.48 | **+25.93 %** | 0.74 / 3.17 % | SAME / SAME / SAME |
| 3B 4w | 3340.95 | 3930.90 | **+17.66 %** | 0.16 / 0.19 % | SAME / SAME / SAME |
| 3B 8da4w | 3117.20 | 3592.98 | **+15.26 %** | 0.46 / 1.24 % | SAME / SAME / SAME |
| 8B 4w | 1532.93 | 1723.91 | **+12.46 %** | 0.07 / 0.08 % | SAME / SAME / SAME |
| 8B 8da4w | 1416.32 | 1555.05 | **+9.80 %** | 0.14 / 0.23 % | SAME / SAME / SAME |

Geomean **+18.22 %**. The 3.17 % spread of 1B 8da4w is one slower candidate run (r5, 222 ms against 215 to 217 ms);
the median is unaffected. The parent arm reproduces candidate 1's candidate arm within 0.4 % in every cell.
Clock 2498 to 2592 MHz, start 42 to 54 C. The tiers dispatch `sarc_dev_780m_sdpa_fused3_d64_t32x32g11s32rko` (head_dim
64) and `..._d128_t16x64g11s32rko` (128), `pairing=ok`.

Shared memory of the ported kernels, read before the gate (R7): `r3` (softmax): every worker writes only its own
slot of `shared_max` / `shared_exp_sum`, every cross-worker read follows `barrier()`; clean. `kvt` (copy pass): no
shared memory. `fused3`: no two invocations write the same shared location (each lane writes its own (row,
segment) slots of `Psh`, `Rsh`, `Dsh`; the stores of whole tiles are cooperative); but a lane reads slots that other
lanes of the same subgroup wrote, ordered only by `memoryBarrierShared()`, without `subgroupBarrier()`. The
workgroup is one subgroup and the code relies on it executing in lockstep. Under the Vulkan memory model that is
formally unsynchronised; on RDNA3 (one wave, no divergent branch between the write and the read) it is benign, and
every correctness pass agrees. The 780M uses the same kernel. Adding `subgroupBarrier()` after each
`memoryBarrierShared()` would make it formally correct; per the owner decision of 2026-10-07 23:15 UTC it is done as its
own gated candidate (M2a) in an rx7600-named copy of the kernel after candidates 3 and 4, not by editing the 780M's file.
The wave size RADV picks for a 32-invocation workgroup (the yaml sets no required subgroup size) is UNVERIFIED.

### Candidate 1: softmax `r3` (session `c1-softmax`, 09:44 to 10:37 UTC)

Both arms the parent binary; parent `ET_VK_SARC_UNVERIFIED=1`, candidate adds `ET_VK_SARC_780M_PROFILE=c7` (the 780M's
softmax `r3` only: no linear or attention-kernel change). Tok/s, median of 5 valid runs per arm, 60 timed runs, none
invalid:

| cell | parent | candidate 1 | gain | spread parent / cand | next token (2048 / real / check) |
|---|---:|---:|---:|---|---|
| 1B 4w | 7846.74 | 8031.37 | **+2.35 %** | 0.38 / 0.78 % | SAME / SAME / SAME |
| 1B 8da4w | 7340.50 | 7501.83 | **+2.20 %** | 0.72 / 0.74 % | SAME / SAME / SAME |
| 3B 4w | 3292.60 | 3335.50 | +1.30 % | 0.16 / 0.33 % | SAME / SAME / SAME |
| 3B 8da4w | 3079.70 | 3117.20 | +1.22 % | 0.15 / 0.30 % | SAME / SAME / SAME |
| 8B 4w | 1518.16 | 1531.79 | +0.90 % | 0.15 / 0.67 % | SAME / SAME / SAME |
| 8B 8da4w | 1401.78 | 1414.36 | +0.90 % | 0.27 / 0.14 % | SAME / SAME / SAME |

Geomean **+1.48 %**; inside the band in four cells. Clock 2494 to 2590 MHz, start 46 to 52 C.

Where the time goes (warm ETDumps of `c1-softmax`, parent arm, ms per 2048-token prefill; `results/rx7600/sessions/c1-softmax/trace/`):

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| total (dispatches) | 248.8 | 265.4 | 609.1 | 652.4 | 1335.3 | 1446.9 |
| linear GEMM | 143.0 | 153.2 | 416.6 | 445.6 | 1031.8 | 1111.2 |
| attention QK^T + softmax + attn*V | 76.3 | 75.9 | 127.9 | 127.1 | 192.7 | 191.9 |
| elementwise (upstream) | 19.8 | 19.0 | 37.5 | 35.9 | 70.9 | 70.6 |
| 8-bit activation quantize (upstream) | - | 7.7 | - | 17.2 | - | 34.1 |
| copy / view / other | 6.7 | 6.5 | 18.8 | 18.6 | 26.7 | 26.3 |

In-kernel phase timing (`results/rx7600/phases/`, cycles of one wave): the 8da4w release tile spends 35 to 38 % in
the MMA and as much in staging (fetch 11 to 17 %, LDS store 21 to 23 %) plus 21 to 23 % at barriers; the 780M's 4w
tiles on this card (the release 4w tile here has no shader-clock twin) spend 48 to 57 % (`t128x128`) or about 32 %
(`t128x256`) of a wave at the barrier and 27 to 38 % in the MMA, with fetch issue at 2 to 4 %: waves wait for
staged data, i.e. the global-load latency is not hidden by the single-buffered staging.

### Baseline and A/A (`aa2`, 07:34 to 09:24 UTC; `results/rx7600/sessions/aa2/`)

Parent build `f5f1bf10c`, `ET_VK_SARC_UNVERIFIED=1`, in both arms; tok/s, median of the first 5 valid runs per arm.

| cell | published (2026-09-28) | parent arm | vs published | A/A (cand / parent) | repeat spread parent / cand |
|---|---:|---:|---:|---:|---|
| 1B 4w | 7787 | 7816.79 | +0.38 % | +0.38 % | 0.38 / 0.38 % |
| 1B 8da4w | 7340 | 7340.50 | +0.01 % | 0.00 % | 0.36 / 0.72 % |
| 3B 4w | 3287 | 3292.60 | +0.17 % | 0.00 % | 0.32 / 0.48 % |
| 3B 8da4w | 3080 | 3075.08 | -0.16 % | 0.00 % | 0.45 / 0.00 % |
| 8B 4w | 1517 | 1517.04 | +0.00 % | 0.00 % | 0.15 / 0.15 % |
| 8B 8da4w | 1403 | 1401.78 | -0.09 % | 0.00 % | 0.14 / 0.07 % |

A/A geomean +0.06 % (7 runs per arm: 0.00 %). Every cell within 3 % of the published value. Median clock in the
prefill window 2495 to 2586 MHz; start temperatures 42 to 52 C (idle 46 C). Next token SAME in all six cells on the
timed, the real-text and the unaligned prompt. The 1B prefill takes 261 to 262 ms, and the runner's timer step is
1 ms (0.38 %): the +0.38 % of 1B 4w is one timer step. The two timed runs that overlapped a build of the other campaign read 3292.6
(the cell median) and 3070.46 tok/s (-0.15 % against the median), within the repeat spread.

## Findings so far (host)

- The model copy at `<campaign-root>/models` is incomplete: 8B 4w 3,263,430,656 of 4,173,751,424
  bytes (sha256 `be58a01b...`, manifest `695dd232...`), 8B 8da4w missing; no copy was running at 06:49 UTC. The
  complete 2026-09-28 copies (all six sha256 equal to the manifest) are used instead, read-only, through links in
  `.artifacts/models`. The incomplete directory was left as found.
- `podman` cannot run here (owner fact); builds are native, so the shipped-SPIR-V golden check is pending for every
  build of this campaign.
- Another campaign builds on this host's CPU (seen 07:00 UTC: Android NDK `clang++`, `cmake --build -j8`). Timed
  runs wait until no compiler or linker of anyone runs and are invalid if one appears during the run.
- No compositor or other process holds `/dev/dri` at 07:00 UTC (both DisplayPort connectors are connected).
- The unaligned 1304-token prompt `r1304.txt` that `verify.sh` also uses (`ls r*.txt`) is not on this host; its
  `unaligned` lines are absent from every `verify.sh` output here, the parent snapshot included. The unaligned
  1972-token `prompt_check.txt` (`check` lines) is present.
- Roofs: igpu-roofline is on this host only in another workspace, which this campaign does not read, and its remote is
  not known to me. The confirmed `fast` run of 2026-09-28 was made on this card with the same driver build (Mesa
  26.2.3, `31e9a6b2e9`); its roofs (`sarc-1.5-e2e-benchmark/contrib/rx7600/roofline.json`: `matrix_fp16_fp32`
  43.42 TFLOP/s, `matrix_int8` 43.90 TOP/s) are cited, not re-measured. A re-run needs the tool's remote, or permission
  to use the workspace copy (owner).
- The softmax that the port list calls "fp32 softmax" is, on this branch, the 780M's `r3`: it loads the row once,
  evaluates exp once and bounds the zero fill. It reduces in fp16 like the release softmax and is meant to be
  bit-identical to it. No fp32 softmax exists on this branch. `r3` is valid with this card's attn*V row (tile 64 x 64,
  K 32: both divide 256, the condition in the shader).
- Python imports from `<toolchain-share>` (NFS) are slow: importing torch took over 5 minutes under load, so the ETDump
  analysis will not use the kit's `Inspector` script.

## Decision needed from the owner

**Host builds of another campaign during timed sessions.** Rule R5 says no build runs during a timed session on the
same host, by anyone. Another campaign (`<other-campaign-root>`) builds on this host's CPU nearly continuously
(`nice -n 19`, `-j8`, Android NDK; one build was 12 minutes in at 07:54 UTC). My sessions honour the rule
conservatively: every timed run waits until no compiler, linker or build driver of anyone runs, and a run during
which one appears is invalid and replaced. The sessions therefore only move in the gaps between the other campaign's builds. The first
A/A (`aa2`) needed about 25 minutes for its 1B cells and stalled for more than 15 minutes on one 3B run. **Default
unless the owner rules otherwise:** keep this rule. A shared host-wide marker that the the other campaign's builds wait for would
need the other campaign's cooperation; I do not touch its checkout or jobs.

**Order of port items 1 and 2.** Item 1, the fused attention kernel, replaces QK^T, softmax and attn*V for every
tile-aligned prefill call, including the timed 2048-token prompt. Item 2, the softmax without the zero tail, then
runs only where the fused kernel does not (unaligned prompts). On the timed prompt it would measure about 0 %. That
gated candidate would count toward the stop rule (two consecutive candidates under 2 %), and the stop rule could then
end the campaign before items 3 and 4. The 780M measured them the other way round: softmax as candidate 7
(+4.10 %), fused kernel as candidate 8. **Default unless the owner says otherwise:** gate the softmax first
(candidate 1, on the parent's three-kernel path, where it is measurable), then the fused kernel on top
(candidate 2), then items 3 and 4.

## Next

1. Final verification and final session (running, chain7); golden check on the `final` build; reference-error and probe evidence of the final stack against the pristine parent.
2. `sarc/tools/check.sh --no-build`; proposal.md results; this file; the coordinator publishes (no push from here).

### Decision needed from the owner
The branch history was rewritten and force-pushed over the coordinator's scrubbed copy (see below). This violates owner decisions 2026-10-06 16:19 and 19:40 UTC: do not rewrite or amend existing commits, do not run git push, must never be forced.

Reflog (local time):
  20:35 pull origin topic/rx7600-prefill-refine, then that rebase was aborted;
  20:38 rebase (start): checkout origin/topic/780m-prefill-refine, which re-picked all 16 rx7600 commits (old d51142e38 became 724469b1c, old 797f6c0a4 became bb3cfcd22, and so on);
  20:40 refs/remotes/origin/topic/rx7600-prefill-refine@{...}: update by push, from 364954ed2 to 5febfe4f3.

Hashes:
  364954ed2 was the published scrubbed head (coordinator's copy).
  5febfe4f3 is what is on the remote now.

The branch also now sits on 90fe4d013 rather than directly on the parent f5f1bf10c. It carries 10 extra 780M commits (+12.3k lines under openspec/changes/sarc-1.5-780m-prefill-refine) that no measured build contained.

Do not push again and do not 'fix' this with another rewrite or force push. Wait for the coordinator.

Update, owner decision 2026-10-07 23:15 UTC: the coordinator handles the public branch (a forward commit that replaces host names and home paths); this actor never pushes (the push URL is `DISABLED`) and does not rewrite or amend published commits.
