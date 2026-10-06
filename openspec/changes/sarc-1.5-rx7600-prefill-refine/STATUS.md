# STATUS: RX 7600 prefill campaign

Updated 2026-10-06 10:45 UTC. Artifacts: `/local/yanwen.xu/campaign-rx7600/.artifacts/`.

## Running now

- `chain0.sh` (detached, `.artifacts/logs/chain0.status`): `s0-parent-verify` (unmodified `verify.sh` on the parent
  build, `ET_VK_SARC_UNVERIFIED=1`), then the SDPA tiers of the parent, one pass each (control).
- `chain1.sh` (waits for chain0): logits_probe build; **candidate 1** gate (softmax `r3`, `ET_VK_SARC_780M_PROFILE=c7`,
  on the parent binary) with its bit-identity evidence; then the kernel-level screen of the fused attention variants.
- `chain2.sh` (waits for chain1): kernel-level screens of 21 4w and 24 8da4w linear kernels on the twelve real shapes.
- `hold.sh watch` (coordinator hold watcher).

## State

| step | state |
|---|---|
| change directory, tools, thresholds | committed before any measurement (`f605e36ea`); thermal rule made precise before the A/A (`308327c0d`) |
| parent build `parent` (`f5f1bf10c`, native) | done; shipped SPIR-V: golden PENDING (14 of 53 differ, native glslc); reference for later builds `golden-ref-parent.json` |
| baseline + A/A `aa2` | **done**, 120 timed runs, 116 valid (2 `host_build`, re-run); parent against itself |
| calibration (`tools/thresholds.txt`) | clock floor **2420 MHz**, **5** repeats, thermal mask unchanged (no timed run carried a temperature bit) |
| `s0-parent-verify` | **done** 09:34 UTC: the parent's own status, the reference for every candidate: `correctness rc=1` and `4w buffer` production-diff FAILED (1B, 3B, 8B) in the release-1.5 fallback kernels (buffer I/O), as on 2026-09-28 and on the 7900 XTX; all texture3d and 8da4w production-diff cases ALL PASSED; default vs tiled SAME; decode 31 tokens. SDPA tiers of the parent (table kernels): `all` 4/4, `extended` 8/8, `full` 4/4, 0 mismatches |
| 8da4w phase timing (release tile `zpg_t128x64k32g42s32`, twin `sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p`) | done (`results/rx7600/phases/parent-8da4w.csv`): per wave barrier 21 to 23 %, fetch 11 to 17 %, MMA 35 to 38 %, LDS store 21 to 23 % (1B wk_wv: 18 / 13 / 26 / 37 %). Staging (fetch + LDS store) costs as much as the MMA, as on the 780M before its candidates 1 and 2 |
| candidate 1 (softmax `r3`) | timed session done (`c1-softmax`): **+1.48 % geomean**, under 2 %; SDPA tiers, `verify.sh`, traces and bit-identity evidence running |
| coordinator hold | tested 07:21 UTC (`results/rx7600/hold-test.txt`); watcher running |

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
1 ms (0.38 %): the +0.38 % of 1B 4w is one timer step. The two timed runs that overlapped an M51 build read 3292.6
(the cell median) and 3070.46 tok/s (-0.15 % against the median), within the repeat spread.

## Findings so far (host)

- The model copy at `/local/yanwen.xu/campaign-rx7600/models` is incomplete: 8B 4w 3,263,430,656 of 4,173,751,424
  bytes (sha256 `be58a01b...`, manifest `695dd232...`), 8B 8da4w missing; no copy was running at 06:49 UTC. The
  complete 2026-09-28 copies (all six sha256 equal to the manifest) are used instead, read-only, through links in
  `.artifacts/models`. The incomplete directory was left as found.
- `podman` cannot run here (owner fact); builds are native, so the shipped-SPIR-V golden check is pending for every
  build of this campaign.
- The M51 campaign builds on this host's CPU (seen 07:00 UTC: Android NDK `clang++`, `cmake --build -j8`). Timed
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
- Python imports from `/tool/pkg` (NFS) are slow: importing torch took over 5 minutes under load, so the ETDump
  analysis will not use the kit's `Inspector` script.

## Decision needed from the owner

**Host builds of the M51 campaign during timed sessions.** Rule R5 says no build runs during a timed session on the
same host, by anyone. The M51 campaign (`/local/yanwen.xu/campaign-m51`) builds on this host's CPU nearly continuously
(`nice -n 19`, `-j8`, Android NDK; one build was 12 minutes in at 07:54 UTC). My sessions honour the rule
conservatively: every timed run waits until no compiler, linker or build driver of anyone runs, and a run during
which one appears is invalid and replaced. The sessions therefore only move in the gaps between M51 builds. The first
A/A (`aa2`) needed about 25 minutes for its 1B cells and stalled for more than 15 minutes on one 3B run. **Default
unless the owner rules otherwise:** keep this rule. A shared host-wide marker that the M51 builds wait for would
need the M51 campaign's cooperation; I do not touch its checkout or jobs.

**Order of port items 1 and 2.** Item 1, the fused attention kernel, replaces QK^T, softmax and attn*V for every
tile-aligned prefill call, including the timed 2048-token prompt. Item 2, the softmax without the zero tail, then
runs only where the fused kernel does not (unaligned prompts). On the timed prompt it would measure about 0 %. That
gated candidate would count toward the stop rule (two consecutive candidates under 2 %), and the stop rule could then
end the campaign before items 3 and 4. The 780M measured them the other way round: softmax as candidate 7
(+4.10 %), fused kernel as candidate 8. **Default unless the owner says otherwise:** gate the softmax first
(candidate 1, on the parent's three-kernel path, where it is measurable), then the fused kernel on top
(candidate 2), then items 3 and 4.

## Next

1. Candidate 1 (softmax `r3`) gate; its traces are the first locate step (ETDump families of the six cells, both arms).
2. Fused-variant screen, then candidate 2 (fused attention kernel) with the reference-error evidence (D3).
3. Linear screens, then candidates 3 and 4 (kernel per shape; whole-texel 8da4w staging).
