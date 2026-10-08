# STATUS: sarc-1.5-orin-fused-port

**2026-10-08 19:55 UTC. Started 19:37 UTC. Nothing measured yet: the parent cross-build waits for the
workstation's build lock; candidate 1 is written, committed (`0f14f2a1a`) and queued to build.**

All times are UTC from `date -u`.

## Running now

- Workstation (detached, `.artifacts/jobs/<job>.{status,out}`):
  - `build-parent`: cross-build of the parent `8973ced76` (tag `parent`). Source tree exported 19:43
    (`build/parent.src.txt`: `git archive` + 30 pinned submodules, tree sha256 `3bbbd4cb...`); the container build
    waits for `~/.cache/gpu-lab/lock-desktop-build`, held exclusively since before 19:45 by the Arc B580
    campaign's own build (pid 4091732, `b580-fused/.../build-both.sh`). Not mine; waiting, as the task says.
  - `build-topic1`: queued behind `build-parent`: `0f14f2a1a` (hook + candidate 1), tag `topic1`.
  - `shadercheck1`: the three new shaders through the cross image's glslc, queued behind the same lock.
- Device `duck-naughty`: nothing of this campaign. Found idle at 19:37 (no runner process, no Actions job, 6.5 GB
  available, swap unused, 53 to 57 C, power mode 15 W, governor `nvhost_podgov`, devfreq 306 to 612 MHz).
  Tools deployed to `~/hmz-sarc-orin-fused/` at 19:48; the probes were tested there (`gtemp` 56 C, `gclk`
  306 MHz idle, sampler 12 lines in 1.2 s, `others` empty, `cool_to` returns).
- Coordinator hold: `tools/HOLD.md` (device: `~/hmz-sarc-orin-fused/HOLD`; builds: `.artifacts/HOLD`).

## Parent

`8973ced76` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack; the softmax hook `307abb2ed` is committed, so the softmax is named by the
dev-zone variable, no local patch). Expected (`s10-final`): 1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
8B 295.53 / 269.19 tok/s (4w / 8da4w). Pristine state: the same commit, no environment.

## Ceiling (computed before candidate 1, from the first campaign's `s10-final` traces)

Attention (QK^T + softmax + attn*V) in the parent, ms per 2048-token prefill, and its share of the dispatch
total (`sarc-1.5-orin-prefill-refine/proposal.md`, "Where the gain is"):

| | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention / total, ms | 302 / 1359 | 302 / 1465 | 516 / 3235 | 517 / 3571 | 783 / 6913 | 784 / 7587 | |
| share | 22.2 % | 20.6 % | 16.0 % | 14.5 % | 11.3 % | 10.3 % | |
| gain if attention cost nothing (ceiling) | +28.6 % | +26.0 % | +19.0 % | +16.9 % | +12.8 % | +11.5 % | +19.0 % |
| gain if 76 % of it goes (the RX 7600's fused kernel) | +20.3 % | +18.6 % | +13.8 % | +12.4 % | +9.4 % | +8.5 % | +13.7 % |
| gain if 50 % of it goes | +12.5 % | +11.5 % | +8.7 % | +7.8 % | +6.0 % | +5.4 % | +8.6 % |

The task expects +5 to +12 % geomean. On this device attention is bound by memory traffic (the first campaign:
half of the softmax is the first read of a row) and shared memory is expensive, so the lower half is as likely
as the upper.

## Candidate 1: what was written (commit `0f14f2a1a`), not measured yet

- Hook: `9d91480b2` = cherry-pick of `1c8861aa7e`, alone, same patch-id (`0027e89f...`), 49 lines in `SDPA.cpp`,
  `sarc/SdpaCoopmat.{cpp,h}`, `sarc/Select.h`. `sarc/tools/check.sh --no-build` on `0f14f2a1a`: `check.sh: PASS`;
  `test_sarc_select` on the release tables `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent,
  unverified off)`, the parent's line.
- `glsl/sarc_dev/sarc_dev_orin_sdpa_fused3sb.{glsl,yaml}`: the RX 7600's `fused3sb` (`b3bb758e38`), identical from
  `#version` on (`diff`: no difference); `sarc_dev_orin_sdpa_{kvt,vt}`: the 780M's copy passes, identical from
  `#version` on. Variants: `rko` (one pass, packed: candidate 1), `ro` (one pass, unpacked), `rk`, `r` (two
  passes) for head_dim 64 (`t32x32`) and 128 (`t16x64`).
- `impl/sarc_dev/orin/SdpaOrinFused.cpp`: port of the 780M's selector; sets `Override::sdpa_fused_*` only when a
  variant is named (profile `orin-fused1` or `ET_VK_SARC_ORIN_SDPA_FUSED`), so with nothing selected both are null.
- `Overrides.cpp`, blocks `orin-fused profiles` / `orin-fused override`: profile `orin-fused1` = the preferences
  of `orin-refine5`; an `orin-fused*` profile names the softmax `orin_g64` itself. Candidate environment:
  `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`.
- `test_llama_microbench.cpp`: the fused kernel is recognised (timing, `fused=` in `[sdpa-kernels]`, pairing: no
  QK^T / softmax / attn*V kernel may run beside it); the 780M's tiers `peaked` and `fused` added.

### Shared-memory reading of the kernel (R7, before any gate)

Read in the source and in the SPIR-V of all eight variants (host glslc, shaderc v2026.1, the cross image's
version): every `OpMemoryBarrier` is followed by `OpControlBarrier` with execution scope Subgroup (7 pairs in the
`t32x32` one-pass variants, 9 in `t16x64`).

| shared object | writer | readers | ordered by |
|---|---|---|---|
| `Psh` scores | `coopMatStore` of the whole subgroup (`qk_block`) | each lane, its own row segment | barrier pair after `qk_block` |
| `Psh` e | each lane, its own segment `e_idx + [0, SEG_V8)`: row `id % M`, segment `id / M`, one lane per slot | `coopMatLoad` (`av_block`) | barrier pair before `av_block`; another after it, before the next `qk_block` store |
| `Rsh` (row maximum per block, row sum) | each lane, slot `e_row * SEGS + e_seg`: one lane per slot | the lanes of the same row | barrier pair after the store, another after the reads |
| `Dsh` rescale divisors | each lane, `e_row * 4 + j`, `j = e_seg, e_seg + SEGS, ...`: disjoint per lane | `coopMatLoad` | barrier pair after the stores, another after the load; inside `if (subgroupAny(...))`, which is uniform over the subgroup |
| `Psh` divisors (end) | each lane, `e_row * P_STRIDE + j`, disjoint per lane | `coopMatLoad` | barrier pair after the stores |

No slot has two writers (no "every lane stores the reduced value"); no read of another lane's slot without an
execution barrier after its store; no write to a slot another lane may still be reading (a barrier pair closes
every read phase). Row ownership uses `gl_SubgroupInvocationID` only, never `gl_LocalInvocationID`. What the
kernel still assumes, as every cooperative-matrix kernel on this device does: a workgroup of 32 is exactly one
full subgroup. The copy passes use no shared memory; every invocation writes its own 4 x 4 block.
Not lockstep-dependent, but worth watching in the tiers: the one-pass form starts from a row maximum of -inf
(`exp(+inf)`, `0 / inf`); the `peaked` tier exercises the rescale path.

## Next step

When `parent` is built: deploy, `chain1` on the device (`s0-parent-verify` with the parent environment,
`s0n-noenv` with nothing selected). When `topic1` is built: A/A + baseline session `s1-aa` (parent build against
`topic1`, both with the parent environment), clock floor, `s2n-noenv` (hook control), then candidate 1 at kernel
level and its gate.

## Thresholds

`tools/thresholds.txt` (committed in `086651dd4`, before any measurement).

## Taken from the 4070 Ti fused port

Nothing yet: `origin/topic/4070ti-fused-port` does not exist at 19:45 UTC (`git fetch origin`).
