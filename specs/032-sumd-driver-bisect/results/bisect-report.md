# SUMD Driver Bisect: 8da4w-Slower-Than-4w Regression

**Feature**: [032-sumd-driver-bisect](../spec.md) | **Status**: Complete — culprit identified

## Summary

On M41 (serial `00000a34cdd4abd3`), running the release/1.3 vanilla ExecuTorch runner (Llama 3.2
1B, 2048-token prefill, 1 rep) against Llama's `4w` and `8da4w` quantization modes, 8da4w should
be faster than 4w (int8 arithmetic vs int4-weight-only) — and is, on most SUMD driver builds
tested. But on some builds (including the M5 EVT1 known-good driver `f14c51b6f8` cross-flashed
onto M41, and every SUMD `main` commit from 2025-08-19 onward through the current 2026-03-31
snapshot) 8da4w is instead ~2.7-4x *slower* than 4w. This study `git bisect`ed SUMD's `main`
branch to find the exact commit that introduced this inversion.

**Culprit**: [`69e887275e26ae4a44e2d6e14bd3e600cec67ac8`](#culprit-commit) ("xgl,sc: fixup! Use
feature json to control WMMA and V_DOT usage" — disables the `dot4_i32_i8` int8-dot-product
instruction pattern), landed 2025-08-18/19. `dot4_i32_i8` is exactly the hardware path 8da4w's
int8×int8→int32 arithmetic would use; disabling it forces 8da4w onto a slower fallback while 4w
(no int8 dot products needed) is unaffected — consistent with every measurement in this study.

**Method**: the literal Nov-2024 range endpoint (and nearby commits) crashed on-device
(`VK_ERROR_DEVICE_LOST`) rather than giving a clean verdict — a genuine driver/hardware
incompatibility unrelated to the regression itself, not a script bug. Rather than resolving that
crash zone, this study found a confirmed crash-free good/bad bracket well inside the original
range (`b6487d67b7`/2025-04-11 good ↔ `f61822f069`/2025-09-19 bad) and bisected within that
354-commit bracket instead — 11 build+flash+measure cycles total, converging on a single
first-bad commit. Every tested commit's SHA, on-device driver identity (md5sum), and 4w/8da4w
prefill numbers are in the Trace table below.

## Trace

One row per commit tested, in the order tested. `verdict`: `good` = 8da4w prefill tok/s > 4w
prefill tok/s (strict comparison, no tie-break); `bad` = otherwise; `skip` = build/flash/crash
failure (reason given).

| # | bisect_role | commit_sha | commit_date | driver_version | 4w prefill tok/s | 8da4w prefill tok/s | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| 898709039d | endpoint-old | 898709039d173379d987ff4c9289cc5be7ee09ef | 2024-11-01 16:18:45 -0500 | 429292467bcac454623b2f3e71c343ab |  |  | skip | 4w run crashed (`vkQueueWaitIdle` returned -4 / VK_ERROR_DEVICE_LOST) — literal Nov-2024 calendar-boundary commit is untestable |
| 626c6bd367 | endpoint-old-probe | 626c6bd367f84fcb0984155899ea64d2a276fc99 | 2024-11-19 14:43:38 -0600 | e9b2a29e4a1d52517d68f3a31aa211a2 |  |  | skip | +100 commits forward; same VK_ERROR_DEVICE_LOST signature — crash zone extends at least this far |
| f61822f069 | endpoint-old-probe | f61822f0690d356b4751288a15f9258d5ff9b79e | 2025-09-19 13:27:16 -0500 | b5b79cfcf36147778c91ebbc3c989be3 | 610.25 | 220.928 | bad | range-midpoint-by-commit-count (line 1528/3046); clean run — confirms crash zone doesn't extend this far |
| b6487d67b7 | skip-adjacent-probe | b6487d67b762157ba2751f47ab2b8100ebb78f07 | 2025-04-11 13:58:23 -0500 | d1b0514f193c9e5cb84933fca3f253ba | 609.342 | 805.665 | good | binary search between crash zone end and the clean-bad midpoint (line 814/3046); **KEY FINDING** — clean run, genuine good verdict. Establishes a confirmed good/bad bracket (this commit ↔ f61822f069) for the real bisect without needing to resolve the Nov2024–Apr2025 crash zone further |
<!-- further rows appended here (>> to this file, by bisect-test.sh itself) as each commit is
     tested. The "## Culprit Commit" section is intentionally NOT present yet -- append_row
     blindly appends to EOF, so it must stay absent until T020 (after git bisect converges), or
     new rows would land after it instead of in the table. Do not add that section back until
     T020, and do not manually re-append rows here -- the script already does it; hand-editing
     caused a duplicate-row bug once already (fixed 2026-07-16). -->
| ec3958eae5 | endpoint-new | ec3958eae55ec3826d829d2a1149ddb4765b8af4 | 2026-03-31 22:00:45 -0500 | a3e454c8f8cb83f06a5f292c058347a5 | 606.635 | 148.837 | bad |  |
| ab6cb4d071 | bisect-step | ab6cb4d071cb37d3f175a391df86241c83b8ba04 | 2025-07-09 18:17:48 -0500 | 3d3ea019bd26083644aea165a06e613d | 605.917 | 806.299 | good |  |
| 0d498f7990 | bisect-step | 0d498f7990f2824f6f3c660032ecc884adfbaf06 | 2025-08-19 00:53:51 -0500 | 2400f5510dfee893c253792d563cddfe | 604.308 | 221.693 | bad |  |
| 77215300a3 | bisect-step | 77215300a32df33b5e901b228f534f01c1b71934 | 2025-07-28 11:26:01 -0500 | e114e38f575b0bc8221ce3282f872360 | 604.843 | 793.798 | good |  |
| 9873412d20 | bisect-step | 9873412d20398611d8585c190cd26a9098f41298 | 2025-08-08 14:08:55 -0500 | f64098f81de2646e310790cbac714bd7 | 606.096 | 794.723 | good |  |
| 212ddce456 | bisect-step | 212ddce456487b6d89d14cedc8450e2d67b6d46f | 2025-08-13 06:21:36 -0500 | 5482cdf0a696f5fc183fb41e6299029f | 604.665 | 792.877 | good |  |
| e3901f2db8 | bisect-step | e3901f2db8f3b928a24899fd1d1bcf53e72e1c08 | 2025-08-15 14:08:29 -0500 | 470bfa6a3c2acaaa3a8c97a04545699f | 605.201 | 792.263 | good |  |
| 635f83ba1a | bisect-step | 635f83ba1a1619c29d00f37730cb2c0000fac25f | 2025-08-18 16:46:38 -0500 | 8b4c8c978ae10d83c8bf59e085adb0b3 | 602.353 | 810.447 | good |  |
| 38e9a781d6 | bisect-step | 38e9a781d67a8fa2b63308e2f06f12e916dd97d6 | 2025-08-18 20:48:42 -0700 | 2c5ac6a3b69c12cfe14e653576f0cfc9 | 601.645 | 808.847 | good |  |
| 69e887275e | bisect-step | 69e887275e26ae4a44e2d6e14bd3e600cec67ac8 | 2025-08-18 23:57:20 -0500 | 062ab055f37a8f57f7fbf5351a60111e | 600.939 | 221.717 | bad |  |
| 0b814fa6d3 | bisect-step | 0b814fa6d3d204817d6e78bdd3b517fb2b21e6dc | 2025-08-18 23:33:23 -0500 | b460447da23f94d8fc970d24f883a67e | 598.655 | 811.41 | good |  |

## Culprit Commit

`git bisect` converged on exactly one first-bad commit (`git bisect log` archived at
`bisect-log-raw.txt`), immediately adjacent to the last-good commit found (24 minutes apart by
author-commit timestamp):

| | SHA | Author | Date | Subject | Driver identity (md5) |
|---|---|---|---|---|---|
| **Last good** | `0b814fa6d3d204817d6e78bdd3b517fb2b21e6dc` | — | 2025-08-18 23:33:23 -0500 | vkgc,sc: Add support for specific bit-wise operations | `b460447da23f94d8fc970d24f883a67e` |
| **First bad** | `69e887275e26ae4a44e2d6e14bd3e600cec67ac8` | Prabhakar Pal \<prabhakar.p1@samsung.com\> | 2025-08-18 23:57:20 -0500 | xgl,sc: fixup! Use feature json to control WMMA and V_DOT usage | `062ab055f37a8f57f7fbf5351a60111e` |

**Commit message body** (metadata only, no driver source was read to produce this report — see
`sumd/CLAUDE.md` Rule 0):

```
Disable patterns that use dot2_f32_f16 and dot4_i32_i8 instructions

Jira-Fixed: GFXSW-61045
Tests-Passing: build
```

**Files touched** (paths only, from `git bisect good`'s own diff-stat output — not opened or
read): `mgfxCompileHelper.cpp`, `SCInterface.h`, `SCShadersSi.h`, `Serialize.cpp`, `Patterns.pdl`,
`SCExpander.cpp`, `SCShaderInfo.{cpp,hpp}`, `SCTargetInfo.hpp`, `SCTargetInfoGFX405.{cpp,hpp}`,
`SHDisassembler.cpp`, `spvTool.cpp` — 13 files, +120/-73 lines.

**Why this explains the regression**: `dot4_i32_i8` is the int8×int8→int32 dot-product
instruction — exactly the hardware acceleration path 8da4w's int8 arithmetic depends on. This
commit's own message states it disables shader-compiler patterns that emit that instruction (and
its fp16 counterpart `dot2_f32_f16`). Disabling it forces 8da4w onto a slower fallback path while
4w — which never needed int8 dot products — is unaffected. This is consistent with every
measurement in this study: 4w prefill tok/s stayed flat (~598-611) across all 15 tested commits
regardless of verdict, while 8da4w swung between ~793-811 (good side) and ~149-222 (bad side).
