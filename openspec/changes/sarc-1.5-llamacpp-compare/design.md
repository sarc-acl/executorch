# Design

## Context

See `proposal.md` for the motivation. What shapes the approach:

- The ExecuTorch side is already measured with a fixed protocol (`sarc-1.5-e2e-benchmark/kit/`, and the
  per-device session and guard tools of the `sarc-1.5-<gpu>-prefill-refine` changes on their topic branches).
  Its stock and SARC numbers are in `sarc-1.5-e2e-benchmark/results/cells.csv`.
- Earlier preparation exists outside the repository, from August 2026: GGUF files of the three models made
  from the same upstream revisions as the ExecuTorch exports, as F16 and as Q4_0 with the embedding forced to
  4 bit, each with a note of its exact conversion. That work used llama.cpp build b10229. Its scripts and any
  results have not been read yet; task group 1 does that first.
- The two runtimes cannot be made identical. Known differences, to be disclosed, not hidden:
  - Q4_0 quantizes in blocks of 32 with a round-to-nearest scale; ExecuTorch `4w` uses groups of 128. The
    August note measures 4.55 against 4.21 bits per weight on the 1B model.
  - llama.cpp has no counterpart of `8da4w` (8-bit dynamic activations). The `8da4w` cells are compared with
    the same llama.cpp numbers as `4w`.
  - llama.cpp splits a prompt into batches of 512 by default; ExecuTorch runs 2048 tokens in one forward pass.
  - The ExecuTorch Vulkan models compute in fp16; ExecuTorch CUDA computes in bf16.
- ExecuTorch's CUDA backend (AOTInductor based, upstream `main`): runs tile-packed 4-bit weights natively;
  cannot export `8da4w`; upstream caps the prompt of the 3B and 8B models at 511 tokens until an export guard
  is removed; is not built for Jetson upstream. Its Llama export goes through a different exporter than the
  Vulkan one.
- Four of the five devices are still running tuning campaigns. A device is exclusive while it measures, and a
  build on the same host disturbs a timed session.

## Goals / Non-Goals

**Goals:**

- One table per device that places stock, SARC and tuned ExecuTorch Vulkan beside llama.cpp, with everything
  a reader needs to repeat it.
- A protocol fixed on one device (Arc B580) before it is applied to the others.
- Numbers in the owner's hands this week, device by device.

**Non-Goals:**

- Making llama.cpp or the CUDA backend faster, or explaining in depth why one side wins.
- Decode speed (a later change).
- Porting the CUDA backend to Jetson, or adding `8da4w` to it.
- Any change to a tuning campaign's branch, rules or schedule.

## Decisions

**D1. Reuse the August model files; add Q4_K_M from the same F16 files.**
Q4_0 stays as converted (uniform 4-bit, embedding included). Q4_K_M is produced with llama.cpp's default
quantization, no forcing, because its role is "what users actually run". Alternative rejected: downloading
community GGUF files, whose source revision and conversion are not ours to vouch for.

**D2. One recent llama.cpp commit for every device, not b10229.**
The Vulkan backend of llama.cpp changes quickly; a two-month-old build invites the objection that a slow
version was chosen. The commit is picked on the day the pilot starts and never moved afterwards. The existing
GGUF files are loaded with it and their output checked before any timing. Builds already present on some
hosts are left alone; this work builds into its own directory.

**D3. Two measuring tools for llama.cpp; the real prompt is the headline.**
(a) The fixed prompt file through llama.cpp's completion binary, one process per run, its own warm-up on,
one generated token, time read from its prompt-evaluation report. (b) `llama-bench` with a 2048-token prompt,
which uses synthetic tokens but is what others will run. (a) is the reported number; (b) is shown beside it,
and a disagreement beyond a threshold fixed in the pilot is investigated before anything is reported.

**D4. Three settings tiers.**
`default`: only full GPU offload. `aligned`: batch and micro-batch 2048. `best`: `aligned` plus whichever
documented performance options help on that backend and device (flash attention first), chosen by a recorded
screen on the 1B model and then frozen. Context length equals the ExecuTorch model's.

**D5. ExecuTorch arms reuse existing, recorded builds where they exist.**
`stock`: upstream `release/1.5`, built as in `sarc-1.5-e2e-benchmark`. `sarc`: the pristine parent build of the
device's tuning campaign, whose baseline was checked against `cells.csv` there. `tuned`: the campaign's latest
gate-accepted profile at the time of measuring, with its commit; re-measured later only if a later profile is
accepted. All three are measured again in the same session as llama.cpp, interleaved; `cells.csv` is a
cross-check, not a source.

**D6. Session protocol is the campaigns' protocol.**
Device lock, foreign-GPU-process guard, clock and temperature sampling, five valid runs per cell, median,
interleaved arms, validity rules calibrated per device from an A/A session. The kit here wraps the existing
tools rather than re-implementing them; llama.cpp arms are added as further arms of the same session script.

**D7. ExecuTorch CUDA arm is "upstream that can run the workload".**
Upstream `main` plus only the export-guard removal, `4w`, bf16, on the RTX 4070 Ti SUPER. The 8B model is
exported again with that fix. The owner's own CUDA kernel work is not an arm of this comparison. `8da4w` on
CUDA and the whole CUDA arm on Jetson read "not supported", with the reason.

**D8. SYCL in a container.**
The oneAPI toolchain is not installed on any host. The SYCL build and its runs happen in a container with the
GPU passed through, the image kept on shared storage. If the container cannot reach the GPU on a host, the
SYCL cells of that device read "not run" with the reason; the comparison does not wait for it.

**D9. Order and device time.**
Arc B580 now (idle; it is the owner's desktop GPU, the owner has agreed). RTX 4070 Ti SUPER when its campaign
has pushed. Arc Pro B70, Radeon 780M and Jetson Orin Nano when their campaigns end or the owner pauses one.
Nothing of this change runs on a device a campaign is using, including builds on that host.

**D10. Where things live.**
Kit and aggregated CSVs in this change directory. Raw logs, builds and model files outside the repository on
shared storage. The per-device tables and charts go to the owner as a private page and CSV; nothing is added
to `sarc-1.5-e2e-benchmark` or any published report by this change.

**D11. Executed directly, not as an agent campaign.**
The work is a fixed protocol with no search loop. Device work is done one device at a time; every reported
number is recomputed from the raw run files before it is shown.

## Risks / Trade-offs

- [The quantizations are not the same format] → Report bits per weight and group size beside every row; show
  Q4_0 and Q4_K_M; never call the comparison "identical configuration".
- [llama.cpp's best settings are found by us, so "best" may be less than an expert would reach] → Record the
  screen, name the options tried, and show the default tier too.
- [The pinned llama.cpp commit has a Vulkan regression on one device] → The August build b10229 is a known
  working version; if the pinned commit fails or is implausibly slow on a device, measure b10229 there as an
  extra arm and say so.
- [The 8B model does not fit on the Orin or the 780M under llama.cpp with a 2048-token batch] → The cell reads
  "does not fit" with the measured memory; no smaller batch is substituted under the `aligned` name.
- [Measuring on the desktop GPU while the desktop is in use] → The foreign-process guard already rejects such
  runs; the pilot is run when the owner is not using the machine.
- [A tuned arm moves after it is measured] → The commit is recorded; a later accepted profile is re-measured
  for that arm only.
- [Timer resolution on fast cards] → The 1B prefill is about 70 to 150 ms; report the resolution of each
  runtime's timer with the result, as the campaigns do.

## Open Questions

- The context length of the ExecuTorch models actually measured (2560 in the shared exports; the August note
  names a 3072 export). Resolved in task 1.3 by reading the model files; llama.cpp follows whatever it is.
- Whether the August work already contains a token-exact prompt for the llama.cpp tokenizer and how it treats
  the beginning-of-sequence token. Resolved in task 1.1.
