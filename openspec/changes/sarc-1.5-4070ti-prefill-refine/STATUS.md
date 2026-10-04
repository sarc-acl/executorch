# STATUS: sarc-1.5-4070ti-prefill-refine

**2026-10-04, gpu-dev-4004. BLOCKED before the first measurement. Nothing has been built or measured.**

## Blocking

The agent's shell on this host is confined ("fence", `TMPDIR=/tmp/hmz-fence-*`): the only writable places are
the working copy and that scratch directory. Denied with `Permission denied`, although owned by `doremy`:

| path | needed for |
|---|---|
| `~/hmz-sarc-4070ti/.artifacts/` | every build, log, ETDump and stage directory (they may not go into the working copy) |
| `~/.cache/gpu-lab/lock-81a511a2-…` (open for append) | the gpu-lab lock; the unmodified `sarc/tools/verify.sh` opens it with `>>` and exits 75 otherwise, so the gate cannot run |
| `~/.cache`, `~/.docker`, `/tmp` | igpu-roofline results; docker client config (worked around with `DOCKER_CONFIG` in the scratch directory) |

The confinement was not worked around: writing into `.artifacts` through the docker daemon, or taking the lock
on a read-only descriptor, would defeat a restriction somebody set on purpose. **Needed from the owner:** allow
writes to `~/hmz-sarc-4070ti/.artifacts/` and to the lock file `~/.cache/gpu-lab/lock-81a511a2-de7e-c3c8-f641-3562c315ffa7`
(and a results directory for igpu-roofline, e.g. under `.artifacts/`), then restart the campaign. Everything
below is ready for that.

## Done (no GPU job was run)

- Host check: RTX 4070 Ti SUPER, driver 615.71.09, idle 45 C, P8, 210 MHz (max 3120 MHz), no GPU process, card
  answering `nvidia-smi`. The six `.pte` files and tokenizers are present. docker works without sudo.
- Build image `localhost/et-vk-build:rocky10` built with docker from `tools/Containerfile`: shaderc v2023.8,
  GCC 14.3.1, the same toolchain as the 780M campaign.
- `tools/` copied from the 780M change and adapted (originals untouched): `common.sh` (paths, lock UUID,
  `nvidia-smi` temperature, GPU-process check, stop-on-`nvidia-smi`-failure for the Xid 79 rule),
  `podman-shim.sh` (docker shim for `build.sh`), `e2e5.sh` (20 ms `nvidia-smi` sampling of clock, busy, power,
  temperature; the sampler and the validity parser were tested on idle and synthetic data only), `gate.sh`,
  `gate_sdpa.sh` (tiers `extended` and `full`, 12 passes each, `pairing=ok` counted), `screen.sh` (resumable),
  `stage.sh`, `trace.sh`, `collect.sh`, `roof_util.py` (roofs are arguments, no stored values).
  None of these scripts has run end to end.
- `sarc/tools/check.sh --no-build`: PASS (31 rows, 109 candidates).

## Findings from reading the code (not measured)

- **SDPA cannot be reached from the dev zone on this device as the tree stands.** The profile mechanism only
  replaces a table choice that exists, and `sdpa_qk_spec_vars`, `sdpa_av_spec_vars` and
  `sdpa_softmax_shader_name` (`impl/sarc/SdpaCoopmat.cpp`) enable the SARC path only when
  `device_has_active_rows()` finds an SDPA row for the device in the release tables. `table_nvidia.cpp` has none.
  Two ways, for the owner to choose:
  1. smallest release-zone hook: two `kUnverified` rows (`kSdpaQk`, `kSdpaAv`) for `"4070 ti super"` in
     `table_nvidia.cpp`, measured through a local patch that is not committed, with `ET_VK_SARC_UNVERIFIED=1`;
  2. no release-zone edit: `impl/sarc_dev/Overrides.cpp` calls the public `sarc::register_rows()` with the same
     two `kUnverified` rows inside a `4070ti` block. Not tried; `test_sarc_select` row counts with the dev zone
     linked would change.
- Subgroup-32 SDPA kernels already exist as sweep variants (QK^T `sweep_t128x64k32g42s32[nf]`,
  `pk_t128x64k32g42s32nf`; attn*V `sweep_t64x64k32g42s32`, `ml_*s32`), all MMA 16x16x16 fp16, a shape this
  device exposes. Whether they run correctly or fast here is unknown.

## Open before the baseline

- CLKMIN for the "normal clock" rule is not set (`--clkmin 0` = record only); choose it from the baseline
  session's samples.
- `verify.sh` takes its unaligned prompt from `r*.txt` in the stage directory; the 780M campaign's `r1304.txt`
  is not in the repository. One has to be supplied or made (`tools/r1304.txt`).
- `trace.sh` needs a python with the ExecuTorch devtools (`TRACE_PY`); none was looked for yet.
- igpu-roofline is on the host only as a copy under `~/.cache/igpu-roofline/fleet-fast-20260926/`; its results
  directory is not writable either. No roof was measured.
- 1B timer quantisation (1 ms timer, about 100 ms prefill): to be reported with ETDump dispatch time alongside.

## Next step once unblocked

`tools/build-both.sh parent` at `6a7cc8cc6`, baseline of the six cells against `cells.csv`
(4w 19692 / 8752 / 4491, 8da4w 20898 / 9660 / 5032 tok/s), then the A/A session, the per-op ETDump breakdown
and the roofline `fast` run.

## Per-cell numbers against the parent

None.
