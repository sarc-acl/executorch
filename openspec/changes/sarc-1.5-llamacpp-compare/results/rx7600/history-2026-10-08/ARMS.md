# Radeon RX 7600: arms of the comparison (written before the timed session, 2026-10-08)

Device: Radeon RX 7600 (RADV NAVI33, 8 GB), the workstation's own card, user-space RADV Mesa 26.2.3 (`31e9a6b2e9`)
through `VK_ICD_FILENAMES` for every arm, ExecuTorch and llama.cpp alike. The tuning campaign
(`sarc-1.5-rx7600-prefill-refine`) is finished. Validity: the campaign's rule as `row.py` applies it (median clock
in the prefill window at least 2420 MHz, the campaign's calibrated floor; at least five clock samples; exactly 2048
prompt tokens; no other GPU user), sampled every 10 ms from the card's hwmon by `kit/hosts/rx7600/host.sh`. The
card has no per-client engine accounting, so there is no foreign-busy ceiling: the session runs while no build
runs on the workstation. Models, prompt and context as on the other devices, except the `.pte` files (below).

## ExecuTorch Vulkan (each with `4w` and `8da4w`)

| arm | build | selection (`env` of the arm, besides the ICD) |
|---|---|---|
| `stock` | upstream `release/1.5` at `985c1ceccc` plus the compile-only backport (`stock-backport-03f41d2031.patch` of `sarc-1.5-e2e-benchmark`), an export of the commit and its pinned submodules built natively with the campaign's toolchain (podman does not run on this host, so `build-stock.sh` was not used) | none |
| `sarc` | `f5f1bf10c`, the campaign's parent (`dev/1.5` at the start of the campaign), the campaign's own binary | `ET_VK_SARC_UNVERIFIED=1`, as the campaign measured its parent |
| `tuned` | `18cc0d53a`, the campaign's final build, the campaign's own binary: an unmerged development branch | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2` (the campaign's final configuration) |

sha256 of `llama_main`: stock `5869094d…`, sarc `cfb5fb8c…`, tuned `ca2d39be…` (sarc and tuned identical to the
campaign's final session).

`.pte` files: the campaign's exports, `*_embq_ctx3072.pte` (context 3072, 4-bit embeddings, group 128), not the
2560-context shared exports of `MODELS.md`, which are not on this host. llama.cpp runs with `-c 2560` as the kit
fixes it. Sizes: 1B 789,135,872 / 827,832,192, 3B 1,886,556,160 / 1,987,087,360, 8B 4,173,751,424 /
4,408,429,824 bytes (`4w` / `8da4w`).

## llama.cpp `b11430`, Vulkan, built on the host with `-DGGML_VULKAN=ON -DGGML_NATIVE=ON`

`q4_0` and `q4_k_m`, each as `default` and `best`, by both timers (`lc`, `lb`): 8 arms. Exact lines: `arms.tsv`
(`llama-completion` and `llama-bench` are the `b11430` binaries). llama.cpp reports `int dot: 1` and
`matrix cores: KHR_coopmat` on this device.

GGUF files regenerated from HuggingFace (`Llama-3.2-1B@4e20de36`, `Llama-3.2-3B@13afe512`, `Llama-3.1-8B@d04e592b`),
F16 with `convert_hf_to_gguf.py` of `b11430`; Q4_0 with `b10229` and the August options (`--pure
--output-tensor-type q4_0 --token-embedding-type q4_0`); Q4_K_M with `kit/make-q4km.sh` (`b11430`, default).
Parameter counts equal `MODELS.md`; every file is 256 bytes smaller than there (header metadata of the newer
converter; tensor data the same size).

Screen that fixed `best` (1B, `q4_0`, llama-bench tok/s, session `screen2`; `screen.csv`): default (`-ub 512`,
flash attention auto) 5839; `-ub 512` on 5841, off 4247; `-ub 1024` on **5879**, off 4265; `-ub 2048` on 5791,
off 3946. `best` = `-b 2048 -ub 1024 -fa on`; the flash-attention-on settings are within 1.5 % of each other,
flash attention off is 27 to 33 % slower. `lc` default 5478 (3 runs), tuned `4w` 10503 (3 runs, the campaign's
final-session median is 10503). A first screen (`screen1`) gave the same figures within 0.2 % but overlapped a
short build of another job on the host for its last runs, so it was redone and is not used.
