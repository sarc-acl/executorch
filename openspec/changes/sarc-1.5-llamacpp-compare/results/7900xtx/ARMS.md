# Radeon RX 7900 XTX: arms of the comparison (written before the timed session, 2026-10-10)

Device: Radeon RX 7900 XTX (RDNA3, Navi31, 24 GB) in `<gpu-host>`, which runs staged binaries only; the session runs on
that host. **Driver for every arm, ExecuTorch and llama.cpp alike: AMDVLK 2025.Q2.1 (LLPC), Vulkan 1.4.313**, selected with
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json` (the tuning campaign's driver). RADV is not used in this set. The tuning
campaign (`sarc-1.5-7900xtx-prefill-refine`) is finished. llama.cpp reports `fp16: 1`, `int dot: 1`, `matrix cores:
KHR_coopmat`, warp size 64 on this driver.

Validity: the campaign's rules as `row.py` applies them, with the adapter `kit/hosts/7900xtx/host.sh`: rc 0; exactly 2048
prompt tokens; no other GPU user before, during or after (holders of the card's DRM nodes, GPU runner programs; the idle
`ollama serve` of the host is not a runner); at least five clock samples in the prefill window (sampled every 5 ms); median
clock in the window at least 2670 MHz (the campaign's calibrated floor); no thermal throttle reason (`gpu_metrics`
`indep_throttle_status` bits 32 to 47 masked by `0xffef`: bit 36 is recorded and does not reject, owner decision of the
campaign); the card's `gpu_busy_percent` at most 5 % in ten samples before each run and no host build running (both waited
for, never forced). No per-client engine accounting on this card: no foreign-busy ceiling in `row.py`. The session starts
with the card's edge and junction temperatures at most 48 C, as the campaign's gate did.

## ExecuTorch Vulkan (each with `4w` and `8da4w`)

| arm | build | selection (`env` of the arm, besides the ICD) |
|---|---|---|
| `stock` | upstream `release/1.5` at `985c1ceccc` plus the compile-only backport (`stock-backport-03f41d2031.patch` of `sarc-1.5-e2e-benchmark`), an export of the commit and its pinned submodules built natively (x86_64) with the same compiler and shader compiler as the campaign's builds (podman does not run on the build workstation, so `build-stock.sh` was not used) | none |
| `sarc` | `90fe4d013`, the campaign's parent (`dev/1.5` at the start of the campaign), the campaign's own binary of its final session | `ET_VK_SARC_UNVERIFIED=1`, as the campaign measured its parent |
| `tuned` | `5950764fa`, the campaign's final build (`c10`, the code of the campaign's final head): an unmerged development branch, the campaign's own binary of its final session | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5` (the campaign's final configuration) |

sha256 of `llama_main`: stock `5869094d…`, sarc `ed0c2f21…`, tuned `e9d03689…` (sarc and tuned identical to the campaign's final
session `final3`; the stock binary is the one of the RX 7600 session).

`.pte` files: the campaign's exports, `*_embq_ctx3072.pte` (context 3072, 4-bit embeddings, group 128; the same files as on
the RX 7600), not the 2560-context shared exports of `MODELS.md`. Sizes: 1B 789,135,872 / 827,832,192, 3B 1,886,556,160 /
1,987,087,360, 8B 4,173,751,424 / 4,408,429,824 bytes (`4w` / `8da4w`). llama.cpp runs with `-c 2560` as the kit fixes it.

## llama.cpp `b11430`, Vulkan

`q4_0` and `q4_k_m`, each as `default` and `best`, by both timers (`lc`, `lb`): 8 arms. Exact lines: `arms.tsv`
(`llama-completion` and `llama-bench` are the `b11430` binaries). Built on the workstation with `-DGGML_VULKAN=ON
-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON` (the kit's native build crashes with an illegal
instruction on the GPU host's CPU, which has no AVX-512; the Vulkan backend and its shaders are the same), copied to the host.

GGUF files as on the RX 7600: regenerated from the HuggingFace revisions of `MODELS.md` (`Llama-3.2-1B@4e20de36`,
`Llama-3.2-3B@13afe512`, `Llama-3.1-8B@d04e592b`), F16 with `convert_hf_to_gguf.py` of `b11430`; Q4_0 with `b10229` and the
August options (`--pure --output-tensor-type q4_0 --token-embedding-type q4_0`); Q4_K_M with `kit/make-q4km.sh` (`b11430`,
default quantization). Every file is 256 bytes smaller than in `MODELS.md` (header metadata of the newer converter).

## Settings screen (fixes `best`)

`screen.csv`, session `screen1`, 2026-10-10 00:47 to 01:00 UTC, on 1B and 3B, for both quantizations: `llama-bench`
(`lb`, one process of five repetitions each) at the default setting (`-ub 512`, flash attention auto) and at `-b 2048` with
`-ub {512, 1024, 2048}` x flash attention {on, off}; `llama-completion` default (`lc`, three runs, repetition 0 discarded); the
tuned `4w` arm as an anchor. Result (tok/s): Q4_0 on 1B 15556 default; 15425 / 13220 (`-ub 512` on / off); **16297** /
12794 (`-ub 1024`); 14905 / 10825 (`-ub 2048`); on 3B 6614 default; 6532 / 5697; **6849** / 5342; 6399 / 4655. Q4_K_M on 1B
12345 default; 12932 / 11056; **13004** / 10988; 12569 / 9376; on 3B 5385 default; 5389 / 4770; **5546** / 4486; 5224 / 3990.
The best setting is the same for both quantizations and both models: `best` = `-b 2048 -ub 1024 -fa on`. Flash attention
off is 11 to 27 % slower than on at the same `-ub`; the three flash-attention-on settings are within 3 to 9 % of each other
(per model and quantization); `best` is 3 to 5 % above the default. The
kit's list was tried in full; no other llama.cpp option was screened.

Known before the timed session: `llama-completion` has no full-prompt warm-up in its process, so its prompt window includes
the card's clock ramp-up; in `screen1` every Q4_0 `lc` run and most 1B Q4_K_M `lc` runs had a median clock of 2500 to 2650 MHz in
the window against the 2670 MHz floor and are marked `clock_low` by `row.py` (`row.py` itself is not changed). The timed
session records them as they are; how they are counted is stated in its README.
