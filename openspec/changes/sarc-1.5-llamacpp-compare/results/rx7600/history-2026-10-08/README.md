# Radeon RX 7600: results, with the replaced runs and the deviations

Session `rx7600-1`, 2026-10-08 17:57 to 19:30 UTC, 14 arms (`arms.tsv`, `ARMS.md`) on 1B, 3B and 8B, after the
tuning campaign had finished. Follow-up session `rx7600-2`, 19:42 to 19:51 UTC, for the 3B cells that a foreign
build disturbed (below). Validity: the campaign's rule as `row.py` applies it, clock floor 2420 MHz, sampled
every 10 ms (`kit/hosts/rx7600/host.sh`); no foreign-busy ceiling (no per-client engine accounting on this card).
Vulkan only, user-space RADV Mesa 26.2.3 for every arm.

Files: `runs.csv` = `rx7600-1` with the build-overlapping runs marked (below), `runs-s2.csv` = `rx7600-2`,
`cells.csv` (the table's source; counts `clock_low`-only runs as the 780M does, see below), `cells-strict.csv`
(the rule as it stands), `checks.csv` (1B and 8B from `rx7600-1`, 3B from `rx7600-2`), `screen.csv`.

## Runs replaced: a foreign build during `rx7600-1`

A build of another job on the workstation (compiler and linker processes, `make -j16`) ran from 18:20:28 to
18:40:24 UTC, seen by a 1-second process watcher beside the session. The campaign's rule rejects runs taken
beside a host build; `row.py` does not see builds, so the runs whose sampled interval overlaps that window are
marked invalid in `runs.csv` with the reason `foreign_build` (row.py itself unchanged): 32 timed 3B runs, the
ten 3B text checks and three 8B `discard` runs (repetition 0 of `lc`, never counted). That left the six
ExecuTorch and four `lc` arms of 3B with one or two quiet valid runs each. They were measured again in
`rx7600-2` (same stage, settings and script, those ten arms only, interleaved as `session.sh` does, five valid
runs each, no build seen), and the 3B cells of those arms come from `rx7600-2` alone. The 3B `lb` arms ran
before the build (18:1x UTC) and are kept from `rx7600-1`. No run of 1B or 8B overlapped the build.

## 8B stock `4w`: workload-limited clock

All eight runs of stock `4w` on 8B ran at a median of 2323 to 2330 MHz, under the 2420 MHz floor, at the same
speed (470.6 to 473.1 tok/s, spread 0.5 %), so `row.py` rejects all of them (`clock_low`) and the session ends
`SESSION_INCOMPLETE` for this one cell. It is the longest prefill of the session (4.3 s) and the same pattern
the 780M showed: the clock is a property of that workload, not a disturbance. As on the 780M, `cells.csv` counts
those runs (`n_accepted` = 8, `clk_med_mhz` = 2326) and the cell is marked; `cells-strict.csv` leaves it empty.
Every other arm reached the floor in every run.

## Result (tok/s, median; llama.cpp = the higher of its two timers at its better setting)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 2709 | 3800 | 7847 | 7314 | 10503 | 10292 | 5882 | 5006 | 1.79 |
| 3B | 975 | 1387 | 3298 | 3084 | 3992 | 3961 | 2449 | 2039 | 1.63 |
| 8B | 472* | 730 | 1518 | 1404 | 1750 | 1742 | 1150 | 941 | 1.52 |

\* counted under the clock note above. Five valid runs in every other ExecuTorch and `lc` cell; `lb` = one
process of five repetitions.

llama.cpp, all eight arms (tok/s): 

| model | lc Q4_0 default | lc Q4_0 best | lb Q4_0 default | lb Q4_0 best | lc Q4_K_M default | lc Q4_K_M best | lb Q4_K_M default | lb Q4_K_M best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 5455 | 5618 | 5842 | 5882 | 4696 | 4709 | 5006 | 4966 |
| 3B | 2376 | 2278 | 2449 | 2284 | 2031 | 1907 | 2039 | 1917 |
| 8B | 1150 | 1120 | 95† | 54† | 941 | 914 | 94† | 54† |

- `best` (`-b 2048 -ub 1024 -fa on`, from the 1B screen) is 3 to 7 % slower than `default` on 3B and 8B (on 1B the two are within 3 %); the
  table takes the faster of the two per model, as the kit's rule says.
- † llama-bench on 8B reads 54 to 95 tok/s against 1150 by llama-completion on the same file and flags. Diagnostic runs
  after the session (one repetition each): `pp512` 1176 / 1253 (flash attention off / on), `pp1024` 1229, but
  `pp1536` 153, `pp2047` 96 and `pp2048` 519 / 96 (off / on), while the combined test `pg 2048,1` reads 1137.
  All layers and the KV cache are on the GPU (`-v`), and the GPU is busy at full power during the slow runs, so
  it is a slow GPU path that llama-bench's prompt-only test reaches on 8B above 1024 tokens on this card. The cause
  is UNVERIFIED. llama-completion is not affected, so llama.cpp's 8B number is the `lc` timer. 1B and 3B
  llama-bench agree with llama-completion within 7 %.
- Q4_K_M is 15 to 18 % slower than Q4_0 here.

Against the campaign's final session (same binaries, environment and `.pte`): tuned `4w` 10503 / 3992 / 1750
against 10503 / 3984 / 1747, SARC `4w` 7847 / 3298 / 1518 against 7847 / 3298 / 1517: within 0.2 %.

Text check (`checks.csv`): every arm continues the real-text prompt fluently. ExecuTorch arms agree with each
other on 1B and 3B. On 8B the two `8da4w` SARC arms (`sarc`, `tuned`) give a different later word from the
`4w` arms and stock. llama.cpp gives a different, equally plausible continuation; its tokenizer splits the check
prompt differently (1980 against 1972 tokens).

## Audit notes

- Thermal: the adapter keeps the campaign's `indep_throttle_status` beside each run (`<run>.clk.full`, not
  committed). Inside the ExecuTorch prefill windows, thermal bits appear in one counted run (3B stock `4w` r1 of
  `rx7600-2`, 2 samples). Without that run the cell's median moves by 0.07 %. For `lc` runs the window cannot be
  rebuilt after the fact, so it was checked over the whole process, where the bits appear around load. Not
  judged by `row.py`, as on the 780M.
- Medians of `cells.csv` and `cells-strict.csv` were recomputed from the runs files independently of
  `aggregate.py`: no difference.
- Other GPU users: none in any run; desktop idle (no DRM client on the card).

## Deviations from the kit

- **`.pte` files:** the campaign's exports at context 3072 (`*_embq_ctx3072.pte`, sizes in `ARMS.md`), not the
  2560-context shared exports of `MODELS.md`, which are not on this host. llama.cpp runs with `-c 2560` as fixed.
- **GGUF files regenerated** from the HuggingFace revisions of `MODELS.md` (`hf download`, safetensors, config
  and tokenizer only), F16 with `convert_hf_to_gguf.py` of `b11430`. Q4_0 with `b10229` and the August options
  `--pure --output-tensor-type q4_0 --token-embedding-type q4_0`, Q4_K_M with `kit/make-q4km.sh` (`b11430`,
  default quantization). Parameter counts are identical to `MODELS.md`. Every file is exactly 256 bytes
  smaller (Q4_0 703,205,280 / 1,815,607,648 / 4,525,773,312; Q4_K_M 807,690,144 / 2,019,373,408 /
  4,920,734,208): the header metadata of the newer converter; which key differs is not established.
- **Converter environment:** a venv with llama.cpp's `requirements-convert_hf_to_gguf.txt` (torch 2.11.0+cpu,
  transformers 4.57.6), installed from PyPI and download.pytorch.org.
- **Stock arm** built natively from an export of `985c1ceccc` plus the backport (podman does not run on this host,
  so `build-stock.sh` was not used), with the same compiler and shader compiler as the campaign's builds.
- **HIP not done:** ROCm is not installed on this host and `/dev/kfd` is not accessible to the user, so
  llama.cpp's HIP backend could not be built or run. Vulkan only.
- **Foreign build** during `rx7600-1`: runs replaced by `rx7600-2` as described above.
