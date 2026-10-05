# Model files

## ExecuTorch (the shared exports of `sarc-1.5-e2e-benchmark` and the tuning campaigns)

Read from each file with the ExecuTorch runtime on 2026-10-05 (constant methods of the program):
`get_max_context_len` = `get_max_seq_len` = **2560**, `use_kv_cache` = true, `use_sdpa_with_kv_cache` = true,
`enable_dynamic_shape` = true, `get_bos_id` = 128000, in all six files.

| file | bytes | parameters stored | bits per weight |
|---|---:|---:|---:|
| `llama3_2-1b_vulkan_4w.pte` | 788,490,112 | 1,498,482,688 | 4.210 |
| `llama3_2-3b_vulkan_4w.pte` | 1,885,262,208 | 3,606,752,256 | 4.182 |
| `llama3_1-8b_vulkan_4w.pte` | 4,172,459,904 | 8,030,261,248 | 4.157 |
| `llama3_2-1b_vulkan_8da4w.pte` | 827,179,392 | 1,498,482,688 | 4.416 |
| `llama3_2-3b_vulkan_8da4w.pte` | 1,985,780,864 | 3,606,752,256 | 4.405 |
| `llama3_1-8b_vulkan_8da4w.pte` | 4,407,123,968 | 8,030,261,248 | 4.391 |

- Parameter counts are those of the August notes for the same architecture (ExecuTorch stores an untied output
  layer for 1B and 3B). Bits per weight = file bytes x 8 / parameters, so it includes the program itself.
- **Group size 128 and 4-bit embeddings (group 32) are not readable from the program metadata.** They are
  inferred: the workspace notes give the export command (`-G 128 -E 4,32`) as reconstructed, and each `4w`
  file is within 0.1 % of the size of the August export that was made with exactly those settings
  (789,135,872 / 1,886,556,160 / 4,173,751,424 bytes at context 3072). A file with fp16 embeddings would be
  several hundred megabytes larger. Treat as inferred, not verified.
- Compute precision fp16 (Vulkan delegate), as in the workspace notes.

Consequence for llama.cpp: context length **2560** (`-c 2560`), not the 3072 of the August runs.

## llama.cpp (GGUF)

All from the same upstream revisions as the August ExecuTorch exports: `Llama-3.2-1B@4e20de36`,
`Llama-3.2-3B@13afe512`, `Llama-3.1-8B@d04e592b`. Whether the shared ExecuTorch exports above were made from
the same checkpoints is not recorded with them; the checkpoint files are Meta's originals of the same models.

| file | bytes | parameters | bits per weight | made by |
|---|---:|---:|---:|---|
| `llama3_2_1b_q4_0.gguf` | 703,205,536 | 1,235,814,432 | 4.552 | August, llama.cpp b10229, note beside the file |
| `llama3_2_3b_q4_0.gguf` | 1,815,607,904 | 3,212,749,888 | 4.521 | same |
| `llama3_1_8b_q4_0.gguf` | 4,525,773,568 | 8,030,261,312 | 4.509 | same |
| `llama3_2_1b_q4_k_m.gguf` | 807,690,400 | 1,235,814,432 | 5.229 | this change, llama.cpp b11430, default quantization (llama-quantize reports 5.18 for the tensors) |
| `llama3_2_3b_q4_k_m.gguf` | 2,019,373,664 | 3,212,749,888 | 5.028 | same (5.01) |
| `llama3_1_8b_q4_k_m.gguf` | 4,920,734,464 | 8,030,261,312 | 4.902 | same (4.89) |

The F16 intermediates (2.5 / 6.4 / 16.1 GB) are kept beside them; sha256 of every copied file against its
source is in `SHA256.source` / `SHA256.nas` in the model directory.
