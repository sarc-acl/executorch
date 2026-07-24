# specs/039: ETDump prefill attribution — make e2e the loss function

## What was wired

`examples/models/llama/main.cpp` already handles `--etdump_path` behind
`ET_EVENT_TRACER_ENABLED`, but the standalone `llama_main` CMake project
(`cmake-out-vk/examples/models/llama`) is a separate `find_package(executorch)`
build that never consumed `EXECUTORCH_ENABLE_EVENT_TRACER`, so the define was
absent and `--etdump_path` silently no-op'd. specs/039 adds a gated block to
`examples/models/llama/CMakeLists.txt`:

    if(EXECUTORCH_ENABLE_EVENT_TRACER)
      add_definitions(-DET_EVENT_TRACER_ENABLED)
      list(APPEND link_libraries etdump flatccrt)
    endif()

(`etdump` / `flatccrt` are imported targets from the installed package.) Default
OFF, so the shipping binary is unchanged. Reconfigure with
`-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` to capture.

## Attribution (1B 8da4w buffer, one 2048-token prefill, 4x2 kernel)

Aggregated with `executorch.devtools.Inspector`, summing `perf_data.raw`
(milliseconds) per leaf kernel, framework envelope events excluded. Leaf-kernel
sum 1043 ms vs wall-clock prefill 1054 ms => prefill is GPU-bound, ~1% CPU
bubble between dispatches.

| category                  | ms     | %      | dispatches | tok/s if −10% |
|---------------------------|--------|--------|-----------|---------------|
| dq8ca linear (int8 WMMA)  | 557.6  | 53.4%  | 112 (16L×7) | 2074        |
| SDPA                      | 345.6  | 33.1%  | 80        | 2030          |
| quantize_and_pack         | 110.4  | 10.6%  | 309       | 1984          |
| dq8ca aux (input quant)   |  27.5  |  2.6%  | 114       | 1968          |
| other                     |   2.1  |  0.2%  | 4         | —             |

Baseline this run: ~1963 tok/s (GPU-busy) / ~1943 wall.

## Consequences for prioritization

- **The dq8ca linear is the #1 kernel at 53%.** Tasks 1–3 targeted the right
  thing; the specs/038 occupancy retile (0.94 → 1.00x microbench) moved e2e
  ~1843 → ~1939 tok/s. From here every 10% of linear kernel time ≈ +111 tok/s,
  so clearing 1957 from ~1939 needs the linear only ~1.5% faster — but it is
  already at microbench parity with the tiled dot4 path, so that last sliver is
  in the noise band, not a clear structural win.
- **SDPA is 33%** and healthy (2.84x vs tiled per the mission); out of scope
  here but the second-largest pool.
- **quantize_and_pack is 11%** — the per-token int8 activation quantization the
  8da4w path pays and the 4w path does not. This is the structural reason 8da4w
  e2e trails 4w e2e, and the largest non-linear lever left; reducing/fusing it
  would help every 8da4w model but is a separate workstream from the WMMA kernel.
