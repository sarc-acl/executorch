# Partial Intel input repair

The owner instruction of 2026-10-15 03:57 UTC authorizes revised inputs with
preserved weights, quantization, context, tokenizer and prompts. This preparation
uses the original six inputs and pinned base `6178971845`. It does not establish
six conforming models or permit timing.

`prepare_rope_inputs.py` changes exactly two existing VkTensor memory-layout
bytes per model: the two `[5120,32]` or `[5120,64]` RoPE prepack outputs become
height-packed instead of width-packed. Their dimensions, dtype, storage type,
constant references and consumers are unchanged. The new initial extents are
`[32,1280,1]` or `[64,1280,1]`, within the preserved matching RGBA16F 3D linear
query's `[2048,2048,2048]` bound. The sliced tables remain width-packed for the
rotary operator. This is a source-derived recipe, awaiting GPU correctness.

The existing interfaces supporting that route are
`serialization/schema.fbs` (`VkTensor.memory_layout`),
`runtime/VulkanBackend.cpp` (`get_resolved_memory_layout`),
`runtime/graph/ops/impl/Staging.cpp` (`add_prepack_standard_node`, which passes
the output layout), `impl/Transfer.cpp` (independent input/output layout constants),
`glsl/nchw_to_image.glsl` (packing from logical tensor coordinates) and
`glsl/transfer_texture.glsl` (slice reads with the input's layout and writes with
the output's layout). Paths are under `backends/vulkan/` at the pinned base.
`impl/RotaryEmbedding.cpp` still receives width-packed slice outputs.

The helper verifies the original full-file hashes and sizes, writes separate
artifact files, then compares every byte against the originals. Only the two
recorded layout bytes may differ. This establishes identical serialized constant
payloads, tensor shapes/dtypes, quantization, methods, operators and graph edges.
It also records each Vulkan constant's payload hash, shape, dtype and staging
dtype, resolving named segments through the PTE extended header. These are
logical serialized payload hashes before the unchanged runtime conversion;
they are not hashes of observed GPU results. The existing CPU-only runtime helper
executes every constant method on each new file and compares all outputs with
serialized metadata, including context 2560. It never loads `forward`.

Original inputs remain unchanged. Tokenizers are copied with their pinned hashes.
Prompts are untouched. Outputs are labeled `PARTIAL_REPAIR_NOT_GPU_VALIDATED`.
Interrupted `.partial` files and failed-job states are preserved. A new attempt
must use a new output directory.

## Remaining independent work

The 8B width conflict remains after this repair. Simply changing packing cannot
represent width 14336 in a 2048-wide 3D image: width packing already gives the
smallest available width, 3584. A buffer route must also handle connected linear
inputs/outputs and storage conversions; changing only offending descriptors is
insufficient. Texture2D substitution is not supported by the current linear IO
variants. A conforming 8B recipe and operator-temporary audit remain unfinished;
this document does not assert that all allowed graph interfaces are exhausted.

## Separate synthetic-gate dependency

`sarc/tools/verify.sh:112` invokes production-diff for all three models and both
buffer/texture3d storage modes. In
`backends/vulkan/test/sarc_dev/test_llama_microbench.cpp:950`,
`run_production_diff` constructs fresh synthetic tensors independently of the PTE.
The 8B `w1_w3` and `w2` cases use K/N 4096/14336 and 14336/4096 at M=2048
(`kLinearModels` near line 490), reproducing the forbidden 3584-wide allocation
for fp16 texture3d IO. Revised model files cannot repair these synthetic inputs.
The earlier `--linear` prefill suite includes these shapes as well.

No gate or synthetic GPU run was launched. Reducing M/N, omitting texture cases,
redirecting them to buffers, or accepting an early failure would change coverage;
none is treated as authorized by the input-only approval. The owner dependency
is recorded once in STATUS.md. Full gate and six-cell matrix evidence remain
absent, regardless of partial input-preparation success.
