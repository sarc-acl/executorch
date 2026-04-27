# int8 coopmat shader interpreted size UBO in wrong K units

## What was attempted

The Phase 4 prototype shader `linear_coopmat_int8.glsl` declares its
input tensor as `int [M, K/4]` (each int packs 4 int8 along K) and
reads the size UBO to drive the chunk-K loop:

```glsl
const uint K = uint(mat1_sizes.x);     // <-- WRONG: this is K/4, not K
const uint K_int = K / 4u;
...
for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {
    // stage A and B for this 32-wide K chunk, do coopmat MMA
}
```

`graph.sizes_ubo(input_a)` returns the *tensor* sizes — for an input
declared as `int [M, K/4]`, that x dimension is the int32 count
(`K/4`), not the logical int8 K count. The shader was then dividing
that by 4 again to get `K_int`, so it ran the chunkK loop only
`K/(4 * TILE_K)` times instead of `K / TILE_K`. The kernel processed
only 1/4 of the K dimension on every shape with K > TILE_K.

## How it surfaced

Random-data tests (`RANDINT8` for `kInt`) failed numerical validation
on every shape with K > 32, with diffs in the 10⁴–10⁵ range — but the
GPU outputs were *plausible* magnitudes, suggesting the kernel was
running real work, just on the wrong amount of data.

The smoking gun came from a ones-only diagnostic:

```text
ones_BERT_QKV (128, 768, 768)
   reference  = 192   (= K/4 = 768/4 because only k%4==0 contributes for
                       int32=0x00000001 byte-pattern)
   GPU        = 48    (= 192 / 4)

ones_LLM_QKV_64tok (64, 4096, 4096)
   reference  = 1024  (= K/4 = 4096/4)
   GPU        = 256   (= 1024 / 4)
```

Exact 1/4 ratio on every shape with K > 32. K=32 (`ones_64x32x64`)
passed because the chunkK loop ran exactly 1 iteration either way
(K/TILE_K = 32/32 = 1, or (K/4)/TILE_K = 8/32 = 0 — but the inner mma
runs at least once).

## The fix

```diff
-    const uint K = uint(mat1_sizes.x);
-    const uint K_int = K / 4u;
+    // mat1 is declared as `int [M, K/4]` (each int packs 4 int8 along K),
+    // so the size UBO's x dimension is the int32 count, not the int8 count.
+    // Convert to logical int8 K for the chunk loop.
+    const uint K_int = uint(mat1_sizes.x);
+    const uint K = K_int * 4u;
```

After: 12/12 shapes pass numerical validation under `RANDINT8` data with
the bench harness's `kInt` validator branch.

## Why this matters

1. The pre-fix bench reported `~6 TFLOP/s` on BERT/LLaMA shapes and
   `~19 TFLOP/s` on the 4096³ cube. Those numbers were measured against
   1/4 of the actual K work, so they over-claim throughput by ~4×.
   **Disregard any int8 coopmat throughput figure cited in this repo
   prior to commit landing of `linear_coopmat_int8.glsl`'s K-scaling
   fix.**
2. The post-fix throughput is **1.3–4.1 TFLOP/s on BERT/LLaMA prefill
   shapes and 5.8 TFLOP/s on the cube**, which inverts the Phase 4
   recommendation: int8 coopmat is *slower* than fp16 coopmat at
   LLaMA-prefill shapes on this device, not faster.
3. There is a closely-related design pitfall: **anywhere a shader
   declares an input as packed int (4× int8 per int along the inner
   dim), the size UBO will report the int count, not the int8 count.**
   Any Phase 5 work that reuses this packing convention should put the
   `K = K_int * 4u` conversion *next to* the size UBO read, ideally as
   a helper function, and have a unit test that exercises K > TILE_K.

## Diagnostic technique that found this

Adding a "ones-only" test case where the expected output value is
trivially predictable from the data pattern — for `int32 = 0x00000001`
inputs interpreted as packed int8, the only non-zero byte is byte 0,
so `output[m][n] = K/4`. When the GPU computed exactly `K/16` instead
of `K/4`, the unit-mismatch hypothesis fell out immediately.

This is worth carrying forward as a convention: **every prototype
shader that consumes a packed integer layout should ship with at
least one ones-only diagnostic test case alongside its random-data
test cases, before claiming throughput numbers.**

## Was the user's SNORM/IEEE-754 hypothesis right?

No. ExecuTorch's quantized Vulkan path stores int8 weights as raw
int32-packed buffers (4×int8 per int32), not as SNORM textures, and
the coopmat int8 path uses the integer cooperative-matrix configs
(`int8 × int8 → int32`), no float interpretation in the inner loop.
SNORM/IEEE-754 mixing isn't in the path.

But the *shape* of the user's guess — "GPU and CPU disagree because
they interpret the same bytes differently" — was the correct
debugging frame; the specific layer was a K-dim unit mismatch instead
of a value-encoding mismatch. The right starting question whenever a
GEMM gives "wrong but plausible-magnitude" outputs is "does the GPU
agree with the CPU on what the input data *means*?"
