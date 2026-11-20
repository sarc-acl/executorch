/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <optional>
#include <string>

namespace vkcompute {

/*
 * End-to-end orchestration for an FP32 / FP16 conv2d computed as im2col ->
 * GEMM.
 *
 * Dataflow (input t_in [1, C_in, H_in, W_in] -> output t_out [1, C_out, H_out,
 * W_out]):
 *
 * The im2col matrix is materialized and consumed in tiles of output-height rows
 * (see kIm2colScratchBudgetBytes in conv2d_gemm_impl): each tile covers oh_tile
 * output-height rows, and the conv is computed in ceil(H_out / oh_tile) tiles.
 * This bounds the im2col scratch to a fixed byte budget regardless of
 * resolution. Per tile:
 *   1. im2col: add_conv2d_im2col_node (conv2d_im2col.glsl) expands t_in's
 *      oh_tile-row window into the im2col scratch. The K (reduction) axis is
 *      laid out as K = (ki * Kw + kj) * Cin_padded + ci, with K_total = Kh * Kw
 *      * align_up_4(C_in). The im2col storage type selects the scratch shape:
 *        - buffer / texture2d: flat [oh_tile * W_out, K_total], width-packed.
 *        - texture3d: [1, K_total, oh_tile, W_out], channels-packed, so the K4
 *          tiles lay along Z.
 *   2. GEMM: add_conv2d_gemm_node (conv2d_gemm.glsl) multiplies the im2col tile
 *      by the packed weight [C_out, K_total] and writes that tile's output rows
 *      of t_out. The packed weight is prepacked on the GPU directly from the
 *      serialized [C_out, C_in, Kh, Kw] weight (no CPU repack), applying the
 *      im2col K-axis decode plus a 4OC x 4IC blocked transpose.
 *
 * This function performs both dispatch and prepack registration. A single
 * im2col scratch TmpTensor (sized for one tile) is reused across all tiles —
 * reusing it serializes the tiles via the backend's read/write barriers and
 * lets the memory planner alias one backing buffer across the non-overlapping
 * im2col lifetimes of every conv2d layer (peak memory tracks the largest single
 * tile's scratch — at most the budget — rather than the sum). The packed weight
 * (shared across tiles) is produced once by a GPU prepack node (PrepackNode
 * running pack_conv2d_gemm_weight.glsl) from the serialized weight.
 *
 * Constraints (asserted internally):
 *   - input batch == 1
 *   - weight rank == 4
 *   - groups == 1 (general grouped conv not yet supported)
 *   - transposed == false
 *
 * `im2col_storage_override` controls the storage type of the im2col
 * intermediate tensor (and, by extension, the conv2d_gemm input-load variant):
 *   - std::nullopt (default): the production path. Storage is auto-selected
 *     from device characteristics and texture-extent limits — byte-for-byte
 *     the same selection used by the registered op.
 *   - a concrete StorageType: force that storage, skipping auto-selection.
 *     Used by tests to exercise each storage variant deterministically and
 *     independently of the device. Must be one of kBuffer / kTexture2D /
 *     kTexture3D.
 */
void conv2d_gemm_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue = {},
    const std::optional<utils::StorageType> im2col_storage_override =
        std::nullopt,
    const Conv2dPwCoopmatVariant* coopmat_variant = nullptr);

/*
 * With `coopmat_variant` (an _i2c variant of conv2d_pw_coopmat, benchmark route
 * im2col_coopmat) the GEMM of each row tile is that cooperative-matrix variant
 * over a texture3d im2col scratch, with weights packed into a buffer by
 * prepack_conv2d_gemm_weight (and, for split-K, one merge over every tile after
 * the last). conv2d_gemm_impl throws when conv2d_im2col_coopmat_ineligible
 * names a reason. While a resize gives a shape the variant cannot run
 * (conv2d_im2col_coopmat_runnable is false), the variant and merge dispatch
 * nothing and each tile runs conv2d_gemm on the same scratch with texture2d
 * weights, which the route keeps beside the buffer ones. A chunked run
 * (Conv2dIm2colCoopmatPlan::chunks > 1) has no such fallback: its multiplies
 * and merge also compute a single-pixel output.
 */

// The serialized [C_out, C_in, K_h, K_w] weight packed on the GPU into the 4OC
// x 4IC blocked layout of a [C_out, K_h * K_w * align_up_4(C_in)] matrix in
// im2col K order (the layout of pack_fp_linear_weight over that matrix), as a
// texture2d or a buffer; a buffer gets `scratch_rows` more rows of N4 * 16
// elements after the weights, which the prepack does not write. A nonzero
// `packed_k4` (buffer only, at least K / 4) gives the weights packed_k4 rows,
// the rows past K / 4 zero.
ValueRef prepack_conv2d_gemm_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const utils::StorageType weight_storage = utils::kTexture2D,
    const int64_t scratch_rows = 0,
    const int64_t packed_k4 = 0);

// How route im2col_coopmat runs a convolution on a variant. The live taps are
// the kernel rows [ki0, ki0 + kh) and columns [kj0, kj0 + kw) that read the
// input for some output pixel at the build-time sizes (kh or kw is 0 when no
// tap does); the others only read padding and are left out of the gather and
// the packed weights. When the gathered matrix of the live taps is deeper than
// maxImageDimension3D, the input channels run in `chunks` chunks of
// `chunk_channels` (a multiple of 4; the last may extend past C_in) on the
// variant's chunked form `multiply` (_ck), whose partial sums one merge adds
// before bias and epilogue. `multiply` is null when a chunked run needs a _ck
// form the variant does not have, chunks is 0 when no chunk fits.
struct Conv2dIm2colCoopmatPlan final {
  int64_t ki0;
  int64_t kh;
  int64_t kj0;
  int64_t kw;
  int64_t chunks;
  int64_t chunk_channels;
  // K of one chunk's gathered matrix: kh * kw * align_up_4(chunk_channels).
  int64_t K_chunk;
  int64_t oh_tile;
  int64_t num_tiles;
  const Conv2dPwCoopmatVariant* multiply;

  // Whether the run is the full convolution in one chunk.
  bool full(const int64_t K_h, const int64_t K_w) const {
    return chunks == 1 && ki0 == 0 && kj0 == 0 && kh == K_h && kw == K_w;
  }
  // K slices of the scratch summed by the merge (1: no merge).
  uint32_t scratch_slices() const {
    return utils::safe_downcast<uint32_t>(chunks) *
        (multiply == nullptr ? 1u : multiply->k_slices);
  }
};

Conv2dIm2colCoopmatPlan conv2d_im2col_coopmat_plan(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out);

// Whether the cooperative-matrix multiply of an unchunked run runs at the
// output's current sizes: more than one output pixel.
bool conv2d_im2col_coopmat_runnable(ComputeGraph& graph, const ValueRef out);

// Why the im2col cooperative-matrix route cannot run the convolution on the
// variant, naming the device limit where one is exceeded, or nullopt. `bias`
// may be none. The epilogue is checked by check_conv2d_epilogue, which
// conv2d_gemm_impl calls first for a cooperative-matrix request.
std::optional<std::string> conv2d_im2col_coopmat_ineligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool transposed);

// How route direct_coopmat runs a convolution on a _dg variant: the live taps
// (as in Conv2dIm2colCoopmatPlan; kh or kw is 0 when no tap reads the input),
// the gathered depth K = kh * kw * align_up_4(C_in) and the packed weight rows
// (K4 rounded up to whole K steps of the variant).
struct Conv2dDirectCoopmatPlan final {
  int64_t ki0;
  int64_t kh;
  int64_t kj0;
  int64_t kw;
  int64_t K;
  int64_t packed_k4;
};

Conv2dDirectCoopmatPlan conv2d_direct_coopmat_plan(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out);

// Why route direct_coopmat cannot run the convolution on the variant, naming
// the device limit where one is exceeded, or nullopt. `bias` may be none; the
// epilogue is checked by check_conv2d_epilogue.
// `allow_diagnostic`: a benchmark case may request a diagnostic variant
// (Conv2dPwCoopmatVariant::diagnostic).
std::optional<std::string> conv2d_direct_coopmat_ineligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool transposed,
    const bool allow_diagnostic = false);

/*
 * Route direct_coopmat: an FP16 2D convolution as one cooperative-matrix
 * multiply on a _dg variant that gathers its A operand from the input (stride,
 * padding and dilation in the shader), with no im2col dispatch or scratch.
 * The weights of the live taps are packed in im2col K order into a buffer,
 * zero up to whole K steps; a split-K variant stores its slices after them and
 * one conv2d_pw_coopmat_skn_merge applies bias and epilogue. Throws when
 * conv2d_direct_coopmat_ineligible names a reason. Every shape a resize gives
 * runs on the variant (there is no fallback kernel).
 */
void conv2d_direct_coopmat_impl(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue = {},
    const bool allow_diagnostic = false);

} // namespace vkcompute
