# conv2d-pw-coopmat Specification

## Purpose

Defines cooperative-matrix execution of FP16 pointwise (1×1) convolution in the ExecuTorch Vulkan backend: when a kernel variant may run, what it computes, how it is selected, and how it is compared against the tiled kernel.

## Requirements

### Requirement: Eligibility is decided from device-reported capabilities
The backend SHALL treat a pointwise convolution as eligible for a cooperative-matrix kernel variant only when the device enumerates the exact cooperative-matrix shape and component types that variant uses and can honor the subgroup size it requires. Eligibility MUST NOT be inferred from vendor or device name.

#### Scenario: Device offers the variant's shape
- **WHEN** the device enumerates a cooperative-matrix property matching the variant's M, N, K sizes and operand and result types, and supports the variant's required subgroup size for compute
- **THEN** a convolution that meets the shape conditions is reported eligible for that variant

#### Scenario: Device lacks the shape
- **WHEN** the device supports cooperative matrices but does not enumerate the variant's shape or result type, or cannot honor the required subgroup size
- **THEN** the convolution is reported ineligible for that variant and its shader is never dispatched

#### Scenario: Device has no cooperative-matrix support
- **WHEN** the device does not support `VK_KHR_cooperative_matrix`
- **THEN** every variant is ineligible and graph construction and execution behave as before

### Requirement: Eligible convolution shapes
The cooperative-matrix kernel SHALL accept only FP16, batch-1, ungrouped, non-transposed 1×1 convolutions with stride 1, padding 0, dilation 1, texture input and output, buffer-stored packed weights, and more than one output pixel. Every other pointwise convolution SHALL be ineligible.

#### Scenario: MobileNet projection layer
- **WHEN** an FP16 1×1 convolution has input `[1,960,7,7]`, weight `[160,960,1,1]`, stride 1 and padding 0
- **THEN** it is eligible on a device that meets the capability requirement

#### Scenario: Single output pixel
- **WHEN** an FP16 1×1 convolution has a 1×1 output plane
- **THEN** it is ineligible

#### Scenario: Strided or padded pointwise convolution
- **WHEN** a 1×1 convolution has stride other than 1 or nonzero padding
- **THEN** it is ineligible

#### Scenario: FP32 convolution
- **WHEN** a 1×1 convolution has an FP32 output
- **THEN** it is ineligible

### Requirement: Results match the reference for any tile alignment
On the shipped correctness cases each kernel variant SHALL match the CPU FP32 reference within the FP16 convolution tolerance (absolute 0.1, relative 0.01) whether or not the pixel and channel counts are multiples of its tile sizes, and MUST NOT write outside the output tensor. Accuracy on other inputs follows from the accumulator type (the two accumulation requirements).

#### Scenario: Aligned shape
- **WHEN** a variant runs a convolution whose pixel count and channel counts are multiples of its tile sizes
- **THEN** every output element is within tolerance of the reference

#### Scenario: Odd spatial size
- **WHEN** a variant runs input `[1,576,7,7]` with weight `[96,576,1,1]` (49 output pixels)
- **THEN** every output element is within tolerance of the reference

#### Scenario: Channel counts not multiples of the tile
- **WHEN** a variant runs a convolution with 24 output channels or 40 input channels
- **THEN** every output element is within tolerance of the reference and no element outside the output tensor is written

#### Scenario: Plane smaller than the work-group tile
- **WHEN** a variant with a 64-pixel M tile runs a convolution with 49 output pixels
- **THEN** every output element is within tolerance of the reference

### Requirement: fp32 accumulation keeps fp32 until the output store
A variant with fp32 accumulation SHALL keep fp32 from the products through the K sum, the bias and the fused epilogue, with the epilogue parameters (activation bounds, residual alpha) passed at full precision, and round to fp16 only at the output store, so it matches the reference within tolerance on the precision regressions of the correctness collection.

#### Scenario: Bias cancels the product
- **WHEN** an fp32-accumulation variant runs input 65, weight 65 and bias -4224 (one input and one output channel)
- **THEN** every output element is 1 within tolerance

#### Scenario: Residual cancels the product
- **WHEN** an fp32-accumulation variant runs input 65 and weight 65 with a fused residual -4224 added
- **THEN** every output element is 1 within tolerance

#### Scenario: Residual alpha that fp16 cannot represent
- **WHEN** an fp32-accumulation variant runs input 64 and weight 64 with a fused residual -4096 scaled by alpha 1.000244140625
- **THEN** every output element is -1 within tolerance

#### Scenario: Ordered cancellation over a long K
- **WHEN** an fp32-accumulation variant runs K = 2048 unit weights over an input that is 768 ones, 512 values of 2^-7 and 768 minus ones along K
- **THEN** every output element is 4 within tolerance

### Requirement: fp16 accumulation has the tiled kernel's precision
A variant with fp16 accumulation rounds the K sum, the bias add and the epilogue to fp16, as the tiled kernel does, and is not required to match the reference where that rounding exceeds the tolerance. Such inputs SHALL be kept as documented expected failures, with the tiled kernel's result on them, in a collection outside the correctness collection.

#### Scenario: fp16 accumulation on the precision regressions
- **WHEN** an fp16-accumulation variant or the tiled kernel runs the four precision regression inputs
- **THEN** the results are recorded in the fp16 accumulation limits collection, which states which of them fail

### Requirement: Dispatch within device limits
A variant SHALL be eligible only when its work group fits the device limits (total invocations, the x dimension of the work group size and, with its required subgroup size, the number of subgroups per work group) and its shared memory fits the device limit, and its dispatch SHALL stay within the device's maximum work group counts for every eligible plane by spreading pixel tiles over more than one dispatch dimension. The reason for ineligibility SHALL name the limit and the device's value.

#### Scenario: More pixel tiles than the work group count limit in one dimension
- **WHEN** a variant with a 16-pixel tile runs a 1025 x 1024 output plane (65600 pixel tiles) on a device that allows 65535 work groups in y
- **THEN** every work group count of the dispatch is within the device limits and every output element is within tolerance of the reference

#### Scenario: Work group beyond a device limit
- **WHEN** a variant's work group exceeds the device's maximum work group size in x or its maximum number of subgroups per work group
- **THEN** the variant is ineligible, no pipeline is created for it, and the reason states the limit and the device's value

#### Scenario: Shared memory of the largest tile
- **WHEN** the 128 x 128 variants are built
- **THEN** each declares at most 48 KiB of shared memory, the smallest limit of the measured devices, and the eligibility check compares the declared size with the device limit

### Requirement: Bias and fused epilogue parity
The cooperative-matrix kernel SHALL apply bias and the fused epilogue (activation, residual addition, per-channel scale) with the same semantics as the tiled pointwise kernel.

#### Scenario: Bias and activation
- **WHEN** an eligible convolution has a bias and a fused clamp activation
- **THEN** the output matches the reference for convolution, bias and activation applied in that order

#### Scenario: No bias
- **WHEN** an eligible convolution has no bias
- **THEN** the output matches the reference for the convolution alone

#### Scenario: Residual epilogue
- **WHEN** an eligible convolution has a fused residual input
- **THEN** the output matches the output of the tiled kernel for the same inputs within tolerance

### Requirement: Variants per GPU family
The kernel SHALL be provided as named variants for the Intel Xe2, AMD and NVIDIA families, each using a cooperative-matrix shape and subgroup size that family's devices enumerate. For each family the variants SHALL cover work-group output tiles of 16, 32 and 64 pixels by 32 and 64 channels and one 128×128 tile, each with fp16 accumulation and with fp32 accumulation where the device offers that result type.

#### Scenario: Variant set for a family
- **WHEN** the shipped variants for a family are listed
- **THEN** they include all seven work-group tiles, and each accumulator type the family's measured device enumerates

#### Scenario: Accumulator type not offered
- **WHEN** a device does not enumerate the result type of an accumulator variant
- **THEN** that variant is ineligible on the device and the measurement report states that it was not run and why

### Requirement: Selection is opt-in and by variant name
Automatic routing of convolution operators SHALL keep using the tiled pointwise kernel. A cooperative-matrix variant SHALL run only when a caller explicitly requests it by name, and a request for an ineligible convolution, an unknown variant, or a variant the device cannot run SHALL be reported as unsupported rather than silently running another kernel.

#### Scenario: Automatic routing unchanged
- **WHEN** a model or benchmark case uses automatic routing on a device where the convolution is eligible
- **THEN** the dispatched shader is a `conv2d_pw_tiled*` variant, as before this change

#### Scenario: Explicit request on an eligible case
- **WHEN** a benchmark case requests a named variant for an eligible convolution on a device that offers its shape
- **THEN** the dispatched shader is that variant

#### Scenario: Variant of another family
- **WHEN** a benchmark case on B580 requests a variant whose cooperative-matrix shape B580 does not enumerate
- **THEN** the case is reported as skipped with the reason, and no result is attributed to the cooperative-matrix kernel

### Requirement: Eligibility follows dynamic shapes
When a graph is resized, the backend SHALL re-evaluate eligibility for the new shape, and a convolution that requested a cooperative-matrix variant but is no longer eligible SHALL execute correctly on the tiled kernel.

#### Scenario: Resize to a single pixel
- **WHEN** a convolution built with a cooperative-matrix variant requested is resized so that its output plane is 1×1
- **THEN** it executes on the tiled kernel and the output matches the reference

#### Scenario: Fallback and recovery keep each kernel's epilogue parameters
- **WHEN** a convolution built with an fp32-accumulation variant and residual alpha 1.000244140625 is resized to an ineligible shape and back
- **THEN** the tiled fallback uses the fp16-rounded parameters of the tiled kernel, and after the resize back the variant uses the full-precision parameters again and gives the fp32 result

### Requirement: Same-case comparison against two tiled baselines
The operator benchmark SHALL run one case definition through three routes: the cooperative-matrix kernel with a named variant, the tiled kernel the shape dispatches under automatic routing, and the tiled kernel forced to the 4x8 tile. Each result SHALL report the dispatched shader name, GPU shader time and validation status.

#### Scenario: Three routes on one case
- **WHEN** a case for `(M,N,K) = (196,96,576)` is run once with each route
- **THEN** the three results use identical inputs and each names its dispatched shader

#### Scenario: Forced 4x8 on a small shape
- **WHEN** the forced 4x8 route runs a shape that automatic routing sends to the narrow tile
- **THEN** the dispatched shader is `conv2d_pw_tiled_4x8_bufw_half`

### Requirement: Benchmark shape sets
The shipped performance cases SHALL cover three shape sets: the MobileNetV2 and V3 Small trace shapes `(49,160,960)`, `(196,64,384)`, `(196,96,576)` and `(49,96,576)`; the stride-1 1×1 layers of ResNet-50 at 224×224 input; and the FP16 shapes of the upstream pointwise performance cases.

#### Scenario: MobileNet shapes are covered
- **WHEN** the shipped performance cases are listed
- **THEN** they include the four MobileNet shapes

#### Scenario: ResNet-50 shapes are covered
- **WHEN** the shipped performance cases are listed
- **THEN** they include 1×1 layers with 3136, 784, 196 and 49 output pixels and channel counts from 64 to 2048, and no stride-2 layer

### Requirement: Device capability record
The benchmark binary SHALL be able to report, on the device it runs on, every enumerated cooperative-matrix property (sizes, component types, scope), the subgroup size range and the driver version, without needing source or a build on that device.

#### Scenario: Desktop GPU
- **WHEN** the capability report is run on B580
- **THEN** it lists the fp16 cooperative-matrix shapes and their result types and the supported subgroup sizes

#### Scenario: Device without cooperative matrices
- **WHEN** the capability report is run on a device that does not support `VK_KHR_cooperative_matrix`
- **THEN** it states that no properties are enumerated and exits successfully

### Requirement: Measurement admits only validated results
Timing SHALL be recorded only for routes, variants and cases that passed validation: a variant is timed only after it passed every case of the correctness collection on the device, a performance case that fails validation keeps its row with the failure and no timings, and a command failure stops the measurement. A verdict SHALL use only rows that passed in every run of their result set, dispatched the expected ordered shader sequence and, for a variant, passed the correctness collection.

#### Scenario: Runs per result set
- **WHEN** a result set is summarized
- **THEN** it holds three runs if it belongs to the first measurement and five otherwise, and a row missing any of them gets no ratio

#### Scenario: Variant fails correctness
- **WHEN** a variant fails a case of the correctness collection on a device
- **THEN** it is not timed on that device and the report lists it as failed with the reason

#### Scenario: Performance case fails validation
- **WHEN** a case of a performance file fails validation during a measurement run
- **THEN** its result keeps the validation failure, has no timing samples, and the row is excluded from the verdict

#### Scenario: Incomplete evidence
- **WHEN** a row lacks one of its result set's runs, its correctness result, or the expected dispatched shader
- **THEN** it gets no ratio and the report states why

#### Scenario: Incomplete correctness evidence
- **WHEN** a variant's expected correctness rows, derived from the exact case files pushed to the device, are missing, duplicated, partly SKIP, or PASS on another shader
- **THEN** the variant is not admitted; a missing, duplicated or unexpected row, or a baseline row that does not PASS on its exact expected shader, stops the measurement before timing

#### Scenario: Command or timing failure
- **WHEN** a measurement step fails as a command (exit status other than 0 or 1, or exit 1 without a recorded validation failure), leaves no complete result, or has a row with a timing or benchmark error
- **THEN** the measurement stops, the row keeps its error and is never counted as passing; an exit status 1 is accepted only when the result holds every expected row and its non-passing rows are recorded validation failures without timing samples

#### Scenario: Incomplete device evidence
- **WHEN** a desktop shape lacks a valid `auto` or `pw_tiled_4x8` row in any run of the result set, the capability record does not pass with the device's name, UUID and driver, or a step's saved command is missing
- **THEN** the device gets no verdict and its table lists the problem, while a variant that fails a run stays listed with its reason and gets no ratio

#### Scenario: Timed rows without complete timing
- **WHEN** a passing row of a performance run lacks any of its defined number of positive GPU and wall timing samples or a shader-breakdown entry for each shader of the expected sequence, or a failed or skipped row carries timing samples
- **THEN** the measurement stops and the device gets no verdict, while a genuine untimed validation failure keeps its row and reason

#### Scenario: Retry under the same tag
- **WHEN** a measurement is rerun under a tag that already holds a completed attempt and a step of the rerun fails before writing its result
- **THEN** the rerun stops, the earlier attempt's completion record and the step's earlier result cannot stand for it, and the device gets no verdict

#### Scenario: Evidence of another workload
- **WHEN** a saved command is not exactly the recorded invocation, an input used by the summary lacks matching local and on-device checksums or differs from the measured commit's case file, or a result row reports another operator or other arguments than its definition
- **THEN** the device gets no verdict, independently of the post-measurement manifest

#### Scenario: Validation record that is not the output comparison
- **WHEN** a passing or failing row lacks the output comparison its definition asks for (method, tolerances, element count, mismatch and non-finite counts consistent with the statuses), or carries an exception instead
- **THEN** the row is an error, not a pass or a numerical failure, and the measurement stops before admitting or timing on it

#### Scenario: Device and deployment identity
- **WHEN** the capability record lacks its cooperative-matrix property list, subgroup sizes or raw driver version, a timed row names another device than the capability record, or the deployment record does not hold both binary checksums of the staged build of its commit
- **THEN** the device gets no verdict, and a new measurement does not start GPU work unless the binaries on the device now are that staged build

#### Scenario: Variant the device does not offer
- **WHEN** every correctness row of a variant is SKIP
- **THEN** the variant is listed as not offered with the SKIP reasons, separately from failed variants

#### Scenario: Summary from raw evidence
- **WHEN** the per-device table is generated
- **THEN** the admission is recomputed from the raw correctness result and the saved case files, every expected performance row is listed (missing and skipped rows with their reasons), and a device with incomplete evidence gets no verdict

#### Scenario: Variant kind with expected skips
- **WHEN** an aligned-only variant skips a correctness case whose channel counts are not multiples of its K step and N tile, a split-K variant skips a case whose K is too short to split, or an epilogue-skip variant skips a case with a bias, activation, residual or scale
- **THEN** the skip is an expected row for that variant kind and does not prevent admission, while any other skip does

### Requirement: Recorded per-device measurement
The change SHALL deliver, for each of B580, B70, Radeon 780M and RTX 4070 Ti SUPER, a per-shape table at `OUT/<device>/table.md` stating for every route and variant the dispatched shader, median GPU shader time, median dispatch wall time and validation status, with the commit, driver version, capability record, exact commands and exact inputs. For Pixel 7a and Galaxy S24+ it SHALL deliver the capability record and the tiled baseline only.

#### Scenario: Table is reproducible
- **WHEN** a reader reruns the recorded commands with the recorded binary on the recorded device
- **THEN** they obtain the same set of rows with the same dispatched shader names

#### Scenario: Earlier measurements are kept
- **WHEN** a new measurement replaces the published tables
- **THEN** the earlier tables, summaries and reports stay under paths named by their commit and say that they are historical

#### Scenario: Published results match the raw evidence
- **WHEN** the verification recomputes every table, summary and the report from a copy of the saved raw results
- **THEN** each equals the published file byte for byte, and removing or changing a raw result makes the verification fail

#### Scenario: Raw evidence changed after publication
- **WHEN** a raw result, saved command or other original evidence file is changed, removed or added after the measurement, even if every derived table stays the same
- **THEN** the verification fails on the recorded checksum manifest of the evidence

#### Scenario: Slower shapes are reported
- **WHEN** a cooperative-matrix variant is slower than the tiled kernel for a shape
- **THEN** the table shows that row with its ratio rather than omitting it

#### Scenario: Device that could not be made to pass
- **WHEN** a family's variants still miscompute or fail to create a pipeline on a device after three fix attempts
- **THEN** the report lists that device as not passed with the observed symptom, and the other devices' results are delivered

### Requirement: Two-times verdict per device and shape
The report SHALL classify every shape on every measured device by the ratio of the dispatched tiled baseline's median GPU time to that of the best cooperative-matrix variant admitted and timed in the same session, each route timed from the start of its first dispatch to the end of its last: shapes at 2.0 or above are candidates for routing, shapes below keep the tiled kernel. A device on which none of the three MobileNetV2 shapes reaches 2.0 SHALL be marked as not ready for end-to-end work.

#### Scenario: Shape reaches the bar
- **WHEN** the best variant for a shape on a device is at least 2.0 times faster than the dispatched tiled baseline
- **THEN** the report marks the shape as a candidate on that device and names the variant, and also states the ratio of the best fp32-accumulation variant, since an fp16-accumulation candidate keeps the tiled kernel's fp16 precision

#### Scenario: Shape misses the bar
- **WHEN** the best variant for a shape on a device is less than 2.0 times faster
- **THEN** the report marks the shape as keeping the tiled kernel and states the ratio

#### Scenario: No MobileNetV2 shape reaches the bar
- **WHEN** none of `(49,160,960)`, `(196,64,384)` and `(196,96,576)` reaches 2.0 on a device
- **THEN** the report marks that device as not ready for end-to-end work

#### Scenario: Timed population is stated
- **WHEN** a device's verdict is reported
- **THEN** the report lists the variants that were admitted and timed in that session, and for each MobileNetV2 shape the technique of its best variant and the ratio against the forced 4x8 tiled kernel

#### Scenario: Ratio within noise of the bar
- **WHEN** a shape's best ratio is within the device's noise band of 2.0
- **THEN** it is classified by the exact value and flagged as within noise of the bar

#### Scenario: Device still below the bar
- **WHEN** no MobileNetV2 shape reaches 2.0 on a device with any timed variant
- **THEN** the report states the best ratio per shape and what the diagnostic record says about where the time goes

### Requirement: Follow-up measurement protocol
Result sets produced by this change SHALL use five passing runs per timed row, with the arms interleaved in a recorded order. A route's GPU time SHALL be the interval from the start of its first dispatch to the end of its last within one invocation, for every arm including the tiled baseline, and each row SHALL record the ordered dispatched shaders and their times.

#### Scenario: Earlier report is kept
- **WHEN** the new result sets are written
- **THEN** the first change's three-run report and its files are unchanged

#### Scenario: Route with two dispatches
- **WHEN** a split-K route dispatches a kernel and a merge shader
- **THEN** its row lists both shader names in order, each shader's time, and a route time that spans both

#### Scenario: Unexpected shader sequence
- **WHEN** a row's dispatched shader sequence differs from the one expected for its route and variant
- **THEN** the row is excluded from ratios and reported with the sequence it produced

#### Scenario: Fewer than five passing runs
- **WHEN** a row has fewer than five runs that passed validation
- **THEN** it is excluded from ratios and from the verdict and listed with its run count

### Requirement: Measurements carry a noise band
Every per-device result set SHALL include an A/A comparison of the device's parent variant against itself on each of the four MobileNet shapes, taken in the same session. The noise band for a shape SHALL be the largest of the five paired values 2·|A − A′| / (A + A′), and not less than 0.02. Two variants whose medians differ by less than the band, measured the same way, SHALL be reported as no change. A session whose band exceeds 0.10 on any MobileNet shape SHALL be rejected and repeated.

#### Scenario: Difference inside the band
- **WHEN** a technique's best variant differs from its parent by less than the noise band on a shape
- **THEN** the report lists the technique as no change for that device and shape

#### Scenario: A/A missing
- **WHEN** a result set has no A/A comparison
- **THEN** its ratios are not used in the attribution table or the verdict

### Requirement: Decision gates are fixed before data
The rules that decide whether the split-K prototype is built and which probes are carried to the fleet run SHALL be written to a gates file, with the parent variant per device, the cases, the formulas, the aggregation over shapes and devices, inclusive boundaries and an inconclusive outcome, and that file SHALL be committed before the first diagnostic result is recorded. Gate outcomes SHALL be computed from the saved results by a script, and the inputs and outcomes SHALL be saved.

#### Scenario: Gates predate the data
- **WHEN** the gates file's commit and the first diagnostic result file are compared
- **THEN** the commit is earlier

#### Scenario: Poor fit
- **WHEN** the fit a gate relies on has a coefficient of determination below the value stated in the gates file
- **THEN** the gate outcome is inconclusive and is handled as the gates file states

#### Scenario: Outcome is reproducible
- **WHEN** the gate script is rerun on the saved results
- **THEN** it prints the same outcomes as recorded

### Requirement: Diagnostic variants keep an observable result
A diagnostic variant SHALL write an output that depends on every K step it executes, SHALL be marked diagnostic in results, and MUST NOT appear in any speedup table. The diagnostic record SHALL state, for each diagnostic variant on B580, evidence from the compiled code that its K loop was not removed.

#### Scenario: Staged-once roof
- **WHEN** the staged-once variant runs a shape
- **THEN** its output differs when K changes, and the record names the tile, grid, K step count, subgroup size and accumulator it shares with its parent

#### Scenario: Diagnostic row in a result set
- **WHEN** a diagnostic variant is timed
- **THEN** its rows are marked diagnostic and excluded from every ratio against the tiled kernel

### Requirement: Aligned-only variants refuse unaligned shapes
A variant that omits bounds checks on K or N SHALL be eligible only when the convolution's input and output channel counts are multiples of that variant's K step and N tile. A request for it on any other shape SHALL be reported as unsupported when the graph is built. The weights fix both channel counts, so a resize cannot make such a graph unaligned; a graph resized to a plane the variant cannot run (a single pixel) SHALL execute correctly on the tiled kernel.

#### Scenario: Aligned MobileNet shape
- **WHEN** an aligned-only variant with K step 16 and N tile 32 is requested for `(196,96,576)`
- **THEN** it is dispatched and the output matches the reference within tolerance

#### Scenario: Unaligned channels
- **WHEN** an aligned-only variant is requested for a convolution with 24 output channels or 40 input channels
- **THEN** the case is reported as skipped with the reason and no other kernel is run in its place, and building the graph through the convolution's graph-construction path fails with the reason

#### Scenario: Partial pixel tile
- **WHEN** an aligned-only variant runs a shape whose pixel count is not a multiple of its M tile, such as 49 pixels
- **THEN** every output element matches the reference within tolerance and no element outside the output tensor is written

### Requirement: Split-K results match the reference
A variant that divides the K steps among several workers for the same output tile SHALL meet the same precision requirements as the unsplit variant with the same accumulator type, for every eligible shape including one whose K is not a multiple of the K step or of the slice count, and SHALL apply bias and the fused epilogue exactly once per output element, after the partial sums are combined. It SHALL be eligible only when every slice holds at least two K steps.

#### Scenario: K not divisible
- **WHEN** a split-K variant with 4 slices runs a convolution with 321 input channels (21 K steps of 16, 11 of 32), or one with K step 16 runs 257 input channels
- **THEN** the output matches the reference within tolerance

#### Scenario: Uneven K that leaves a slice short
- **WHEN** a split-K variant with 4 slices and K step 32 is requested for 257 input channels (9 K steps, so the last slice gets none)
- **THEN** the case is reported as skipped with the reason

#### Scenario: Scratch for every reachable plane
- **WHEN** a split-K graph is built with a logical plane smaller than its output storage and is resized up to the full storage
- **THEN** the split-K kernel and merge still run and the output matches the reference; and a request whose packed weights and scratch exceed the device's storage buffer range is refused at graph construction with a reason naming the limit

#### Scenario: K too short to split
- **WHEN** a split-K variant with 4 slices is requested for a convolution with 40 input channels
- **THEN** the case is reported as skipped with the reason

#### Scenario: fp32 accumulation through the merge
- **WHEN** an fp32-accumulation split-K variant runs the first change's bias-cancellation and ordered-cancellation precision cases
- **THEN** the results are the ones required of the unsplit fp32 variant

#### Scenario: Bias, activation and residual
- **WHEN** a split-K variant runs a convolution with bias and a fused clamp, and one with a fused residual input
- **THEN** each output matches the reference with bias and epilogue applied once

#### Scenario: Large plane and resize
- **WHEN** a split-K variant runs a 1025×1024 plane, and a graph using it is resized to a 1×1 plane and back
- **THEN** the dispatch stays within the device's work-group count limits and every output matches the reference

### Requirement: New variants stay opt-in and leave existing kernels unchanged
Every variant added by this change SHALL be reachable only by an explicit request naming it. Automatic routing SHALL keep dispatching the tiled pointwise kernel for convolutions it routes as pointwise, and every variant name of the first change SHALL keep its dispatched shader and output.

#### Scenario: Automatic routing unchanged
- **WHEN** the upstream pointwise performance cases run with automatic routing after this change
- **THEN** the dispatched shader names are the same as before this change

#### Scenario: Existing named variant unchanged
- **WHEN** a variant name from the first change is requested after this change
- **THEN** the same shader is dispatched and the correctness cases give the same pass and skip set

### Requirement: Kept variants are valid Vulkan usage
Every variant that appears in a speedup table for a device SHALL have run its correctness cases under the Vulkan validation layers on that device with no validation message.

#### Scenario: Validation message
- **WHEN** a variant produces a validation message on a device
- **THEN** it is excluded from timing on that device and listed with the message identifier

### Requirement: Per-technique attribution against a named parent
For each technique added by this change, the report SHALL state, per device on which it was measured and for each MobileNet shape, the time of the technique's variant, of the one named parent variant that differs from it by that option only, and of the first change's best variant, all from the same session. A technique screened out before the fleet run SHALL be listed for the other devices as not measured.

#### Scenario: Techniques covered
- **WHEN** the attribution table is read
- **THEN** it has a row for aligned-only, K step, thread grid, small tile, wide weight load, storage-image input, epilogue skip, column-major weights and split-K, each naming its parent variant

#### Scenario: Technique helps
- **WHEN** a technique's variant is faster than its parent by more than the noise band on a shape and device
- **THEN** the report shows both times and the ratio

#### Scenario: Technique screened out
- **WHEN** a technique was not carried to the fleet run
- **THEN** the report shows its ratios on the screening devices and marks the remaining devices as not measured

### Requirement: Clamped-read variants accept any output channel count
A variant whose K loop has no bounds checks and reads past the last output channel from the last valid weight position SHALL be eligible for every convolution whose input channel count is a multiple of its K step, whatever the output channel count and output pixel count, SHALL produce output within the precision requirements of the checked variant with the same tile and accumulator, and SHALL NOT write any element outside the output tensor.

#### Scenario: Tile wider than the channel count
- **WHEN** a clamped-read variant with a 128-channel tile runs `(49,160,960)`, `(196,64,384)` and `(196,96,576)`
- **THEN** each output matches the reference within tolerance

#### Scenario: Channel count not a multiple of four
- **WHEN** a clamped-read variant runs a convolution with 24 output channels and one with 30
- **THEN** every output element matches the reference within tolerance and no element outside the output tensor is written

#### Scenario: Input channels not a multiple of the K step
- **WHEN** a clamped-read variant is requested for a convolution with 40 input channels
- **THEN** the request is reported as unsupported with the reason

#### Scenario: Bias, epilogue and precision
- **WHEN** a clamped-read variant runs the bias, clamp, residual and fp32 precision cases of its checked parent
- **THEN** the results are those required of the parent

#### Scenario: Resize
- **WHEN** a graph built with a clamped-read variant is resized to a 1x1 output plane and back
- **THEN** the 1x1 execution runs a tiled shader and both outputs match the reference

### Requirement: Stacked variants keep each technique's guarantees
A variant that combines clamped reads with split-K or with a different subgroup grid SHALL meet the correctness requirements of each technique it combines, including split-K's eligibility and merge rules.

#### Scenario: Clamped reads with split-K
- **WHEN** a clamped-read split-K variant runs the split-K correctness cases whose input channel count is a multiple of its K step
- **THEN** every case passes, with bias and epilogue applied once after the merge

#### Scenario: Existing variants unchanged
- **WHEN** the correctness collection runs for every variant that existed before this change
- **THEN** its pass and skip set and dispatched shaders are unchanged

### Requirement: Fast-pass result is reported with its limits
The result of this change SHALL be reported as a B580 kernel-level result on the four MobileNet shapes: for each shape the time of the automatically routed tiled kernel, of the forced 4x8 tiled kernel, of the previous best variant and of each new variant, from one session with an A/A noise band, with the generated-code check of each new variant, and with a statement that no other device and no other shape set was measured.

#### Scenario: Best stacked variant per shape
- **WHEN** the report is read
- **THEN** it names for each MobileNet shape the fastest admitted variant, its ratio to the tiled kernel and to the previous best, and whether the difference exceeds the noise band

#### Scenario: Generated code checked before timing
- **WHEN** a new variant's row appears in the table
- **THEN** the record shows that its B580 generated code has the K loop and its parent's multiply-add count per K step, and gives its load and conditional counts per K step beside the parent's

#### Scenario: Technique that does not stack
- **WHEN** a stacked variant is not faster than the better of its two parents beyond the noise band
- **THEN** the report says so with the three times
