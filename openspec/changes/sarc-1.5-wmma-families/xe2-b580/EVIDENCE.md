# Recovery evidence for one independent review

The measured stage is **incomplete/BLOCKED**. Review the new allocation finding
and completed CPU evidence; there is no performance or correctness acceptance.
Read the full assembled CAMPAIGN.md, including its final recovery instruction,
before review. No GPU work, parent rebuild or model allocation is needed to
independently check this evidence.

## Review order

1. BLOCKER.md traces serialized values through the actual allocator, including
   image type, flags, pNext, format, tiling, usage, axes and creation before any
   operator rewrite. Compare its references to base `6178971845` and the preserved
   parent export. The original 8B finding stands; the common RoPE height conflict
   expands the blocker to all six cells.
2. Recompute offending extents/IDs/calls with `audit_allocations.py`, using the
   fixed models and saved matching query. Check `image_extents.cpp` and the
   existing-parent-linked helper output independently of that Python arithmetic.
   `allocate_memory=false` still calls `vkCreateImage`; shared storage does not
   defer image object creation. No known-invalid allocation was attempted.
3. Check runtime_constants.cpp and its output against preserved serialized
   metadata. It executes nine constant methods per model, 60 scalar outputs;
   all six runtime context lengths are 2560. No forward is loaded, and this
   executable links no Vulkan backend. Group/embedding remain serialized facts.
4. Compare results/selector-dispatch.csv to the CPU selector output and the pinned
   Xe2 manifest: nine profiles, 432 linear rows, 36 fallback attention rows,
   twelve names covered. Identity aliases match control; single-scheme profiles
   preserve the other scheme. Fused pair and softmax also match control.
   These are source selections, not observed GPU dispatches.
5. Check matrix.json and results/module-declarations.csv against parent module
   hashes/declarations. All twelve agree with source shared-byte totals. Parent
   binaries/libraries still match their five recorded hashes; unmodified golden
   check PASS (53). No candidate selector build was produced or measured.
6. Check the base-to-head diff boundaries and absence of shader/release/tool/
   golden/prompt/tolerance changes. No changed shader needs a synchronization
   diff audit. Unmodified check.sh --no-build PASS: release 1240, dev 1921 checks.

## Exact private evidence

Paths here are relative to the own artifact root named in CAMPAIGN.md; raw paths,
device identifiers and model data stay outside the source repository.

| Path | Evidence |
|---|---|
| preflight/recovery/allocation-audit.json | Offending serialized values, extents, shared-object IDs and consuming operator arguments for all six models |
| preflight/recovery/allocation-source-hashes.json | Twelve source files matched to base objects and preserved export |
| preflight/recovery/image-extents.out | Extents computed by the existing parent archive's pure functions |
| preflight/recovery/runtime-constants.csv | Sixty executed CPU constant outputs |
| preflight/recovery/runtime-constant-validation.json | Output equality and context validation |
| preflight/recovery/*-compile-command.json | Exact compiler argument arrays, including archive paths |
| preflight/recovery/*compile*.log | Helper compilation diagnostics, including the corrected missing-declaration attempt |
| preflight/recovery/dispatch-source-hashes.json | Exact CPU selector inputs |
| preflight/recovery/*-dispatch.csv and *-dispatch.stderr | Nine profile outputs and attention controls |
| preflight/recovery/dispatch-validation.json | Coverage, identities, changed selections and invariant attention controls |
| preflight/recovery/parent-golden.txt | Fresh check of unchanged shipped modules |
| preflight/recovery/check-no-build.out | Unmodified zone/twins/selector checks |
| preflight/recovery/provenance.json and SHA256SUMS | Helper/compiler/archive identity and raw evidence checksums |
| preflight/storage-properties-tiling.json | Preserved exact format/type/tiling/usage query; flags are zero in query_mma.py |
| preflight/parent-modules.json | Parent SPIR-V declaration audit |
| preflight/model-hashes.json and models.json | Original fixed input hashes and serialized graph metadata |
| build/parent-v2.src.txt and src/parent-v2.export-manifest | Pristine build source, recursive export, image/compiler and binary hashes |

There is no runs.csv because no run occurred. results/cell-coverage.csv explicitly
records six blocked cells with zero executed runs; no synthetic run is invented.
Missing GPU gates, next-token/decode checks, calibration, valid-run medians,
uncertainty and six-cell/family margins remain missing/BLOCKED. Source/kernel
checks cannot substitute for them.

After this single substantive review, park for the conforming-resolution dependency
recorded once in STATUS.md. Do not reopen the same blocker for another unchanged
review, query or preparation pass.
