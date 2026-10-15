# Revised Intel input admission

The owner decision of 2026-10-15 03:57 UTC authorizes conforming versions of all
six fixed inputs. B580 owns the common recipe/evidence bundle. This run performs
no duplicate export investigation. The bundle is absent at this checkpoint.
All original artifacts and the resource blocker are preserved.

Before admitting revised inputs for any model GPU job, the B70 actor will check:

1. The versioned recipe, source/export commits, compiler/environment and hashes
   of every original and revised model. Preserve originals and retain separate
   locations and dataset IDs for the revised files.
2. Logical tensor constant equivalence, including quantized weights, scales,
   zero points/corrections, embeddings and other constants. Changed container
   bytes or storage layouts require logical comparison, not file-hash equality.
   Quantization, context2560, tokenizers and prompts retain the fixed semantics.
3. Runtime constant methods with the preserved CPU-only exported helper on all
   six revised files; serialized group128 linear and group32 embedding metadata.
4. Resource extents, formats, types, tiling, usage, samples, layers and allocation
   sizes against matching B70 queries. Cover revised graph resources and runtime
   generated/prepacked resources. Reuse existing queries when still applicable;
   the existence of a historical successful run is not a legality proof.
5. Selection against the twelve reviewed inventory entries and seven profiles.
   The original texture3d coverage map remains historical input context. A revised
   buffer or split layout does not prove that the same kernels are dispatched.
   Preserve shape predicates and attention setup and record any unreachable
   candidates explicitly; do not claim missing coverage passed.
6. Resource legality for the unchanged full gate and identity helper. Correctness
   on the exact preserved binaries, all prescribed cells and storage modes, and
   next-token comparisons must precede candidate timing. No gate relaxation,
   hardware-limit waiver, new shader body or runtime release-zone edit is allowed.

## Synthetic fixture dependency

The gate constructs production cases independently of the PTE files:

| Exact-export source | Behavior |
|---|---|
| sarc/tools/verify.sh:65 | Linear microbench for both schemes |
| sarc/tools/verify.sh:106 | Three-model/two-storage production-diff loop |
| backends/vulkan/test/sarc_dev/test_llama_microbench.cpp:492 | 8B w1_w3 K=4096,N=14336 and w2 K=14336,N=4096 |
| backends/vulkan/test/sarc_dev/test_llama_microbench.cpp:227 | Half texture3d input/output with explicit width packing |
| backends/vulkan/test/sarc_dev/test_llama_microbench.cpp:969 | Production-diff constructs M=2048,batch=1 fixtures |
| backends/vulkan/test/sarc_dev/utils.cpp:1492 | Input tensors retain the fixture storage and layout |
| backends/vulkan/test/sarc_dev/utils.cpp:1512 | Output tensors retain the fixture storage and layout |
| tools/linear_identity/main.cpp:19 | Same production shapes in both storage modes |

For texture3d, the w1_w3 output and w2 input require width 14336/4=3584.
The existing matching B70 query reports 2048. Therefore a PTE-only revision
cannot establish the full gate's resource legality. This finding is source-only;
no illegal fixture, allocation probe or GPU command was executed. The common
bundle/owner relay must address this dependency without weakening the gate.
The exact exported source commit is `2fcc1c7e96a7774d80b66095aadb79700d3b48e3`.
Private derivation and source hashes are in preflight/gate-resource-audit.json.

## Queue admission and provenance

After evaluating real evidence, the actor creates the artifact-local
preflight/revised-input-admission.json with status READY_FOR_CORRECTNESS,
a dataset_id beginning revised-, distinct model_root/original_model_root,
and files mapping the existing six model and three tokenizer relative paths
to size_bytes and sha256. The evidence map contains absolute path/sha256
records for recipe, logical_constant_equivalence, runtime_metadata,
resource_legality, selector_coverage and gate_resource_legality.
This is a B70 admission record, not a schema imposed on B580's external bundle.
No such real admission record exists yet. Hash validation cannot establish the
truth of an evidence report; the actor must evaluate it before creating admission.
READY_FOR_CORRECTNESS does not mean a correctness gate or timing check passed.

Both prepared queues check admission before launch. Stages preserve INPUTS.json;
the gate refuses a different receipt for an existing stage. Timing and dispatch
collection verify the staged input identity against the configured common root.
Central rows and evidence JSON identify revised dataset, receipt and model hashes,
so revised-input numbers cannot be mistaken for original-input measurements.
Historical original numbers remain context; the candidate margin still uses
same-session revised-input controls and the preregistered limits are unchanged.

CPU-only fixtures cover missing evidence/models, original-root misuse, changed
file hashes, unadmitted status, and all 61 synthetic aggregate rows' provenance.
These tests contain no real models and are not device or correctness evidence.
They are saved under preflight/revised-input-fixture in the artifact root.

Gate source audit SHA256: `acf1bd491065caded39f363d32cc93f2783fa7cc5b0ea440c76058fd0fb585a6`.
