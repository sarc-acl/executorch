# M51 capability record — summary

From the `device_coopmat_properties` check run on the campaign board, before any
kernel dispatch (task 1.3). This committed summary carries shapes, types, scope
and subgroup sizes only; the raw record with device identifiers stays local.

## Enumerated cooperative-matrix properties

| M×N×K | A | B | C / result | scope | saturating |
|---|---|---|---|---|---|
| 16×16×16 | float16 | float16 | float32 | subgroup | no |
| 16×16×16 | float16 | float16 | float16 | subgroup | no |
| 16×16×16 | uint8 | uint8 | sint32 | subgroup | no |
| 16×16×16 | sint8 | sint8 | sint32 | subgroup | no |
| 16×16×16 | uint8 | sint8 | sint32 | subgroup | no |
| 16×16×16 | sint8 | uint8 | sint32 | subgroup | no |
| 16×16×16 | uint8 | uint8 | sint32 | subgroup | yes |
| 16×16×16 | sint8 | sint8 | sint32 | subgroup | yes |
| 16×16×16 | uint8 | sint8 | sint32 | subgroup | yes |
| 16×16×16 | sint8 | uint8 | sint32 | subgroup | yes |

## Subgroup sizes

- Supported range: 32–64; required-size control for compute is supported.
- The campaign's variants use required subgroup size 32, the device default is 64.

## Stop-condition evaluation (SPEC, "M51 capability record gates the campaign")

- A 16×16×16 FP16 property with an FP32 result and one with an FP16 result is
  enumerated at subgroup scope: met.
- Required subgroup size 32 for compute: met.
- Vulkan version at least 1.3: met.

The campaign continues; the AMD family and the `m51` family are admissible.
