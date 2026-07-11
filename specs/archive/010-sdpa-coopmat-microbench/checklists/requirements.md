# Specification Quality Checklist: SDPA Coopmat Correctness + Microbenchmark

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-05
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- Grounded directly in this workstream's own history: the SDPA coopmat
  shaders/dispatch code already sitting in the working tree (imported from
  `yanwen/quant-dev-active` in a prior session), `007`'s precedent for the
  equivalent linear-coopmat correctness+microbenchmark question, and `003`'s
  existing classification of SDPA as "no WMMA implementation exists" (now
  partially addressed by the import, prefill only). This feature's scope is
  the natural next step (verify and measure what already exists), not an
  inference.
- "Technical" terminology in this spec (shader/file names, SPIR-V,
  kernel-dispatch, tile alignment) matches the established style of every
  prior spec in this workstream (001-009): the actual stakeholder is the
  contributor themselves, doing GPU kernel performance engineering, not a
  non-technical business audience. Consistent with precedent, not a
  deviation.
- One scope-defining judgment call was resolved as a documented Assumption
  rather than a [NEEDS CLARIFICATION] marker: SDPA's shape/dispatch
  behavior is independent of the `4w`/`8da4w` quantization scheme (verified
  by reading `SDPA.cpp`'s `sdpa_buf_half`/`sdpa_coopmat_device_ok` gates
  directly, not guessed), so this feature measures one configuration per
  target model (three total) rather than the constitution's default six --
  matching `007`'s own precedent of documenting a deliberate scope
  narrowing (its `lm_head` exclusion) rather than treating the constitution's
  default as unconditionally binding.
- All items pass on first validation pass; no spec revisions were required.
