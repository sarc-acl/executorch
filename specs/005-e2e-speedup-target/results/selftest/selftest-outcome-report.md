# SYNTHETIC SELF-TEST DATA — NOT A REAL MEASUREMENT

Every entry below is constructed synthetic data (research.md Decision 3), proving the verdict engine's logic before it is ever pointed at a real re-measurement. Do not read any number here as a real optimization result.

## Summary

| Model | Scheme | Verdict | Observed multiplier | Combined e2e change | Synthetic |
|---|---|---|---:|---:|---|
| llama-3.1-8b | 4w | **met** | 2.00x | +4.9% | True |
| llama-3.1-8b | 8da4w | **exceeded** | 2.60x | +5.0% | True |
| llama-3.2-3b | 4w | **missed** | 1.40x | +2.5% | True |
| llama-3.2-3b | 8da4w | **regressed** | 0.80x | -1.9% | True |
| llama-3.2-1b | 4w | **not_comparable** | n/a | n/a | True |
| llama-3.2-1b | 8da4w | **met** | 2.00x | +4.0% | True |

## Detail

### llama-3.1-8b / 4w

- verdict: **met**
- observed prefill multiplier: 2.0000x
- combined e2e change (tracked, not a pass/fail bar): +4.9%

### llama-3.1-8b / 8da4w

- verdict: **exceeded**
- observed prefill multiplier: 2.6000x
- combined e2e change (tracked, not a pass/fail bar): +5.0%

### llama-3.2-3b / 4w

- verdict: **missed**
- observed prefill multiplier: 1.4000x
- combined e2e change (tracked, not a pass/fail bar): +2.5%

### llama-3.2-3b / 8da4w

- verdict: **regressed**
- observed prefill multiplier: 0.8000x
- combined e2e change (tracked, not a pass/fail bar): -1.9%

### llama-3.2-1b / 4w

- verdict: **not_comparable**
- not directly comparable: prefill workload was 1024 tokens, not the required 2048

### llama-3.2-1b / 8da4w

- verdict: **met**
- observed prefill multiplier: 2.0000x
- combined e2e change (tracked, not a pass/fail bar): +4.0%
