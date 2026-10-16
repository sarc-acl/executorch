# Partial RoPE bundle applicability on B70 originals

The detached CPU audit completed at 2026-10-15T04:47:58.312599+00:00 with exit code 0.
Queued-to-terminal wall time was 112.5 seconds. No GPU/model method ran,
no PTE was created and no input admission or gate decision was made.

All six original full-file hashes and virtual revised hashes match the delivered
B580 evidence. All 3,313 serialized Vulkan constant records match, including
payload hash, shape, dtype and staging dtype. The recipe changes two existing
RoPE layout descriptor bytes per model (12 total); virtual outputs preserve
all other bytes. Source files total 14,066,296,448 bytes. Per-model hashes and counts
are in partial-rope-audit.csv.

Serialized constant-method values agree with the delivered runtime observations,
including context2560. These are local serialized-metadata checks against B580's
reported runtime values. Local runtime execution of revised files was not performed:
no revised files were materialized. Tokenizer-copy correctness and full resource
legality of revised models are not established by this audit.

Recipe source commit: `8c6c59c0ff922bd437d78fea7ec7c08dba1815fb`.
Local audit source commit: `9b3375c76`.
Delivered archive SHA256: `ba113d7778555427639cea0da7444bdfa018791428fd1e218e8ebdc3f343005b`.

Private artifact evidence, relative to the assigned artifact root:

| File | SHA256 |
|---|---|
| jobs/rope-audit-01.json | fe222dd8d0b57af527441cdf9949b850f7900c13ff9ec907975ddc5734449577 |
| logs/rope-audit-01.log | 2a62bbca7b1cd449c4db4045f1a6c482403891e272f46a16e629693b724ff2c5 |
| input-preparation/b70-rope-audit-01/evidence.json | 94f784c658f787d589c34a4229576f4fc2482358bb493f6ab39b9022e312a22c |
| input-preparation/b580-rope-bundle-receipt.json | 8ce9b362143724150fa14d7e2445e870e957c193ca40509367f7bfe7c15d6d3c |

The 8B width, operator-temporary and independently generated synthetic-gate
resource blockers remain unresolved. Whole-model queried legality, revised-file
runtime checks, GPU correctness, parent snapshot, A/A and the six-cell matrix
remain incomplete. RESOURCE_LIMIT_BLOCKED is preserved. This result confirms
CPU recipe applicability only; it is not six conforming inputs or a gate pass.
No source, valid build or original model was replaced. No GPU job, push, PR,
service, driver or hardware-policy change occurred. The existing owner dependency
in STATUS.md remains pending; independent work is finished and the run is parked.
