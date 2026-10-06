# STATUS: RX 7600 prefill campaign

Updated 2026-10-06 07:30 UTC. Artifacts: `/local/yanwen.xu/campaign-rx7600/.artifacts/`.

## Running now

- Native build of the parent `f5f1bf10c` (tag `parent`; main build done 07:19 UTC, traced build running).

## State

| step | state |
|---|---|
| change directory, tools, thresholds | written (`proposal.md`, `tools/thresholds.txt`), committed before any measurement |
| parent build `parent` | main build done, traced build running |
| `s0-parent-verify` | next |
| baseline + A/A | after the snapshot |

## Findings so far (host)

- The model copy at `/local/yanwen.xu/campaign-rx7600/models` is incomplete: 8B 4w 3,263,430,656 of 4,173,751,424
  bytes (sha256 `be58a01b...`, manifest `695dd232...`), 8B 8da4w missing; no copy was running at 06:49 UTC. The
  complete 2026-09-28 copies (all six sha256 equal to the manifest) are used instead, read-only, through links in
  `.artifacts/models`. The incomplete directory was left as found.
- `podman` cannot run here (owner fact); builds are native, so the shipped-SPIR-V golden check is pending for every
  build of this campaign.
- The M51 campaign builds on this host's CPU (seen 07:00 UTC: Android NDK `clang++`, `cmake --build -j8`). Timed
  runs wait until no compiler or linker of anyone runs and are invalid if one appears during the run.
- No compositor or other process holds `/dev/dri` at 07:00 UTC (both DisplayPort connectors are connected).
- The unaligned 1304-token prompt `r1304.txt` that `verify.sh` also uses (`ls r*.txt`) is not on this host; its
  `unaligned` lines are absent from every `verify.sh` output here, the parent snapshot included. The unaligned
  1972-token `prompt_check.txt` (`check` lines) is present.
- Python imports from `/tool/pkg` (NFS) are slow: importing torch took over 5 minutes under load, so the ETDump
  analysis will not use the kit's `Inspector` script.

## Decision needed from the owner

**Order of port items 1 and 2.** Item 1, the fused attention kernel, replaces QK^T, softmax and attn*V for every
tile-aligned prefill call, including the timed 2048-token prompt. Item 2, the softmax without the zero tail, then
runs only where the fused kernel does not (unaligned prompts). On the timed prompt it would measure about 0 %. That
gated candidate would count toward the stop rule (two consecutive candidates under 2 %), and the stop rule could then
end the campaign before items 3 and 4. The 780M measured them the other way round: softmax as candidate 7
(+4.10 %), fused kernel as candidate 8. **Default unless the owner says otherwise:** gate the softmax first
(candidate 1, on the parent's three-kernel path, where it is measurable), then the fused kernel on top
(candidate 2), then items 3 and 4.

## Next

1. `s0-parent-verify` on the parent build.
2. Baseline + A/A session; write the clock floor into `tools/thresholds.txt`.
3. Coordinator-hold test (`HOLD-TEST`), recorded here.
4. Locate (ETDump families for the six cells).
