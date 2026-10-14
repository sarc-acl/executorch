#!/usr/bin/env python3
"""probe_prompts.py <tokenizer.model> <out ids file> <out meta csv> <text file> [...]: the prompt set of the
near-tie evidence (owner decision 2026-10-04). Real text only: the given files are tokenised as the runner does
(no BOS) and concatenated; prompt i is a window of that token stream.
  prompt 0: the first file whole (the gate's unaligned prompt, to tie the set to the differing item);
  32 tile-aligned lengths 64, 128, .. 2048 (the lengths at which the SARC SDPA kernels are engaged);
  8 lengths that are not multiples of 16 (the path every device takes for an unaligned prompt).
Each window starts at a different offset, so the 41 prompts are 41 different texts and 41 different lengths.
The meta file keeps, per prompt, its length, offset and the token that follows it in the text (for perplexity)."""
import sys
from pytorch_tokenizers import get_tokenizer
t = get_tokenizer(sys.argv[1]); first = t.encode(open(sys.argv[4]).read(), bos=False, eos=False); stream = []
for f in sys.argv[4:]: stream += t.encode(open(f).read(), bos=False, eos=False)
lengths = [64 * k for k in range(1, 33)] + [57, 250, 411, 777, 1001, 1500, 1771, 1972]
rows = [(len(first), 0, stream[len(first)], first)]
for i, L in enumerate(lengths):
    off = (i * 173 + 31) % (len(stream) - L - 1); rows.append((L, off, stream[off + L], stream[off:off + L]))
with open(sys.argv[2], "w") as o, open(sys.argv[3], "w") as m:
    m.write("prompt,tokens,offset,next_token_in_text\n")
    for i, (L, off, nxt, ids) in enumerate(rows):
        o.write(" ".join(map(str, ids)) + "\n"); m.write(f"{i},{L},{off},{nxt}\n")
print(f"{len(rows)} prompts from {len(stream)} tokens of text")
