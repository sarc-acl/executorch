#!/usr/bin/env python3
"""nexttoken.py: compare the next token of parent and candidate for one prompt from their llama_main logs.

  nexttoken.py <prompt file> <expected prompt tokens> <parent log> <cand log> <parent rc> <cand rc> [--timed]

Prints one CSV row (FIELDS without model and scheme). The verdict is SAME or DIFFER only when both runs are usable:
rc 0, a PyTorchObserver line, the expected number of prompt tokens, a non-empty generated text that begins with
the prompt itself, and a non-empty token after it. Anything else is INVALID:<reasons>; two empty or failed
outputs are never SAME. The generated text is the log without runtime lines (the filter of sarc/tools/verify.sh).
--timed is for the timed prefill logs (--warmup): the prompt echo is not required there, the texts are compared
whole and the token fields stay empty."""
import hashlib, json, re, sys
FIELDS = ["model", "scheme", "prompt", "prompt_sha256", "expected_tokens", "parent_rc", "cand_rc", "parent_prompt_tokens",
          "cand_prompt_tokens", "parent_token_hex", "cand_token_hex", "parent_out_sha256", "cand_out_sha256", "verdict"]
RUNTIME = re.compile(rb"PyTorchObserver|^[IWE] |^\[sarc_dev\]")

def generated(log):
    try: return b"".join(l for l in open(log, "rb").read().splitlines(keepends=True) if not RUNTIME.search(l))
    except OSError: return b""

def observer(log):
    obs = None
    try:
        for line in open(log, errors="replace"):
            i = line.find("PyTorchObserver")
            if i >= 0:
                try: obs = json.loads(line[line.index("{", i):])
                except ValueError: pass
    except OSError: pass
    return obs

def one(prompt, want, log, rc, timed):
    """-> (prompt_tokens, token bytes or None, sha256 of the generated text, reasons)"""
    why = []; out = generated(log); obs = observer(log); pt = "" if obs is None else obs.get("prompt_tokens", "")
    if str(rc) != "0": why.append("rc")
    if obs is None: why.append("no_stats")
    elif str(pt) != str(want): why.append("prompt_tokens")
    if not out.strip(): why.append("empty_output")
    tok = None
    if not timed and out.strip():
        if out.startswith(prompt):
            tok = out[len(prompt):]
            if tok.endswith(b"\n"): tok = tok[:-1]
            if not tok: why.append("no_token")
        else: why.append("prompt_not_echoed")
    return pt, tok, hashlib.sha256(out).hexdigest(), why

def evaluate(prompt_path, want, plog, clog, prc, crc, timed=False):
    prompt = open(prompt_path, "rb").read()
    ppt, ptok, psha, pwhy = one(prompt, want, plog, prc, timed)
    cpt, ctok, csha, cwhy = one(prompt, want, clog, crc, timed)
    why = [f"parent:{w}" for w in pwhy] + [f"cand:{w}" for w in cwhy]
    if why: verdict = "INVALID:" + "+".join(why)
    else: verdict = "SAME" if psha == csha and ptok == ctok else "DIFFER"
    hx = lambda t: "" if t is None else t.hex()
    return dict(prompt=prompt_path.rsplit("/", 1)[-1], prompt_sha256=hashlib.sha256(prompt).hexdigest(), expected_tokens=str(want),
                parent_rc=str(prc), cand_rc=str(crc), parent_prompt_tokens=str(ppt), cand_prompt_tokens=str(cpt),
                parent_token_hex=hx(ptok), cand_token_hex=hx(ctok), parent_out_sha256=psha, cand_out_sha256=csha, verdict=verdict)

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--timed"]
    r = evaluate(a[0], a[1], a[2], a[3], a[4], a[5], "--timed" in sys.argv)
    print(",".join(r[k] for k in FIELDS[2:]))
