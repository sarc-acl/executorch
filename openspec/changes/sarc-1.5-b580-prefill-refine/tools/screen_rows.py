#!/usr/bin/env python3
"""screen_rows.py linear <raw dir> <scheme> <token> <rep> [--no-supersede]
screen_rows.py sdpa   <raw dir> <profile> <rep>        [--no-supersede]

The saved-row bookkeeping of screen.sh and screen_sdpa.sh: one CSV row per (configuration, repeat, shape),
appended once, never rewritten by a later run. Decides from the saved row keys whether one (configuration,
repeat) still has to be measured.

  exit 0   every expected shape of this (configuration, repeat) has its row. Rows that were missing were recovered
           from the validated cached result of that run (the JSON of a linear run, the log of an SDPA run);
           only keys that are not in the CSV are appended, so recovery can be repeated any number of times.
  exit 10  the run has to be made: there is no validated cached result. Whatever an earlier failed or
           interrupted attempt left (log, JSON, markers, and its partial CSV rows together with a copy of the
           CSV they were in) is first moved to <raw dir>/superseded/<reason>/, never overwritten.
           With --no-supersede nothing is moved or removed (used after a run, and by the recover-only mode).

CSV: linear rows.csv, key token,rep,model,op,storage,variant; SDPA screen.csv, key profile,rep,model,op,variant.
A CSV with a malformed or duplicated line, or whose last record is not terminated by a newline (an append
that was interrupted; such a record is dropped even if its field count fits), is copied to
superseded/csv-damaged-<utc>/ and rewritten with its trusted rows before anything else; the dropped record is
then recovered from the cached result like any other missing row.
A cached result is valid when the run is not marked interrupted (<base>.started without <base>.rc), its
status is one the suite returns for a completed run (0, or 1: the suite reports an unexpected dispatch that
way on this card, and the stock SDPA kernels of `base` likewise), the result is complete (JSON parses with a
positive time for every case; SDPA log of a run that ended with such a status) and its shapes are exactly
the screen's expected shapes (EXPECT_LIN / EXPECT_SDPA below, defined by the screen's command line, never
inferred from saved rows or results). The stock-kernel log of `base` ends with the suite's dispatch warning,
not with a geomean line, so no closing line is required.
Results saved before these markers existed have no .started and, for linear screens, no .rc: they are
judged by the completeness of the result alone, and their rc and temperature columns stay empty. An SDPA
result always needs its status marker."""
import csv, datetime, glob, io, json, os, shutil, sys

LIN = ("rows.csv", "token,rep,model,op,storage,variant,M,K,N,kernel,kernel_median_us,kernel_mean_us,kernel_cov,dispatch,rc,temp_c".split(","), 6)
SDPA = ("screen.csv", "profile,rep,model,op,variant,mean_us,stdev_us,dispatch,temp_c,rc".split(","), 5)
# The shapes of a screen, fixed by the command line the screen scripts run and by nothing that was saved:
# test_llama_microbench --linear --regime=prefill --storage=texture3d times the four linear shapes of the three
# models on the coopmat arm; --sdpa times QK^T, softmax, attn*V and their total for the three head
# configurations on the stock (tiled) and the selected (coopmat) arm. A result with any other shape set, and a
# CSV that lacks one of these keys, is incomplete, however much or little evidence exists beside it.
MODELS = ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b")
EXPECT_LIN = frozenset((m, o, "texture3d", "coopmat") for m in MODELS for o in ("wq_wo", "wk_wv", "w1_w3", "w2"))
EXPECT_SDPA = frozenset((m, o, v) for m in MODELS for o in ("qk", "softmax", "av", "total") for v in ("tiled", "coopmat"))
utc = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def line(fields):
    b = io.StringIO(); csv.writer(b, lineterminator="\n").writerow(fields); return b.getvalue()


def load(d, spec):
    """rows of the CSV by key; a damaged file is preserved under superseded/ and rewritten clean"""
    name, header, nk = spec; p = os.path.join(d, name)
    if not os.path.exists(p):
        open(p, "w").write(line(header)); return {}
    text = open(p, newline="").read(); rows = {}; bad = not text.endswith("\n")
    lines = text.split("\n")
    lines.pop()   # after the last newline: empty, or the unterminated record of an interrupted append, which is not
                  # trusted even when its field count happens to fit (a cut inside the last field looks whole)
    if not lines or next(csv.reader([lines[0]])) != header: bad = True
    for l in lines[1:]:
        try: f = next(csv.reader([l]))
        except Exception: f = []
        k = tuple(f[:nk])
        if len(f) != len(header) or k in rows: bad = True; continue
        rows[k] = f
    if bad:
        s = os.path.join(d, "superseded", f"csv-damaged-{utc()}"); os.makedirs(s, exist_ok=True); shutil.copy2(p, s)
        tmp = p + ".tmp"; open(tmp, "w").write(line(header) + "".join(line(f) for f in rows.values())); os.replace(tmp, p)
        print(f"{name}: damaged lines dropped, original kept in {s}")
    return rows


def marker(base):
    """(valid status, rc, temp) of a run from its .started / .rc markers; rc None = saved before the markers"""
    rcf = base + ".rc"
    if not os.path.exists(rcf): return (not os.path.exists(base + ".started")), None, ""
    f = open(rcf).read().split(); return True, (f[0] if f else "?"), (f[1] if len(f) > 1 else "")


def lin_cache(d, q, tok, rep):
    base = os.path.join(d, f"{q}-{tok}-r{rep}"); ok, rc, temp = marker(base)
    if not ok or rc not in (None, "0", "1"): return None
    try: cases = [c for c in json.load(open(base + ".json"))["cases"] if c.get("suite", "linear") == "linear"]
    except Exception: return None
    if not cases or any(not c.get("kernel_median_us", 0) > 0 for c in cases): return None
    return {(tok, rep, c["model"], c["op"], c["storage"], c["variant"]):
            [tok, rep, c["model"], c["op"], c["storage"], c["variant"], c["M"], c["K"], c["N"], c["kernel"], c["kernel_median_us"],
             c["kernel_mean_us"], c["kernel_cov"], c["dispatch"], rc or "", temp] for c in cases}


def sdpa_cache(d, prof, rep):
    log = os.path.join(d, f"{prof}-r{rep}.log"); ok, rc, temp = marker(log)
    if rc is None or rc not in ("0", "1") or (rc == "1" and prof != "base"): return None
    try: t = open(log, errors="replace").read()
    except OSError: return None
    out = {}
    for l in t.splitlines():
        f = l.split(",")   # RESULT,sdpa,<model>,<scheme>,<regime>,<op>,<K>,<N>,<mean_us>,<stdev_us>,-1,<dispatch>,SKIPPED,<kv>,<variant>
        if len(f) >= 15 and f[0] == "RESULT" and f[1] == "sdpa" and f[4] == "prefill":
            out[(prof, rep, f[2], f[5], f[14])] = [prof, rep, f[2], f[5], f[14], f[8], f[9], f[11], temp, rc]
    return out or None


def main():
    kind, d = sys.argv[1], sys.argv[2]; keep = "--no-supersede" in sys.argv
    if kind == "linear":
        spec = LIN; q, cfg, rep = sys.argv[3:6]; cache = lambda c, r: lin_cache(d, q, c, r)
        stem = f"{q}-{cfg}-r{rep}"; full = EXPECT_LIN
    else:
        spec = SDPA; cfg, rep = sys.argv[3:5]; cache = lambda c, r: sdpa_cache(d, c, r)
        stem = f"{cfg}-r{rep}"; full = EXPECT_SDPA
    name, header, nk = spec; rows = load(d, spec)
    mine = cache(cfg, rep)
    if mine and {k[2:] for k in mine} != full: mine = None    # a result with missing (or foreign) shapes is not a result
    have = {k[2:] for k in rows if k[:2] == (cfg, rep)}
    if have >= full: return 0
    if mine:
        new = [f for k, f in mine.items() if k not in rows]
        with open(os.path.join(d, name), "a") as f:
            f.write("".join(line(x) for x in new)); f.flush(); os.fsync(f.fileno())
        print(f"{stem}: {len(new)} row(s) recovered from the cached result, no GPU work"); return 0
    if not keep:
        old = [f for f in glob.glob(os.path.join(d, glob.escape(stem) + ".*")) if os.path.isfile(f)]
        if old or have:
            s = os.path.join(d, "superseded", f"interrupted-{stem}-{utc()}"); os.makedirs(s, exist_ok=True)
            for f in old: shutil.move(f, s)
            if have:   # partial rows of the failed attempt: kept with the CSV they were in, removed from the live CSV
                p = os.path.join(d, name); shutil.copy2(p, s)
                tmp = p + ".tmp"; open(tmp, "w").write(line(header) + "".join(line(f) for k, f in rows.items() if k[:2] != (cfg, rep))); os.replace(tmp, p)
            print(f"{stem}: earlier attempt without a valid result moved to {s} ({len(old)} file(s), {len(have)} partial row(s))")
    return 10


sys.exit(main())
