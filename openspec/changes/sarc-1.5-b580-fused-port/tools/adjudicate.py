#!/usr/bin/env python3
"""adjudicate.py: which rows of a session's runs.csv an analysis may count. runs.csv is never rewritten.
countable(session, rows) returns (rows that may count as valid timed runs, problems):
  - a row keyed in tools/adjudication.csv (session, log, rep, slot) takes the adjudicated validity;
  - a timed row stored valid=1 whose fbusy_pct is not a share (empty, or outside 0 to 100 %) and that has no
    adjudication row is not counted either and is returned as a problem: the analysis must then fail, and a
    person adds the keyed row. Nothing here makes an invalid run valid.
adjudicate.py <stage/<session>/raw> prints the stored and the adjudicated counts."""
import csv, os, sys
HERE = os.path.dirname(os.path.realpath(__file__))
ADJ = {(r["session"], r["log"], r["rep"], r["slot"]): r for r in csv.DictReader(open(os.path.join(HERE, "adjudication.csv")))}
def session_of(raw_dir): return os.path.basename(os.path.dirname(os.path.realpath(raw_dir)))
def countable(session, rows):
    ok, problems = [], []
    for r in rows:
        if not r["log"].startswith("logs/prefill"): continue
        a = ADJ.get((session, r["log"], r["rep"], r["slot"]))
        if a:
            if a["adjudicated_valid"] == "1" and r["valid"] != "1": problems.append(f'{r["log"]}: adjudication may not validate a run stored invalid')
            elif a["adjudicated_valid"] == "1": ok.append(r)
            continue
        if r["valid"] != "1": continue
        try: share = 0.0 <= float(r["fbusy_pct"]) <= 100.0
        except ValueError: share = False
        if share: ok.append(r)
        else: problems.append(f'{r["log"]} rep {r["rep"]} slot {r["slot"]}: stored valid=1 with fbusy_pct={r["fbusy_pct"]!r}, no adjudication row')
    return ok, problems
if __name__ == "__main__":
    d = sys.argv[1]; rows = list(csv.DictReader(open(os.path.join(d, "runs.csv")))); s = session_of(d)
    timed = [r for r in rows if r["log"].startswith("logs/prefill")]; ok, problems = countable(s, rows)
    print(f"session {s}: rows {len(rows)}, timed {len(timed)}, stored valid {sum(r['valid'] == '1' for r in timed)}, countable {len(ok)}, adjudicated rows {sum(1 for k in ADJ if k[0] == s)}, problems {len(problems)}")
    for p in problems: print("PROBLEM", p)
    sys.exit(1 if problems else 0)
