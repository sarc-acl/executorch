#!/usr/bin/env python3
"""test_near_tie.py: GPU-free replay of the recorded next-token flips through near_tie.py decide (MERGE-PLAN M1).

Reads only committed campaign results under openspec/changes/; the logits tables are rewritten into
logits_probe-style JSON files in a temporary directory and the CLI is run on them. Expected outcomes:
  ACCEPT      B580 (b580-refine0, s2-c0) and B70 (xe2-refine1, s2-c1), 8B 8da4w on prompt_check.txt: the fp32 CPU
              reference of that prompt (e2e benchmark, 1972 runner tokens, the same ids) is a near tie.
  REJECT      the Orin's orin-refine1 (3B maximum error larger than the parent's), with its own error tables,
              also when given near-tie logits; a synthetic wrong kernel; synthetic not-a-near-tie and
              flip-not-reproduced cases.
  ACCEPT      780M candidate 8, and the B580 and B70 candidates, 8B 8da4w on prompt_2048.txt, against the fp32 CPU
              reference measured 2026-10-15 (sarc/golden/ref-logits/8b-prompt_2048.json, top-2 margin 0.4776). The
              same 780M case without a reference file must fail closed.
  Merged-binary shape: SDPA error logs written in the output shape of a microbench built from 6178971845 (two fp32
              [sdpa-error] lines per case, a third fp64 one with ET_VK_SDPA_ERROR_REPORT=1 except for fused cases).
Exit 0 when every case comes out as expected."""
import json, os, re, shutil, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "near_tie.py")
CH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "openspec", "changes")
B580 = f"{CH}/sarc-1.5-b580-prefill-refine/results/b580"
B70 = f"{CH}/sarc-1.5-xe2-prefill-refine/results/xe2"
M780 = f"{CH}/sarc-1.5-780m-prefill-refine/results/780m"
ORIN = f"{CH}/sarc-1.5-orin-prefill-refine/results/orin"
CPU_REF_CHECK_8B = f"{CH}/sarc-1.5-e2e-benchmark/evidence/logits/cpu_fp32_runner.json"
T = tempfile.mkdtemp(prefix="near_tie_test.")


def write_json(name, pairs):
    p = os.path.join(T, name + ".json")
    json.dump({"top10": [[i, v] for i, v in pairs]}, open(p, "w"))
    return p


def differing_md(path, section, col):
    """Logits of one arm (P, C, PT, CT) from a probe_analysis.py differing.md section."""
    lines = open(path).read().split("\n## ")
    sec = next(s for s in lines if s.startswith(section))
    head = next(l for l in sec.splitlines() if l.startswith("| token id"))
    k = [c.strip().split(" ")[0] for c in head.strip("|").split("|")].index(col)
    out = []
    for l in sec.splitlines():
        if re.match(r"\| \d+ \|", l):
            c = [x.strip() for x in l.strip("|").split("|")]
            if c[k] != "not run":
                out.append((int(c[0]), float(c[k])))
    return out


def csv_780m(path, arm):
    rows = [l.split(",") for l in open(path) if re.match(r"\d+,", l)]
    head = next(l for l in open(path) if l.startswith("token_id")).strip().split(",")
    k = head.index("logit_" + arm)
    return [(int(r[0]), float(r[k])) for r in rows]


def decide(item, ref, old, new, olde, newe):
    r = subprocess.run([sys.executable, TOOL, "decide", "--item", item, "--ref", ref, "--old-logits", old,
                        "--new-logits", new, "--old-err", olde, "--new-err", newe,
                        "--json-out", os.path.join(T, re.sub(r"\W+", "_", item) + ".json")],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        print(r.stdout, r.stderr)
        raise SystemExit(f"near_tie.py failed on {item}")
    return r.returncode, r.stdout


fails = 0


def case(label, expect, item, ref, old, new, olde, newe, must=()):
    """expect: ACCEPT, REJECT or NO-REF. must: regexes that the output has to contain."""
    global fails
    rc, out = decide(item, ref, old, new, olde, newe)
    got = "ACCEPT" if rc == 0 else "REJECT"
    ok = got == ("ACCEPT" if expect == "ACCEPT" else "REJECT") and all(re.search(m, out) for m in must)
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} [{expect}] {label}\n{out}")


B580_ERR, B70_ERR = f"{B580}/sdpa-error/c0-full.csv", f"{B70}/sdpa-error"
NOREF = os.path.join(T, "no-reference-recorded.json")  # deliberately absent
REF_2048_8B = os.path.join(os.path.dirname(HERE), "golden", "ref-logits", "8b-prompt_2048.json")
EXPECT_2048 = "ACCEPT"  # the outcome measured 2026-10-15 (margin 0.4776); the three builds' tokens rank 56646 to 128101 in that reference

# --- the real flips, 8B 8da4w, prompt_check.txt (gate-check1972): parent default -> candidate default ---------------
md = f"{B580}/probe/s2-c0/differing.md"
case("B580 b580-refine0 (s2-c0), 8B 8da4w prompt_check.txt", "ACCEPT", "b580 s2-c0 8b 8da4w prompt_check: parent vs candidate",
     CPU_REF_CHECK_8B, write_json("b580-check-P", differing_md(md, "8b-8da4w gate-check1972", "P")),
     write_json("b580-check-C", differing_md(md, "8b-8da4w gate-check1972", "C")),
     B580_ERR + "#parent", B580_ERR + "#b580-refine0", must=[r"margin 0\.3900", r"old top-1 id 45647", r"new top-1 id 70159"])
md = f"{B70}/probe/s2-c1/differing.md"
case("B70 xe2-refine1 (s2-c1), 8B 8da4w prompt_check.txt", "ACCEPT", "b70 s2-c1 8b 8da4w prompt_check: parent vs candidate",
     CPU_REF_CHECK_8B, write_json("b70-check-P", differing_md(md, "8b-8da4w gate-check1972", "P")),
     write_json("b70-check-C", differing_md(md, "8b-8da4w gate-check1972", "C")),
     f"{B70_ERR}/parent-full.txt", f"{B70_ERR}/refine1-full.txt", must=[r"margin 0\.3900", r"new mismatches 0"])

# --- the real flips on prompt_2048.txt (2048 x " the", ids 2048 x 279), against the fp32 CPU reference measured
#     2026-10-15 with cpu_ref_logits.py (sarc/golden/ref-logits/8b-prompt_2048.json) -------------------------------------
c = f"{M780}/probe/c8-differ-8b-8da4w-prompt_2048.csv"
case("780M candidate 8 (c8-fused3), 8B 8da4w prompt_2048.txt", EXPECT_2048,
     "780m c8 8b 8da4w prompt_2048: parent vs candidate", REF_2048_8B,
     write_json("780m-2048-P", csv_780m(c, "parent-default")), write_json("780m-2048-C", csv_780m(c, "cand-default")),
     f"{M780}/sdpa-error/c8-fused3.csv#parent", f"{M780}/sdpa-error/c8-fused3.csv#cand",
     must=[r"margin 0\.4776", r"old top-1 id 118", r"new top-1 id 53"])
for dev, md, olde, newe in (("B580 b580-refine0", f"{B580}/probe/s2-c0/differing.md", B580_ERR + "#parent", B580_ERR + "#b580-refine0"),
                            ("B70 xe2-refine1", f"{B70}/probe/s2-c1/differing.md", f"{B70_ERR}/parent-full.txt", f"{B70_ERR}/refine1-full.txt")):
    tag = dev.split()[0].lower()
    case(f"{dev}, 8B 8da4w prompt_2048.txt", EXPECT_2048, f"{tag} 8b 8da4w prompt_2048: parent vs candidate",
         REF_2048_8B, write_json(f"{tag}-2048-P", differing_md(md, "8b-8da4w gate-the2048", "P")),
         write_json(f"{tag}-2048-C", differing_md(md, "8b-8da4w gate-the2048", "C")), olde, newe,
         must=[r"margin 0\.4776", r"old top-1 id 247", r"new top-1 id 53"])
case("780M c8 without any reference file: fails closed", "REJECT", "780m c8, reference absent", NOREF,
     os.path.join(T, "780m-2048-P.json"), os.path.join(T, "780m-2048-C.json"),
     f"{M780}/sdpa-error/c8-fused3.csv#parent", f"{M780}/sdpa-error/c8-fused3.csv#cand", must=[r"reason: no fp32 CPU reference"])

# --- the Orin candidate rejected for a larger maximum error on 3B --------------------------------------------------
oe, ne = f"{ORIN}/sdpa-error1/stock.txt", f"{ORIN}/sdpa-error1/refine1.txt"
case("Orin orin-refine1 (s2-c1), 1B 8da4w unaligned: recorded error tables, no logits recorded", "REJECT",
     "orin s2-c1 1b 8da4w unaligned: default vs tiled", NOREF, os.path.join(T, "absent-old.json"), os.path.join(T, "absent-new.json"),
     oe, ne, must=[r"3b_head_config_s2048: maximum error larger"])
case("Orin orin-refine1 error tables with near-tie logits borrowed from the B580 case (error rule alone)", "REJECT",
     "orin refine1 errors + near-tie logits", CPU_REF_CHECK_8B, os.path.join(T, "b580-check-P.json"), os.path.join(T, "b580-check-C.json"),
     oe, ne, must=[r"reason: 3b_head_config_s2048: maximum error larger than the old arm's \(0\.00157 > 0\.001408\)"])

# --- the output shape of a test_llama_microbench built from dev/1.5 6178971845 ---------------------------------------
# Every case prints two fp32 [sdpa-error] lines (l. 2271 `max_abs=`, %g; l. 2326 `elements=`, %.4e), with
# ET_VK_SDPA_ERROR_REPORT=1 a third (l. 2395 `n=`, fp64 reference) except for 4070 Ti-style fused cases, whose
# verdict block (l. 2341) returns first; then [sdpa-kernels] and one [sdpa-correctness] line. No such log is
# committed (it needs a GPU), so the logs below are written in that shape from the Orin's recorded numbers
# (sdpa-error1/stock.txt and refine1.txt): the reader must pick one format, not count lines.
def orin_vals(path):
    v = {}
    for l in open(path):
        m = re.match(r"\[sdpa-error\] (\S+) max_abs=(\S+) rms=(\S+)", l)
        if m:
            v[m[1]] = (float(m[3]), float(m[2]))
    return v


def merged_log(name, vals, fused=False, report=False, passes=1, skew_b=1.0):
    q = {"1b": (32, 8, 64), "3b": (24, 8, 128), "8b": (32, 8, 128)}
    out = ["DEVICE,stub (shape of a 6178971845 binary)"]
    for _ in range(passes):
        for c, (rms, mx) in vals.items():
            qh, kvh, d = q.get(c[:2], (2, 1, 64))
            n = 2048 * qh * d
            out.append(f"[sdpa-error] {c} max_abs={mx:g} rms={rms:g} max_abs_ref=0.998757")
            out.append(f"[sdpa-error] {c} elements={n} rms_err={rms * skew_b:.4e} max_abs_err={mx:.4e} ref_rms=3.7698e-02")
            corr = f"[sdpa-correctness] {c} S=2048 input_pos=0 D={d} Q_H={qh} KV_H={kvh} qk_coopmat=yes av_coopmat=yes mismatches=0/{n} PASSED"
            if fused:
                out.append(f"[sdpa-kernels] {c} qk=? softmax=? av=? fused=sarc_sdpa_fused_x no_mask_fill=no pairing=ok")
                out.append(corr.replace("qk_coopmat=yes av_coopmat=yes", "qk_coopmat=NO av_coopmat=NO"))
            else:
                if report:  # fp64 reference line: deliberately different values, it must not be read
                    out.append(f"[sdpa-error] {c} n={n} rms={rms / 2:.6e} max={mx / 2:.6e} ref_rms=3.769800e-02")
                out.append(f"[sdpa-kernels] {c} qk=sarc_sdpa_qk_x softmax=sarc_sdpa_attn_weights_softmax av=sarc_sdpa_av_x fused=? no_mask_fill=yes pairing=ok")
                out.append(corr)
        out.append("[sdpa-correctness] tier=full cases_run=4")
    p = os.path.join(T, name + ".log")
    open(p, "w").write("\n".join(out) + "\n")
    return p


ov, nv = orin_vals(oe), orin_vals(ne)
case("merged-binary shape: Orin numbers, old with ET_VK_SDPA_ERROR_REPORT=1 (3 lines), new fused (2 lines), 3 passes",
     "REJECT", "merged shape, orin refine1 numbers", CPU_REF_CHECK_8B, os.path.join(T, "b580-check-P.json"),
     os.path.join(T, "b580-check-C.json"), merged_log("m-old", ov, report=True), merged_log("m-new", nv, fused=True, passes=3),
     must=[r"format A", r"reason: 3b_head_config_s2048: maximum error larger than the old arm's \(0\.00157 > 0\.001408\)",
           r"1b_head_config_s2048 \(fp32 ref\): rms 8\.547e-05 -> 3\.485e-05"])
case("merged-binary shape: same numbers in both arms (not larger), new fused", "ACCEPT", "merged shape, equal errors",
     CPU_REF_CHECK_8B, os.path.join(T, "b580-check-P.json"), os.path.join(T, "b580-check-C.json"),
     merged_log("m-eq-old", ov, report=True), merged_log("m-eq-new", ov, fused=True), must=[r"format A"])
case("merged-binary shape: the two fp32 lines of one log disagree", "REJECT", "merged shape, A/B disagree",
     CPU_REF_CHECK_8B, os.path.join(T, "b580-check-P.json"), os.path.join(T, "b580-check-C.json"),
     merged_log("m-dis-old", ov), merged_log("m-dis-new", ov, skew_b=1.5), must=[r"two fp32 \[sdpa-error\] lines disagree on rms"])

# --- synthetic -----------------------------------------------------------------------------------------------------
def f32(name, vals):
    p = os.path.join(T, name + ".f32")
    open(p, "wb").write(struct.pack(f"<{len(vals)}f", *vals))
    return p


def log(name, errs, mis=0):
    p = os.path.join(T, name + ".log")
    open(p, "w").write("".join(f"[sdpa-error] {c} max_abs={m} rms={r} max_abs_ref=1.0\n"
                               f"[sdpa-correctness] {c} S=2048 input_pos=0 mismatches={mis}/100 PASSED\n" for c, (r, m) in errs.items()))
    return p


P = ("1b_head_config_s2048", "3b_head_config_s2048", "8b_head_config_s2048")
good = log("old-good", {c: (1e-4, 1e-3) for c in P})
case("synthetic wrong kernel: clear reference winner, errors x10, mismatches", "REJECT", "synthetic wrong kernel",
     f32("ref-clear", [0.0, 12.0, 9.0, 1.0]), f32("old-a", [0.0, 12.0, 9.0, 1.0]), f32("new-a", [0.0, 7.0, 9.5, 1.0]),
     good, log("new-bad", {c: (1e-3, 1e-2) for c in P}, mis=37),
     must=[r"not a near tie: reference top-2 margin 3\.0000", r"rms error larger", r"maximum error larger", r"37 mismatches"])
case("synthetic: near tie in the reference, but the 8B rms error is larger", "REJECT", "synthetic error larger",
     f32("ref-tie", [0.0, 10.2, 10.0, 1.0]), f32("old-b", [0.0, 10.1, 10.0, 1.0]), f32("new-b", [0.0, 10.0, 10.1, 1.0]),
     good, log("new-8b", {**{c: (1e-4, 1e-3) for c in P}, "8b_head_config_s2048": (1.01e-4, 1e-3)}),
     must=[r"8b_head_config_s2048: rms error larger"])
case("synthetic: near tie, errors equal, but the two arms' logits pick the same token", "REJECT", "synthetic not reproduced",
     os.path.join(T, "ref-tie.f32"), f32("old-c", [0.0, 10.2, 10.0, 1.0]), f32("new-c", [0.0, 10.1, 10.0, 1.0]), good, good,
     must=[r"do not reproduce the flip"])
case("synthetic: margin exactly at the threshold (0.5) is not below it", "REJECT", "synthetic margin 0.5",
     f32("ref-half", [0.0, 10.5, 10.0]), f32("old-d", [0.0, 10.1, 10.0]), f32("new-d", [0.0, 10.0, 10.1]), good, good,
     must=[r"not a near tie: reference top-2 margin 0\.5000"])
case("synthetic: near tie, errors equal (not larger), full-vector .f32 logits", "ACCEPT", "synthetic accepted",
     os.path.join(T, "ref-tie.f32"), os.path.join(T, "old-b.f32"), os.path.join(T, "new-b.f32"), good, good,
     must=[r"margin 0\.2000"])

if not fails:
    shutil.rmtree(T)
print(f"test_near_tie: {'PASS' if not fails else f'FAIL ({fails}), scratch kept in {T}'}")
sys.exit(1 if fails else 0)
