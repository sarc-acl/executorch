"""Run one unchanged-profile control gate, then the bounded A/A calibration."""

import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


tools = Path(__file__).resolve().parent
artifacts = Path(os.environ["WMMA_CAMPAIGN_ROOT"]) / ".artifacts"
stage = artifacts / "stage/s0-topic-mm-control"
env = dict(os.environ, WMMA_MATRIX_PROFILE="")
if not (artifacts / "checks/enabled-features/summary.json").exists():
    subprocess.run(["python3", str(tools / "feature_probes.py")], env=env, check=True)
parent_stage = artifacts / "stage/s0-parent-mm-verify"
if not (parent_stage / "snapshot-summary.json").exists():
    assert not parent_stage.exists(), "partial replacement parent snapshot exists; diagnose before retry"
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), "parent-mm-r1", parent_stage.name], env=env, check=True)
if not (stage / "snapshot-summary.json").exists():
    assert not stage.exists(), "partial control gate exists; diagnose it before another attempt"
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), "matrix-mm-r1", stage.name], env=env, check=True)
summary = json.loads((stage / "snapshot-summary.json").read_text())
assert summary["observations"] == 37 and not summary["aborted"] and summary["verify_returncode"] == 0
assert (stage / "COMMIT").read_bytes() == (artifacts / "build/matrix-mm-r1/COMMIT").read_bytes()
reference = (parent_stage / "verify.out").read_text()
current = (stage / "verify.out").read_text()
reference = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", reference)
current = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", current)
diff = "".join(difflib.unified_diff(reference.splitlines(True), current.splitlines(True), fromfile="parent", tofile="topic-control"))
(stage / "parent-output.diff").write_text(diff)
assert not diff, "control verify output differs from parent with rates removed"
old = (artifacts / "stage/s0-parent-verify/verify.out").read_text()
old = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", old)
old_diff = "".join(difflib.unified_diff(old.splitlines(True), reference.splitlines(True), fromfile="old-parent", tofile="feature-enabled-parent"))
(parent_stage / "old-parent-output.diff").write_text(old_diff)
assert not old_diff, "feature-enabled parent behavior differs; diagnose before A/A"
subprocess.run(["timeout", "--signal=TERM", "--kill-after=60", "4h", "python3", str(tools / "aa_session.py")], env=env, check=True)
