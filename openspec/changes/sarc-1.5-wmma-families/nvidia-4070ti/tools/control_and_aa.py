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
stage = artifacts / "stage/s0-topic-control"
env = dict(os.environ, WMMA_MATRIX_PROFILE="")
if not (stage / "snapshot-summary.json").exists():
    assert not stage.exists(), "partial control gate exists; diagnose it before another attempt"
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), "matrix-r1", "s0-topic-control"], env=env, check=True)
summary = json.loads((stage / "snapshot-summary.json").read_text())
assert summary["observations"] == 37 and not summary["aborted"] and summary["verify_returncode"] == 0
assert (stage / "COMMIT").read_bytes() == (artifacts / "build/matrix-r1/COMMIT").read_bytes()
reference = (artifacts / "stage/s0-parent-verify/verify.out").read_text()
current = (stage / "verify.out").read_text()
reference = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", reference)
current = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", current)
diff = "".join(difflib.unified_diff(reference.splitlines(True), current.splitlines(True), fromfile="parent", tofile="topic-control"))
(stage / "parent-output.diff").write_text(diff)
assert not diff, "control verify output differs from parent with rates removed"
subprocess.run(["timeout", "--signal=TERM", "--kill-after=60", "4h", "python3", str(tools / "aa_session.py")], env=env, check=True)
