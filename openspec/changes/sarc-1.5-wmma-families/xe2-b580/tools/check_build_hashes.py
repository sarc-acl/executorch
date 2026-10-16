#!/usr/bin/env python3
import hashlib
from pathlib import Path
import re
import sys

manifest = Path(sys.argv[1]).read_text()
assert "BUILD_BOTH_OK" in manifest.splitlines()
recorded = {}
for line in manifest.splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match:
        recorded[Path(match[2])] = match[1]
paths = []
for root in map(Path, sys.argv[2:]):
    paths.append(root / "llama/examples/models/llama/llama_main")
    libraries = list((root / "llama").rglob("libllama_runner.so"))
    assert libraries, f"missing runner library in {root}"
    paths.extend(libraries)
paths.append(Path(sys.argv[2]) / "tests/test_llama_microbench")
for path in paths:
    assert path in recorded, f"unrecorded build artifact: {path}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == recorded[path], f"build artifact changed: {path}"
print(f"BUILD_HASHES_PASS {len(paths)} files")
