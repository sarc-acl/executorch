#!/usr/bin/env python3
"""Run an authorized target script through bash -s with separately quoted arguments."""

import argparse
from pathlib import Path
import shlex
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    command = "bash -s -- " + " ".join(shlex.quote(argument) for argument in args.arguments)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                             args.target, command], input=args.script.read_bytes(), check=False)
    raise SystemExit(result.returncode)
