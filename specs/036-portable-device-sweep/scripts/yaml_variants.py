"""Idempotent tsweep-yaml variant management + rebuild driver.

Only ever mutates the existing tsweep yamls listed in SHADER_INFO (adding a
new yaml file would need a cmake reconfigure because ShaderLibrary.cmake
file-GLOBs the shader dir at configure time -- specs/041-dbuf4-tile-sweep's
two new dbuf4 yaml files were created once, out-of-band, with a seed variant
already present; this module only appends further candidates to files that
already exist on disk). Entries are text-appended in the exact committed format so
the diff stays reviewable; NAME strings must match what QuantizedLinear.cpp
constructs (base + "_" + token + "_buffer_<weight_storage>_half") or the
runtime hard-aborts with "Could not find ShaderInfo".

Rebuild = regenerate spv.cpp (per-shader SPIR-V cache: only new variants hit
glslc) + relink backend, bench, and llama_main. glslc failures are mapped back
to tokens via the tsweep token substring in the error log so callers can
remove + blocklist them and rebuild once more.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tile_constraints import parse_token

REPO = Path(__file__).resolve().parent.parent.parent.parent
GLSL = REPO / "backends/vulkan/runtime/graph/ops/glsl"

SHADER_INFO = {
    "q4gsw": {
        "yaml": GLSL / "linear_q4gsw_coopmat_tsweep.yaml",
        "base": "linear_q4gsw_coopmat",
        "extra_fields": {},
    },
    "dq8ca": {
        "yaml": GLSL / "linear_dq8ca_q4gsw_coopmat_tsweep.yaml",
        "base": "linear_dq8ca_q4gsw_coopmat",
        "extra_fields": {"WEIGHT_NBITS": 4},
    },
    # specs/041-dbuf4-tile-sweep: dbufN loop-structure variants (N covers
    # whichever of 1-4 isn't already the production winner for that scheme).
    # "base" stays the same production kernel name as their siblings --
    # QuantizedLinear.cpp always prepends the same base regardless of token
    # namespace; only the yaml file and the token prefix
    # (tsweep_dbufN_t... vs tsweep_t...) differ.
    "q4gsw_dbuf2": {
        "yaml": GLSL / "linear_q4gsw_coopmat_tsweep_dbuf2.yaml",
        "base": "linear_q4gsw_coopmat",
        "extra_fields": {},
    },
    "q4gsw_dbuf3": {
        "yaml": GLSL / "linear_q4gsw_coopmat_tsweep_dbuf3.yaml",
        "base": "linear_q4gsw_coopmat",
        "extra_fields": {},
    },
    "q4gsw_dbuf4": {
        "yaml": GLSL / "linear_q4gsw_coopmat_tsweep_dbuf4.yaml",
        "base": "linear_q4gsw_coopmat",
        "extra_fields": {},
    },
    "dq8ca_dbuf1": {
        "yaml": GLSL / "linear_dq8ca_q4gsw_coopmat_tsweep_dbuf1.yaml",
        "base": "linear_dq8ca_q4gsw_coopmat",
        "extra_fields": {"WEIGHT_NBITS": 4},
    },
    "dq8ca_dbuf3": {
        "yaml": GLSL / "linear_dq8ca_q4gsw_coopmat_tsweep_dbuf3.yaml",
        "base": "linear_dq8ca_q4gsw_coopmat",
        "extra_fields": {"WEIGHT_NBITS": 4},
    },
    "dq8ca_dbuf4": {
        "yaml": GLSL / "linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4.yaml",
        "base": "linear_dq8ca_q4gsw_coopmat",
        "extra_fields": {"WEIGHT_NBITS": 4},
    },
}

TOKEN_RE = re.compile(r"tsweep_(?:dbuf[1-4]_)?t\d+x\d+k\d+g\d\ds\d+")


def existing_tokens(shader):
    return set(TOKEN_RE.findall(SHADER_INFO[shader]["yaml"].read_text()))


def variant_entry(shader, token, weight_storage):
    info = SHADER_INFO[shader]
    p = parse_token(token)
    lines = [f"    - NAME: {info['base']}_{token}_buffer_{weight_storage}_half"]
    for field, val in info["extra_fields"].items():
        lines.append(f"      {field}: {val}")
    lines += [
        f"      WEIGHT_STORAGE: {weight_storage}",
        f"      WG_TILE_M: {p['wg_tile_m']}",
        f"      WG_TILE_N: {p['wg_tile_n']}",
        f"      WG_TILE_K: {p['wg_tile_k']}",
        f"      SG_GRID_X: {p['sg_grid_x']}",
        f"      SG_GRID_Y: {p['sg_grid_y']}",
        f"      SUBGROUP_SIZE: {p['subgroup_size']}",
    ]
    return "\n".join(lines) + "\n"


def ensure_variants(shader, tokens, both_storages=False):
    """Append yaml entries for tokens not already present. Returns the list of
    tokens actually added (idempotent: second call returns [])."""
    yaml_path = SHADER_INFO[shader]["yaml"]
    present = existing_tokens(shader)
    storages = ("texture2d", "buffer") if both_storages else ("texture2d",)
    added = []
    chunks = []
    for tok in tokens:
        parse_token(tok)  # reject malformed input before touching the file
        if tok in present:
            continue
        for st in storages:
            chunks.append(variant_entry(shader, tok, st))
        present.add(tok)
        added.append(tok)
    if chunks:
        text = yaml_path.read_text()
        if not text.endswith("\n"):
            text += "\n"
        yaml_path.write_text(text + "".join(chunks))
    return added


def remove_variants(shader, tokens):
    """Remove all entries whose NAME contains any of the given tokens.
    Returns the number of entries removed."""
    yaml_path = SHADER_INFO[shader]["yaml"]
    doomed = set(tokens)
    out, removed, skipping = [], 0, False
    for line in yaml_path.read_text().splitlines(keepends=True):
        if re.match(r"\s*- NAME:", line):
            tok = TOKEN_RE.search(line)
            skipping = tok is not None and tok.group(0) in doomed
            if skipping:
                removed += 1
                continue
        elif skipping and re.match(r"\s{6}\w+:", line):
            continue
        else:
            skipping = False
        out.append(line)
    if removed:
        yaml_path.write_text("".join(out))
    return removed


@dataclass
class BuildResult:
    ok: bool
    failed_tokens: list
    log_excerpt: str


BUILD_STEPS = (
    ["cmake", "--build", "cmake-out-vk", "--target", "install", "--config", "Release"],
    # The custom_ops tests are a separately-configured build dir with their own
    # generated shader library -- building only the main tree leaves the bench
    # dispatching stale shaders.
    [
        "cmake",
        "--build",
        "cmake-out-vk/backends/vulkan/test/custom_ops",
        "--target",
        "test_coopmat_linear_bench",
        "--config",
        "Release",
    ],
    ["cmake", "--build", "cmake-out-vk/examples/models/llama", "--config", "Release"],
)


def rebuild(repo_root=REPO, jobs=None, log_path=None):
    import os

    j = str(jobs or os.cpu_count())
    log = []
    for step in BUILD_STEPS:
        proc = subprocess.run(
            step + ["-j", j], cwd=repo_root, capture_output=True, text=True
        )
        log.append(f"$ {' '.join(step)}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode != 0:
            full = "\n".join(log)
            if log_path:
                Path(log_path).write_text(full)
            failed = sorted(set(TOKEN_RE.findall(proc.stdout + proc.stderr)))
            return BuildResult(False, failed, full[-4000:])
    full = "\n".join(log)
    if log_path:
        Path(log_path).write_text(full)
    return BuildResult(True, [], full[-1000:])
