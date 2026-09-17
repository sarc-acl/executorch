"""Idempotent tsweep-yaml variant management + rebuild driver.

Only ever mutates the two existing tsweep yamls (adding a new yaml file would
need a cmake reconfigure because ShaderLibrary.cmake file-GLOBs the shader dir
at configure time). Entries are text-appended in the exact committed format so
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

REPO = Path(__file__).resolve().parents[4]
GLSL = REPO / "backends/vulkan/runtime/graph/ops/glsl"

# The tsweep families this branch still ships. "prefix" is what the runtime
# env vars (ET_VK_{Q4GSW,DQ8CA}_COOPMAT_VARIANT) and the yaml NAMEs use;
# tile_constraints' canonical token carries no family, so the two are joined by
# runtime_token() below.
#
# dq8ca pins MMA_K=32: NVIDIA enumerates coopmat<int8> only at 16x16x32, so the
# shipped 16x16x16 tile cannot create a pipeline here (see the mk32 variants in
# the yaml and the ET_VK_COOPMAT_ANY_DEVICE gate in QuantizedLinear.cpp).
SHADER_INFO = {
    "q4gsw": {
        "yaml": GLSL / "linear_q4gsw_coopmat_tsweep_dbuf4.yaml",
        "base": "linear_q4gsw_coopmat",
        "prefix": "tsweep_dbuf4_",
        "extra_fields": {},
        # (IO_STORAGE, WEIGHT_STORAGE) exactly as the shipped tile ships them:
        # the correctness bench asks for all of these, and a tile missing one
        # aborts the process with "Could not find ShaderInfo".
        "storages": (("buffer", "texture2d"), ("buffer", "buffer"), ("texture3d", "texture2d")),
    },
    "dq8ca": {
        "yaml": GLSL / "linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpgtr.yaml",
        "base": "linear_dq8ca_q4gsw_coopmat",
        "prefix": "tsweep_dbuf4zpgtr_mk32_",
        "extra_fields": {"WEIGHT_NBITS": 4, "MMA_K": 32},
        # dq8ca ships no buffer-weight variant; adding one would be a new,
        # unvalidated shader configuration rather than a tile change.
        "storages": (("buffer", "texture2d"), ("texture3d", "texture2d")),
    },
}

TOKEN_RE = re.compile(r"tsweep_(?:dbuf4|dbuf4zpgtr_mk32)_t\d+x\d+k\d+g\d\ds\d+")


def runtime_token(shader, tok):
    """canonical 'tsweep_t...' -> the family-prefixed token the runtime wants."""
    return SHADER_INFO[shader]["prefix"] + tok[len("tsweep_") :]


def canonical_token(tok):
    """Inverse of runtime_token(), for reading tokens back out of yaml/logs."""
    return "tsweep_" + tok[tok.index("_t", tok.index("tsweep_") + 6) + 1 :]


def existing_tokens(shader):
    return {
        canonical_token(t)
        for t in TOKEN_RE.findall(SHADER_INFO[shader]["yaml"].read_text())
    }


def variant_entry(shader, token, io_storage, weight_storage):
    info = SHADER_INFO[shader]
    p = parse_token(token)
    rt = runtime_token(shader, token)
    lines = [f"    - NAME: {info['base']}_{rt}_{io_storage}_{weight_storage}_half"]
    if io_storage != "buffer":
        lines.append(f"      IO_STORAGE: {io_storage}")
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


def ensure_variants(shader, tokens, both_storages=True):
    """Append yaml entries for tokens not already present. Returns the list of
    tokens actually added (idempotent: second call returns []).

    both_storages is kept for call-site compatibility; every storage pair the
    shader family ships is always emitted, because the bench aborts the whole
    process on the first one it cannot find."""
    del both_storages
    yaml_path = SHADER_INFO[shader]["yaml"]
    present = existing_tokens(shader)
    added = []
    chunks = []
    for tok in tokens:
        parse_token(tok)  # reject malformed input before touching the file
        if tok in present:
            continue
        for io_st, w_st in SHADER_INFO[shader]["storages"]:
            chunks.append(variant_entry(shader, tok, io_st, w_st))
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
            skipping = tok is not None and canonical_token(tok.group(0)) in doomed
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
    ["cmake", "--build", "cmake-out-nt", "--target", "install", "--config", "Release"],
    # The custom_ops tests are a separately-configured build dir with their own
    # generated shader library -- building only the main tree leaves the bench
    # dispatching stale shaders.
    [
        "cmake",
        "--build",
        "cmake-out-nt/backends/vulkan/test/custom_ops",
        "--target",
        "test_coopmat_linear_bench",
        "--config",
        "Release",
    ],
    ["cmake", "--build", "cmake-out-nt/examples/models/llama", "--config", "Release"],
)


def rebuild(repo_root=REPO, jobs=None, log_path=None):
    import os

    j = str(jobs or os.cpu_count())
    log = []
    # The custom_ops shader library GLOBs the glsl dir at configure time, so a
    # yaml whose CONTENT changed does not invalidate its generated spv.cpp --
    # the bench then reports missing_shader for every new tile while the main
    # tree has it. Deleting the generated file is what forces regeneration.
    stale_spv = (
        repo_root
        / "cmake-out-nt/backends/vulkan/test/custom_ops/vulkan_compute_shaders/spv.cpp"
    )
    if stale_spv.exists():
        stale_spv.unlink()
    for step in BUILD_STEPS:
        proc = subprocess.run(
            step + ["-j", j], cwd=repo_root, capture_output=True, text=True
        )
        log.append(f"$ {' '.join(step)}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode != 0:
            full = "\n".join(log)
            if log_path:
                Path(log_path).write_text(full)
            failed = sorted(
                {canonical_token(t) for t in TOKEN_RE.findall(proc.stdout + proc.stderr)}
            )
            return BuildResult(False, failed, full[-4000:])
    full = "\n".join(log)
    if log_path:
        Path(log_path).write_text(full)
    return BuildResult(True, [], full[-1000:])
