"""Unit checks for tile_constraints.py against ground truth already in the
repo: the committed tsweep yaml variants, the specs/025 known glslc failures,
and the specs/035 780M gate results (results/replay-780m TSVs).

Run: python3 test_constraints.py   (or pytest test_constraints.py)
"""

import re
from pathlib import Path

import tile_constraints as tc

SPEC_DIR = Path(__file__).resolve().parent.parent
REPO = SPEC_DIR.parent.parent
GLSL = REPO / "backends/vulkan/runtime/graph/ops/glsl"
REPLAY = SPEC_DIR / "results/replay-780m"

# AMD 780M / RADV limits (cross-checked against vulkaninfo on this box; the
# real sweep queries them via device_fingerprint instead of trusting these).
LIMITS_780M = tc.DeviceLimits(
    max_shared_mem_bytes=65536,
    max_wg_invocations=1024,
    subgroup_sizes=(32, 64),
)
GROUP_SIZE = 128  # both buffer ptes on this box (run_buffer_exports.sh)

TOKEN_RE = re.compile(r"tsweep_t\d+x\d+k\d+g\d\ds\d+")


def yaml_tokens(name):
    return sorted(set(TOKEN_RE.findall((GLSL / name).read_text())))


def gate_pass_tokens(tsv):
    toks = []
    for line in (REPLAY / tsv).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "PASS":
            toks.append(parts[0])
    return toks


def test_round_trip():
    for m in tc.WG_TILE_MN_CHOICES:
        for k in tc.WG_TILE_K_CHOICES:
            for gx in tc.SG_GRID_CHOICES:
                for sub in (32, 64):
                    t = tc.token(m, 64, k, gx, 2, sub)
                    p = tc.parse_token(t)
                    assert (
                        tc.token(
                            p["wg_tile_m"],
                            p["wg_tile_n"],
                            p["wg_tile_k"],
                            p["sg_grid_x"],
                            p["sg_grid_y"],
                            p["subgroup_size"],
                        )
                        == t
                    )
    for bad in (
        "tsweep_t64x64",
        "dbuf2",
        "tsweep_t64x64k32g222s32",
        "tsweep_t64xAk32g12s32",
        "linear_q4gsw_coopmat",
    ):
        try:
            tc.parse_token(bad)
            raise AssertionError(f"should have raised: {bad}")
        except ValueError:
            pass


def test_committed_yaml_tokens_parse_and_mostly_valid():
    """Every committed variant must parse; validity is checked per shader.
    Committed tokens were all accepted by glslc, so any the model calls
    invalid must be invalid only for runtime-legality reasons that glslc
    can't see -- there are none expected under group_size=128."""
    for shader, yaml in (
        ("q4gsw", "linear_q4gsw_coopmat_tsweep.yaml"),
        ("dq8ca", "linear_dq8ca_q4gsw_coopmat_tsweep.yaml"),
    ):
        toks = yaml_tokens(yaml)
        assert toks, f"no tokens found in {yaml}"
        invalid = []
        for t in toks:
            c = tc.derive_token(shader, t, GROUP_SIZE, LIMITS_780M)
            if not c["valid"]:
                invalid.append((t, c["invalid_reasons"]))
        assert not invalid, f"{shader}: committed tokens judged invalid: {invalid}"


def test_known_glslc_failures_invalid():
    for t in (
        "tsweep_t32x16k16g12s64",
        "tsweep_t64x16k16g12s64",
        "tsweep_t128x16k16g12s64",
    ):
        c = tc.derive_token("dq8ca", t, GROUP_SIZE, LIMITS_780M)
        assert not c["valid"], t
        assert any("temp_B" in r for r in c["invalid_reasons"]), c["invalid_reasons"]


def test_780m_gate_pass_tokens_all_legal():
    for shader, tsv in (("q4gsw", "4w_gate.tsv"), ("dq8ca", "8da4w_gate.tsv")):
        toks = gate_pass_tokens(tsv)
        assert len(toks) >= 15, f"{tsv}: unexpectedly few PASS rows ({len(toks)})"
        for t in toks:
            c = tc.derive_token(shader, t, GROUP_SIZE, LIMITS_780M)
            assert c["valid"], (shader, t, c["invalid_reasons"])


def test_enumeration_contains_known_winners():
    legal_4w = {
        c["token"] for c in tc.enumerate_legal("q4gsw", LIMITS_780M, GROUP_SIZE)
    }
    legal_dq = {
        c["token"] for c in tc.enumerate_legal("dq8ca", LIMITS_780M, GROUP_SIZE)
    }
    assert "tsweep_t128x64k32g22s32" in legal_4w  # 780M 4w winner
    assert "tsweep_t128x64k16g22s32" in legal_4w  # M5 4w winner
    assert "tsweep_t64x128k32g41s32" in legal_dq  # 780M dq8ca winner
    assert "tsweep_t64x32k32g12s64" in legal_dq  # M5 dq8ca winner
    print(f"legal universe: q4gsw={len(legal_4w)} dq8ca={len(legal_dq)}")


def test_xclipse_quirk():
    xclipse = tc.DeviceLimits(
        65536, 1024, (32, 64), quirks=frozenset({"no_int8_wmma_sg32"})
    )
    c = tc.derive_token("dq8ca", "tsweep_t64x128k32g41s32", GROUP_SIZE, xclipse)
    assert not c["valid"]
    c = tc.derive_token("q4gsw", "tsweep_t128x64k32g22s32", GROUP_SIZE, xclipse)
    assert c["valid"]  # quirk is int8-shader-specific


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
