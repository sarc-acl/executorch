#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit checks of gen_conv2d_coopmat_table.py on synthetic sessions:
python3 test_gen_conv2d_coopmat_table.py"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_conv2d_coopmat_table as gen  # noqa: E402

SHAPE = "sweep 1x1 s1x1 p0x0 d1x1 1008-192 14x14"


def arm(*runs):
    return {"t": sorted(runs)[len(runs) // 2], "runs": list(runs), "passed_runs": len(runs)}


def skipped(reason):
    return {"t": None, "passed_runs": 0, "reason": reason}


def session(arms, band=0.02, **extra):
    return {
        "device": "dev",
        "device_name": "Synthetic GPU",
        "driver": "driver 1",
        "band": {SHAPE: band},
        "shapes": {SHAPE: {"set": "s1", "arms": arms}},
        **extra,
    }


def write(root, name, s):
    d = root / name
    d.mkdir()
    (d / "session.json").write_text(json.dumps(s))
    props = [{"validation": {"vendor_device_id": "8086:e20b"}}]
    (d / "coopmat-properties.json").write_text(json.dumps(props))
    return d


class Nomination(unittest.TestCase):
    def test_one_arm_per_route_by_own_session_ratio(self):
        a = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc": arm(40, 40, 40),
                "xe2_t16x32_f16_alc": arm(50, 50, 50),
            },
            band=0.10,
        )
        b = session(
            {
                "auto": arm(92, 92, 92),
                "A'": arm(92, 92, 92),
                "xe2_t32x64_f16_alc_dg": arm(38, 38, 38),
                "xe2_t32x64_f16_alc_simt_dg": arm(10, 10, 10),
            },
            band=0.10,
        )
        nomination, (info, code, _) = gen.nominate(SHAPE, [("a", a), ("b", b)])
        self.assertEqual(
            nomination["arms"], ["xe2_t32x64_f16_alc_dg", "xe2_t32x64_f16_alc"]
        )
        self.assertNotIn("xe2_t32x64_f16_alc_simt_dg", info["nominated"])
        # The sweep decision picks the lower route time among qualifying arms.
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc_dg")

    def test_stock_drift_is_unstable(self):
        s = session(
            {
                "auto": arm(71.2, 71.2, 11.2),
                "A'": arm(71.0, 71.2, 11.2),
                "nvidia_t16x32_f16_alc": arm(6.6, 9.1, 6.7),
            }
        )
        _, (info, code, detail) = gen.nominate(SHAPE, [("s", s)])
        self.assertEqual(code, "unstable")
        self.assertIn("stock spread", detail)

    def test_reason_codes_without_a_time(self):
        cases = {
            "unsupported": skipped("SKIP x cannot run this case: K step"),
            "validation": {"t": None, "passed_runs": 1, "reason": "FAIL; PASS"},
        }
        for code, record in cases.items():
            s = session(
                {"auto": arm(1, 1, 1), "A'": arm(1, 1, 1), "xe2_t16x32_f16": record}
            )
            nomination, (_, got, _) = gen.nominate(SHAPE, [("s", s)])
            self.assertEqual(got, code)
            self.assertEqual(nomination, {"arms": [], "reason": code})
        s = session({"auto": arm(1, 1, 1), "A'": arm(1, 1, 1)})
        self.assertEqual(gen.nominate(SHAPE, [("s", s)])[1][1], "not-timed")


def confirmation(arms, nominated, band=0.02):
    return session(
        arms,
        band=band,
        confirmation=True,
        nominations={SHAPE: {"arms": nominated}},
    )


class Confirmation(unittest.TestCase):
    def test_lower_absolute_time_wins(self):
        # In the sweeps the pointwise arm had the larger ratio (2.5 against
        # 2.42, different stock times); in the confirmation session the direct
        # arm is faster and is the row.
        s = confirmation(
            {
                "auto": arm(100, 101, 100, 100, 99),
                "A'": arm(100, 100, 101, 99, 100),
                "xe2_t32x64_f16_alc": arm(41, 41, 42, 41, 41),
                "xe2_t32x64_f16_alc_dg": arm(37, 37, 38, 37, 37),
            },
            ["xe2_t32x64_f16_alc_dg", "xe2_t32x64_f16_alc"],
        )
        info, code, _ = gen.confirm(SHAPE, s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc_dg")
        self.assertEqual(info["route"], "direct_coopmat")

    def test_not_reproduced(self):
        s = confirmation(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc": arm(104, 104, 104),
            },
            ["xe2_t32x64_f16_alc"],
        )
        self.assertEqual(gen.confirm(SHAPE, s)[1], "below-threshold")

    def test_control_arm_is_not_emitted(self):
        s = confirmation(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc_simt": arm(10, 10, 10),
                "xe2_t32x64_f16_diag_once": arm(5, 5, 5),
                "xe2_t32x64_f16_alc": arm(40, 40, 40),
            },
            ["xe2_t32x64_f16_alc_simt", "xe2_t32x64_f16_diag_once", "xe2_t32x64_f16_alc"],
        )
        info, code, _ = gen.confirm(SHAPE, s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc")
        self.assertEqual(
            info["not_routable"], ["xe2_t32x64_f16_alc_simt", "xe2_t32x64_f16_diag_once"]
        )

    def test_unstable_arm(self):
        s = confirmation(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc": arm(30, 40, 50),
            },
            ["xe2_t32x64_f16_alc"],
        )
        info, code, detail = gen.confirm(SHAPE, s)
        self.assertEqual(code, "unstable")
        self.assertIn("arm spread", detail)


class Refusal(unittest.TestCase):
    def test_table_needs_confirmation_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = write(
                Path(tmp),
                "sweep",
                session({"auto": arm(1, 1, 1), "A'": arm(1, 1, 1)}),
            )
            before = gen.TABLE.read_text()
            r = subprocess.run(
                [sys.executable, "-B", str(HERE / gen.__name__) + ".py", "table", str(d)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a confirmation session", r.stderr)
            self.assertEqual(gen.TABLE.read_text(), before)


if __name__ == "__main__":
    unittest.main()
