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
    return {
        "t": sorted(runs)[len(runs) // 2],
        "runs": list(runs),
        "passed_runs": len(runs),
    }


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
            self.assertEqual((nomination["arms"], nomination["reason"]), ([], code))
        s = session({"auto": arm(1, 1, 1), "A'": arm(1, 1, 1)})
        self.assertEqual(gen.nominate(SHAPE, [("s", s)])[1][1], "not-timed")


def confirmation(arms, nominated, band=0.02):
    return session(
        arms,
        band=band,
        confirmation=True,
        runs=gen.CONFIRM_RUNS,
        nominations={SHAPE: {"arms": nominated}},
    )


def five(t):
    return arm(t, t, t, t, t)


def confirm(s):
    return gen.confirm(SHAPE, s, s["nominations"][SHAPE])


class NominationGate(unittest.TestCase):
    def test_faster_unstable_arm_loses_nomination(self):
        # Same route: the faster arm (ratio 2.5) spreads 0.5 over its runs, the
        # slower one (ratio 2.0) is stable and qualifies; the stable arm is
        # nominated. The direct route's only arm loses (0.98x): no nominee.
        s = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t128x128_f16_alc": arm(30, 40, 50),
                "xe2_t32x64_f16_alc": arm(50, 50, 50),
                "xe2_t32x64_f16_alc_dg": arm(102, 102, 102),
            }
        )
        nomination, (info, code, _) = gen.nominate(SHAPE, [("s", s)])
        self.assertEqual(nomination, {"arms": ["xe2_t32x64_f16_alc"]})
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc")

    def test_each_observation_gated_against_its_own_session(self):
        # The reference sweep's A/A band is 0.213 (above 0.10), the retime's
        # 0.02 with stable runs: the arm qualifies by its retime observation
        # (ratio 1.22) and is nominated, although the reference observation and
        # the larger of the two bands would reject it. The reference's stock
        # time (100 against 81.97) does not filter the retime out.
        ref = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "amd_t16x32_f16_alc": arm(80, 80, 80),
            },
            band=0.213,
        )
        retime = session(
            {
                "auto": arm(81.97, 81.97, 81.97),
                "A'": arm(81.97, 81.97, 81.97),
                "amd_t16x32_f16_alc": arm(67.19, 67.19, 67.19),
            }
        )
        nomination, (info, code, _) = gen.nominate(
            SHAPE, [("sweep", ref), ("retime", retime)]
        )
        self.assertEqual(nomination, {"arms": ["amd_t16x32_f16_alc"]})
        self.assertIsNone(code)
        self.assertEqual(info["nominated"]["amd_t16x32_f16_alc"]["session"], "retime")
        self.assertAlmostEqual(
            info["nominated"]["amd_t16x32_f16_alc"]["ratio"], 81.97 / 67.19
        )
        # The reference sweep alone nominates nothing (unstable).
        nomination, (_, code, _) = gen.nominate(SHAPE, [("sweep", ref)])
        self.assertEqual((nomination["arms"], code), ([], "unstable"))

    def test_validation_failure_in_any_session_excludes_the_arm(self):
        good = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc": arm(50, 50, 50),
            }
        )
        bad = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t32x64_f16_alc": {
                    "t": None,
                    "passed_runs": 2,
                    "reason": "FAIL; PASS",
                },
            }
        )
        nomination, _ = gen.nominate(SHAPE, [("good", good), ("bad", bad)])
        self.assertEqual(nomination["arms"], [])

    def test_no_qualifying_arm_keeps_the_reason(self):
        for arms, code in (
            ({"xe2_t32x64_f16_alc": arm(30, 40, 50)}, "unstable"),
            ({"xe2_t32x64_f16_alc": arm(98, 98, 98)}, "below-threshold"),
        ):
            s = session({"auto": arm(100, 100, 100), "A'": arm(100, 100, 100), **arms})
            nomination, (_, got, _) = gen.nominate(SHAPE, [("s", s)])
            self.assertEqual((nomination["arms"], nomination["reason"]), ([], code))
            self.assertEqual(got, code)
            info, got, _ = gen.confirm(SHAPE, {"shapes": {}, "band": {}}, nomination)
            self.assertEqual(got, code)


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
        info, code, _ = confirm(s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc_dg")
        self.assertEqual(info["route"], "direct_coopmat")

    def test_not_reproduced(self):
        s = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": five(104),
            },
            ["xe2_t32x64_f16_alc"],
        )
        self.assertEqual(confirm(s)[1], "below-threshold")

    def test_control_arm_is_not_emitted(self):
        s = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc_simt": five(10),
                "xe2_t32x64_f16_diag_once": five(5),
                "xe2_t32x64_f16_alc": five(40),
            },
            [
                "xe2_t32x64_f16_alc_simt",
                "xe2_t32x64_f16_diag_once",
                "xe2_t32x64_f16_alc",
            ],
        )
        info, code, _ = confirm(s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc")
        self.assertEqual(
            info["not_routable"],
            ["xe2_t32x64_f16_alc_simt", "xe2_t32x64_f16_diag_once"],
        )

    def test_unstable_arm(self):
        s = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": arm(30, 40, 40, 40, 50),
            },
            ["xe2_t32x64_f16_alc"],
        )
        info, code, detail = confirm(s)
        self.assertEqual(code, "unstable")
        self.assertIn("arm spread", detail)

    def test_incomplete_confirmation_cannot_emit_a_row(self):
        # A row needs all five runs of auto, A' and the chosen arm.
        good = {"auto": five(100), "A'": five(100), "xe2_t32x64_f16_alc": five(40)}
        for arm_name, record in (
            ("auto", arm(100, 100, 100)),
            ("A'", {**five(100), "passed_runs": 4}),
            ("xe2_t32x64_f16_alc", arm(40, 40, 40, 40)),
        ):
            s = confirmation({**good, arm_name: record}, ["xe2_t32x64_f16_alc"])
            info, code, _ = confirm(s)
            self.assertEqual(code, "not-timed", arm_name)
            self.assertNotIn("best", info)
        self.assertIsNone(confirm(confirmation(good, ["xe2_t32x64_f16_alc"]))[1])

    def test_last_session_covering_the_nomination(self):
        nomination = {"arms": ["xe2_t32x64_f16_alc", "xe2_t32x64_f16_alc_dg"]}
        first = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": five(40),
                "xe2_t32x64_f16_alc_dg": five(50),
            },
            nomination["arms"],
        )
        # The later session timed only one of the two nominees: not used.
        partial = confirmation(
            {"auto": five(100), "A'": five(100), "xe2_t32x64_f16_alc_dg": five(30)},
            ["xe2_t32x64_f16_alc_dg"],
        )
        info, code, _, name = gen.decide(
            SHAPE, [("first", first), ("partial", partial)], nomination
        )
        self.assertEqual((name, info["best"]), ("first", "xe2_t32x64_f16_alc"))
        fresh = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": five(45),
                "xe2_t32x64_f16_alc_dg": five(30),
            },
            nomination["arms"],
        )
        info, code, _, name = gen.decide(
            SHAPE,
            [("first", first), ("partial", partial), ("fresh", fresh)],
            nomination,
        )
        self.assertEqual((name, info["best"]), ("fresh", "xe2_t32x64_f16_alc_dg"))
        _, code, _, name = gen.decide(SHAPE, [("partial", partial)], nomination)
        self.assertEqual((code, name), ("not-timed", None))


class Refusal(unittest.TestCase):
    def run_table(self, s):
        with tempfile.TemporaryDirectory() as tmp:
            d = write(Path(tmp), "session", s)
            nominations = Path(tmp) / "nominations.json"
            nominations.write_text(
                json.dumps({"dev": {SHAPE: {"arms": ["xe2_t32x64_f16_alc"]}}})
            )
            before = gen.TABLE.read_text()
            r = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / gen.__name__) + ".py",
                    "table",
                    "--nominations",
                    str(nominations),
                    "--check",
                    str(d),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(gen.TABLE.read_text(), before)
            return r

    def test_table_needs_confirmation_sessions(self):
        r = self.run_table(session({"auto": arm(1, 1, 1), "A'": arm(1, 1, 1)}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a confirmation session", r.stderr)

    def test_table_needs_five_run_sessions(self):
        s = confirmation(
            {"auto": arm(1, 1, 1), "A'": arm(1, 1, 1)}, ["xe2_t32x64_f16_alc"]
        )
        s["runs"] = 3
        r = self.run_table(s)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("confirmation session of 5 runs", r.stderr)


if __name__ == "__main__":
    unittest.main()
