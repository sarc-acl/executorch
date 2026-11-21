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

    def test_full_name_arms_are_routable(self):
        # conv2d_simt_ and conv1d_simt_ are kernels, not the FP16 _simt
        # control variants.
        a = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "conv2d_simt_t64x64_m4n2_dg_float": arm(40, 40, 40),
                "conv1d_simt_pw_t64x32_half": arm(50, 50, 50),
                "xe2_t32x64_f16_alc_simt": arm(10, 10, 10),
            }
        )
        nomination, (info, code, _) = gen.nominate(SHAPE, [("a", a)])
        self.assertEqual(
            nomination["arms"], ["conv2d_simt_t64x64_m4n2_dg_float"]
        )
        self.assertIsNone(code)
        self.assertEqual(gen.route_of("conv1d_simt_pw_t64x32_half"), "simt")
        self.assertEqual(gen.route_of("b1x1"), "b1x1")
        self.assertEqual(gen.route_of("q8ta_conv2d_dw_w1"), "dw_tile")
        self.assertTrue(gen.routable("q8ta_conv2d_dw_w1"))
        self.assertFalse(gen.routable("xe2_t32x64_f16_alc_simt"))

    def test_stock_drift_nominates_a_win_against_every_stock_run(self):
        # Stock spread 0.84: the arm's slowest run (9.1) still beats the
        # fastest stock run (11.2) by more than 5%; nominated, with the
        # conservative ratio 11.2 / 9.1, not the ratio of medians (10.6).
        s = session(
            {
                "auto": arm(71.2, 71.2, 11.2),
                "A'": arm(71.0, 71.2, 11.2),
                "nvidia_t16x32_f16_alc": arm(6.6, 9.1, 6.7),
            }
        )
        nomination, (info, code, _) = gen.nominate(SHAPE, [("s", s)])
        self.assertEqual(nomination, {"arms": ["nvidia_t16x32_f16_alc"]})
        self.assertIsNone(code)
        self.assertAlmostEqual(info["ratio"], 11.2 / 9.1)
        self.assertAlmostEqual(info["median_ratio"], 71.2 / 6.7)

    def test_win_on_medians_only_is_unstable(self):
        s = session(
            {
                "auto": arm(71.2, 71.2, 11.2),
                "A'": arm(71.0, 71.2, 11.2),
                "nvidia_t16x32_f16_alc": arm(10.8, 12.0, 10.9),
            }
        )
        nomination, (_, code, detail) = gen.nominate(SHAPE, [("s", s)])
        self.assertEqual((nomination["arms"], code), ([], "unstable"))
        self.assertIn("wins on medians", detail)

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
        schedule="paired-reverse",
        nominations={SHAPE: {"arms": nominated}},
    )


def five(t):
    return arm(t, t, t, t, t)


def confirm(s):
    return gen.confirm(SHAPE, s, s["nominations"][SHAPE])


class NominationGate(unittest.TestCase):
    def test_arm_with_a_losing_run_loses_nomination(self):
        # Same route: the faster arm on medians (ratio 2.5) has a run slower
        # than stock (104); the slower one (ratio 2.0) qualifies and is
        # nominated. The direct route's only arm loses (0.98x): no nominee.
        s = session(
            {
                "auto": arm(100, 100, 100),
                "A'": arm(100, 100, 100),
                "xe2_t128x128_f16_alc": arm(30, 40, 104),
                "xe2_t32x64_f16_alc": arm(50, 50, 50),
                "xe2_t32x64_f16_alc_dg": arm(102, 102, 102),
            }
        )
        nomination, (info, code, _) = gen.nominate(SHAPE, [("s", s)])
        self.assertEqual(nomination, {"arms": ["xe2_t32x64_f16_alc"]})
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc")

    def test_each_observation_judged_in_its_own_session(self):
        # In the reference sweep one stock run is fast (60): the arm (80) wins
        # on medians but not against every stock run. In the retime it beats
        # every stock run (ratio 1.22) and is nominated from there. The
        # reference's stock time (100 against 81.97) does not filter the retime
        # out.
        ref = session(
            {
                "auto": arm(100, 100, 60),
                "A'": arm(100, 100, 100),
                "amd_t16x32_f16_alc": arm(80, 80, 80),
            },
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

    def test_validation_is_judged_per_session(self):
        # The arm qualifies in the good sweep and failed the reference in the
        # other one: the good sweep nominates it, the failed sweep cannot.
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
        nomination, (info, _, _) = gen.nominate(SHAPE, [("good", good), ("bad", bad)])
        self.assertEqual(nomination["arms"], ["xe2_t32x64_f16_alc"])
        self.assertEqual(info["nominated"]["xe2_t32x64_f16_alc"]["session"], "good")
        nomination, (_, code, _) = gen.nominate(SHAPE, [("bad", bad)])
        self.assertEqual((nomination["arms"], code), ([], "validation"))

    def test_no_qualifying_arm_keeps_the_reason(self):
        for arms, code in (
            ({"xe2_t32x64_f16_alc": arm(30, 40, 104)}, "unstable"),
            ({"xe2_t32x64_f16_alc": arm(98, 98, 98)}, "below-threshold"),
        ):
            s = session({"auto": arm(100, 100, 100), "A'": arm(100, 100, 100), **arms})
            nomination, (_, got, _) = gen.nominate(SHAPE, [("s", s)])
            self.assertEqual((nomination["arms"], nomination["reason"]), ([], code))
            self.assertEqual(got, code)
            info, got, _ = gen.confirm(SHAPE, {"shapes": {}, "band": {}}, nomination)
            self.assertEqual(got, code)


class StockReason(unittest.TestCase):
    """Design D3: a stock arm that failed the reference is `validation`; a stock
    arm without a time for another reason is `not-timed`."""

    FAILED = {"t": None, "passed_runs": 4, "reason": "FAIL; PASS"}
    MISSING = {"t": None, "passed_runs": 0, "reason": "not run"}
    # An exception during setup or the run (etvk_ops_bench reports it as
    # validation FAIL with an error; i2c.py summarize records ERROR).
    ERRORED = {"t": None, "passed_runs": 0, "reason": "ERROR"}
    ERRORED_ONCE = {"t": None, "passed_runs": 4, "reason": "ERROR; PASS"}
    CASES = (
        ("failed", "validation"),
        ("missing", "not-timed"),
        ("errored", "not-timed"),
        ("errored_once", "not-timed"),
    )

    def stock(self, name):
        return {
            "failed": self.FAILED,
            "missing": self.MISSING,
            "errored": self.ERRORED,
            "errored_once": self.ERRORED_ONCE,
        }[name]

    def test_nomination(self):
        for which, code in self.CASES:
            stock = self.stock(which)
            for name in gen.STOCK:
                arms = {"auto": arm(100, 100, 100), "A'": arm(100, 100, 100)}
                arms[name] = stock
                s = session({**arms, "xe2_t32x64_f16_alc": arm(50, 50, 50)}, band=None)
                nomination, (_, got, _) = gen.nominate(SHAPE, [("s", s)])
                self.assertEqual((nomination["reason"], got), (code, code), name)
                self.assertEqual(nomination["arms"], [])

    def test_confirmation(self):
        for which, code in self.CASES:
            stock = self.stock(which)
            for name in gen.STOCK:
                arms = {
                    "auto": five(100),
                    "A'": five(100),
                    "xe2_t32x64_f16_alc": five(40),
                }
                arms[name] = stock
                s = confirmation(arms, ["xe2_t32x64_f16_alc"])
                info, got, _ = confirm(s)
                self.assertEqual(got, code, name)
                self.assertNotIn("best", info)


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

    def test_previous_row_kept_unless_beaten_by_the_rule(self):
        arms = {
            "auto": arm(100, 100, 100, 100, 100),
            "A'": arm(100, 100, 100, 100, 100),
            "xe2_t32x64_f16_alc": arm(50, 50, 50, 50, 50),
            "xe2_t64x64_f16_g4x2_alc": arm(49, 49, 49, 49, 49),
            "xe2_t64x32_f16_g2x4_alc": arm(40, 40, 40, 40, 40),
        }
        s = confirmation(arms, ["xe2_t64x64_f16_g4x2_alc", "xe2_t32x64_f16_alc"])
        kept = {"arms": ["xe2_t64x64_f16_g4x2_alc", "xe2_t32x64_f16_alc"],
                "previous_row": "xe2_t32x64_f16_alc"}
        info, code, _ = gen.confirm(SHAPE, s, kept)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc")
        beaten = {"arms": ["xe2_t64x32_f16_g2x4_alc", "xe2_t32x64_f16_alc"],
                  "previous_row": "xe2_t32x64_f16_alc"}
        info, code, _ = gen.confirm(SHAPE, s, beaten)
        self.assertEqual(info["best"], "xe2_t64x32_f16_g2x4_alc")

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

    def test_steady_arm_against_two_level_stock(self):
        # The RTX 4070 Ti SUPER case of the proposal: the stock route changes
        # level between runs (spread above 3, A/A band 1.2), the arm is steady
        # and its slowest run beats the fastest stock run: a row, with the
        # conservative ratio 154 / 46.
        s = confirmation(
            {
                "auto": arm(614, 164, 620, 154, 614),
                "A'": arm(612, 160, 618, 156, 615),
                "nvidia_t16x32_f16_alc_dg": arm(42, 46, 46, 42, 42),
            },
            ["nvidia_t16x32_f16_alc_dg"],
            band=1.2,
        )
        info, code, _ = confirm(s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "nvidia_t16x32_f16_alc_dg")
        self.assertAlmostEqual(info["ratio"], 154 / 46)
        self.assertAlmostEqual(info["median_ratio"], 614 / 42)
        self.assertGreater(info["stock_spread"], 0.10)
        rows, _ = gen.rows_of(
            [("s", s)],
            s["nominations"],
            (0x10DE, 0x2705),
            {SHAPE.removeprefix("sweep "): {"label": "x"}},
        )
        self.assertTrue(rows[0]["comment"].endswith(f"{154 / 46:.2f}x"))

    def test_win_on_medians_only_is_unstable(self):
        # Four runs at 40, one at 98: wins on medians (2.5) but its slowest run
        # is not 5% faster than the fastest stock run (100 / 98).
        s = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": arm(40, 40, 40, 40, 98),
            },
            ["xe2_t32x64_f16_alc"],
        )
        info, code, detail = confirm(s)
        self.assertEqual(code, "unstable")
        self.assertIn("wins on medians", detail)
        self.assertAlmostEqual(info["ratio"], 100 / 98)

    def test_lowest_slowest_run_is_chosen(self):
        # Both qualify; the first has the lower median (40) but a slow run
        # (60), the second the lower slowest run (46): the second is the row.
        s = confirmation(
            {
                "auto": five(100),
                "A'": five(100),
                "xe2_t32x64_f16_alc": arm(40, 40, 40, 41, 60),
                "xe2_t32x64_f16_alc_dg": arm(45, 45, 45, 46, 46),
            },
            ["xe2_t32x64_f16_alc_dg", "xe2_t32x64_f16_alc"],
        )
        info, code, _ = confirm(s)
        self.assertIsNone(code)
        self.assertEqual(info["best"], "xe2_t32x64_f16_alc_dg")
        self.assertAlmostEqual(info["ratio"], 100 / 46)

    def test_arm_with_a_failed_or_missing_run_does_not_qualify(self):
        # The faster arm failed the reference in one run or lacks a run: it is
        # not a candidate; the slower complete arm is the row, and alone the
        # configuration gets validation or not-timed.
        slow = {"xe2_t32x64_f16_alc_dg": five(50)}
        for record, code in (
            ({"t": None, "passed_runs": 4, "reason": "FAIL; PASS"}, "validation"),
            (arm(20, 20, 20, 20), "not-timed"),
        ):
            stock = {"auto": five(100), "A'": five(100)}
            s = confirmation(
                {**stock, "xe2_t32x64_f16_alc": record, **slow},
                ["xe2_t32x64_f16_alc_dg", "xe2_t32x64_f16_alc"],
            )
            info, got, _ = confirm(s)
            self.assertIsNone(got)
            self.assertEqual(info["best"], "xe2_t32x64_f16_alc_dg")
            s = confirmation(
                {**stock, "xe2_t32x64_f16_alc": record}, ["xe2_t32x64_f16_alc"]
            )
            info, got, _ = confirm(s)
            self.assertEqual(got, code)
            self.assertNotIn("best", info)

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


class NominateSessions(unittest.TestCase):
    def test_configurations_of_every_session(self):
        # An add-on sweep may time configurations the first session lacks.
        labels = ["sweep " + c["label"] for c in json.loads(gen.CONFIGS.read_text())]
        other = next(x for x in labels if x != SHAPE)
        stock = {"auto": arm(100, 100, 100), "A'": arm(100, 100, 100)}
        a = session({**stock, "xe2_t32x64_f16_alc": arm(40, 40, 40)})
        b = session({**stock, "xe2_t32x64_f16_alc": arm(40, 40, 40)})
        b["shapes"][other] = b["shapes"][SHAPE]
        b["band"][other] = 0.02
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [str(write(Path(tmp), n, s)) for n, s in (("a", a), ("b", b))]
            out = Path(tmp) / "nominations.json"
            subprocess.run(
                [sys.executable, "-B", str(HERE / gen.__name__) + ".py"]
                + ["nominate", "--out", str(out), *dirs],
                check=True,
                capture_output=True,
            )
            self.assertEqual(list(json.loads(out.read_text())["dev"]), [SHAPE, other])


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

    def test_table_needs_the_confirmation_schedule(self):
        s = confirmation(
            {"auto": five(1), "A'": five(1), "xe2_t32x64_f16_alc": five(1)},
            ["xe2_t32x64_f16_alc"],
        )
        del s["schedule"]
        r = self.run_table(s)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("paired-reverse", r.stderr)

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
