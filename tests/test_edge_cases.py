"""Focused edge-case audit for Policy (gaps beyond test_policy_smoke)."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

from policy import Policy

ROOT = Path(__file__).resolve().parents[1]


def load_creatives() -> list[dict]:
    with (ROOT / "data" / "creatives.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def toy(*ids: str) -> list[dict]:
    return [{"creative_id": cid} for cid in ids]


def assert_valid(test: unittest.TestCase, shares: dict[str, float], ids: list[str]) -> None:
    test.assertEqual(set(shares), set(ids))
    test.assertTrue(all(math.isfinite(v) and v >= 0.0 for v in shares.values()))
    test.assertTrue(math.isclose(sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-12))
    n = len(ids)
    cap = max(Policy.DEFAULT_MAX_SHARE, 1.0 / n)
    for v in shares.values():
        test.assertLessEqual(v, cap + 1e-12)


class PolicyEdgeCaseTests(unittest.TestCase):
    def test_skipped_decision_day_uses_earlier_history_only(self) -> None:
        """Calling day 65 with history ending at day 1 is valid (skipped days)."""
        policy = Policy(toy("A", "B"), 2000.0, 0)
        history = [
            {"day": 1, "creative_id": "A", "spend": 100.0, "revenue": 400.0},
            {"day": 1, "creative_id": "B", "spend": 100.0, "revenue": 50.0},
        ]
        shares = policy.allocate(65, history)
        assert_valid(self, shares, ["A", "B"])
        self.assertGreater(shares["A"], shares["B"])

    def test_unknown_creative_ids_are_ignored(self) -> None:
        policy = Policy(toy("A", "B"), 2000.0, 0)
        base = [{"day": 1, "creative_id": "A", "spend": 100.0, "revenue": 200.0}]
        with_unknown = base + [
            {"day": 1, "creative_id": "UNKNOWN", "spend": 999.0, "revenue": 9999.0}
        ]
        self.assertEqual(policy.allocate(2, base), policy.allocate(2, with_unknown))

    def test_repeated_identical_nonempty_history_is_deterministic(self) -> None:
        policy = Policy(toy("A", "B", "C"), 2000.0, 0)
        history = [
            {"day": 1, "creative_id": "A", "spend": 200.0, "revenue": 800.0},
            {"day": 1, "creative_id": "B", "spend": 200.0, "revenue": 100.0},
        ]
        a = policy.allocate(2, history)
        b = policy.allocate(2, history)
        self.assertEqual(a, b)
        assert_valid(self, a, ["A", "B", "C"])

    def test_all_zero_revenue_with_positive_spend_is_valid(self) -> None:
        policy = Policy(toy("A", "B"), 2000.0, 0, test_budget_fraction=0.0)
        history = [
            {"day": 1, "creative_id": "A", "spend": 100.0, "revenue": 0.0},
            {"day": 1, "creative_id": "B", "spend": 100.0, "revenue": 0.0},
        ]
        shares = policy.allocate(2, history)
        assert_valid(self, shares, ["A", "B"])
        self.assertAlmostEqual(shares["A"], 0.5, places=12)
        self.assertAlmostEqual(shares["B"], 0.5, places=12)

    def test_equal_rates_split_exploit_mass_evenly(self) -> None:
        policy = Policy(
            toy("A", "B", "C"),
            2000.0,
            0,
            test_budget_fraction=0.0,
            low_spend_threshold=50.0,
        )
        history = [
            {"day": 1, "creative_id": "A", "spend": 100.0, "revenue": 200.0},
            {"day": 1, "creative_id": "B", "spend": 100.0, "revenue": 200.0},
            {"day": 1, "creative_id": "C", "spend": 100.0, "revenue": 200.0},
        ]
        shares = policy.allocate(2, history)
        assert_valid(self, shares, ["A", "B", "C"])
        self.assertAlmostEqual(shares["A"], shares["B"], places=12)
        self.assertAlmostEqual(shares["B"], shares["C"], places=12)

    def test_every_loaded_creative_is_returned_with_valid_shares(self) -> None:
        creatives = load_creatives()
        ids = [str(c["creative_id"]) for c in creatives]
        self.assertGreaterEqual(len(ids), 2)
        policy = Policy(creatives, 2000.0, 7)
        history = [
            {"day": 1, "creative_id": ids[0], "spend": 500.0, "revenue": 2000.0},
            {"day": 1, "creative_id": ids[1], "spend": 500.0, "revenue": 100.0},
        ]
        shares = policy.allocate(61, history)
        assert_valid(self, shares, ids)
        self.assertEqual(len(shares), len(ids))

    def test_single_exploiter_with_exploration_cap_redistributes_residual(self) -> None:
        """One performance-pool creative + 10% reserve + 20% cap.

        Residual above the cap is redistributed to explorers (who keep their
        reserve floors). Documented expected shape for n=50:
        - lone exploiter ends at the 20% cap
        - explorers jointly hold the remaining 80%
        """
        ids = [f"C{i:02d}" for i in range(50)]
        policy = Policy(
            toy(*ids),
            2000.0,
            7,
            test_budget_fraction=0.10,
            low_spend_threshold=50.0,
            max_share=0.20,
        )
        history = [{"day": 1, "creative_id": "C00", "spend": 1000.0, "revenue": 8000.0}]
        shares = policy.allocate(2, history)
        assert_valid(self, shares, ids)
        self.assertAlmostEqual(shares["C00"], 0.20, places=12)
        explorer_mass = sum(shares[c] for c in ids if c != "C00")
        self.assertAlmostEqual(explorer_mass, 0.80, places=12)
        self.assertEqual(sum(1 for c in ids if c != "C00" and shares[c] > 0), 49)
        # Each explorer gets a positive share (reserve + redistributed residual).
        for c in ids:
            if c == "C00":
                continue
            self.assertGreater(shares[c], 0.0)
            dollars = shares[c] * 2000.0
            # Individual explorer slice is typically < $1 here; that is expected
            # when the reserve is split across many cold creatives.
            _ = dollars


class SimulatorEdgeCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        from sim.simulator import Simulator, load_config, load_creatives

        self.cfg = load_config()
        self.creatives = load_creatives()
        self.sim = Simulator(self.cfg, self.creatives, seed=7)
        self.creative_id = str(self.creatives[0]["creative_id"])

    def test_allocation_under_one_dollar_delivers_zero(self) -> None:
        out = self.sim._deliver(61, self.creative_id, 0.99)
        self.assertEqual(out["spend"], 0.0)
        self.assertEqual(out["revenue"], 0.0)

    def test_allocation_of_exactly_one_dollar_is_eligible(self) -> None:
        out = self.sim._deliver(61, self.creative_id, 1.0)
        self.assertGreater(out["spend"], 0.0)

    def test_morning_snapshot_excludes_decision_day_and_applies_lag(self) -> None:
        baseline = [
            {
                "day": 60,
                "creative_id": self.creative_id,
                "budget": 10.0,
                "spend": 10.0,
                "impressions": 100.0,
                "clicks": 1.0,
                "purchases": 1.0,
                "revenue": 100.0,
            }
        ]
        truth = [
            {
                "day": 61,
                "creative_id": self.creative_id,
                "budget": 50.0,
                "spend": 50.0,
                "impressions": 500.0,
                "clicks": 5.0,
                "purchases": 2.0,
                "revenue": 200.0,
            }
        ]
        # Morning of day 62: see day 61 with lag, not day 62; baseline day 60 intact.
        visible = self.sim._visible_history(baseline, truth, decision_day=62)
        days = {int(r["day"]) for r in visible}
        self.assertIn(60, days)
        self.assertIn(61, days)
        self.assertNotIn(62, days)
        lagged = [r for r in visible if int(r["day"]) == 61][0]
        # Age=1 ⇒ configured reported_fraction_by_age["1"].
        frac = self.sim.frac_by_age[1]
        self.assertAlmostEqual(lagged["revenue"], 200.0 * frac, places=12)
        self.assertAlmostEqual(lagged["purchases"], 2.0 * frac, places=12)
        # Eventual (truth) revenue remains unscaled in truth_rows for scoring.
        self.assertAlmostEqual(truth[0]["revenue"], 200.0, places=12)

    def test_policy_run_order_does_not_change_shared_rng_results(self) -> None:
        from baselines import B1Uniform
        from policy import Policy
        from sim.simulator import Simulator

        def revenues(order: list[str]) -> dict[str, float]:
            out: dict[str, float] = {}
            for name in order:
                sim = Simulator(self.cfg, self.creatives, seed=11)
                if name == "Policy":
                    result = sim.run(
                        "Policy",
                        lambda: Policy(self.creatives, 2000.0, 11),
                    )
                else:
                    result = sim.run(
                        "B1",
                        lambda: B1Uniform(self.creatives, 2000.0, 11),
                    )
                out[name] = result.total_revenue
            return out

        a = revenues(["Policy", "B1"])
        b = revenues(["B1", "Policy"])
        self.assertEqual(a["Policy"], b["Policy"])
        self.assertEqual(a["B1"], b["B1"])


if __name__ == "__main__":
    unittest.main()
