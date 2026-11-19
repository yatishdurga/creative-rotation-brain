"""Tests for Policy: uniform fallback and revenue-per-dollar ranking."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

from policy import Policy

ROOT = Path(__file__).resolve().parents[1]
CREATIVES_CSV = ROOT / "data" / "creatives.csv"


def load_creatives() -> list[dict]:
    with CREATIVES_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def toy_creatives(*ids: str) -> list[dict]:
    return [{"creative_id": cid} for cid in ids]


def assert_valid_shares(test: unittest.TestCase, shares: dict[str, float], ids: list[str]) -> None:
    test.assertEqual(set(shares), set(ids))
    test.assertTrue(all(v >= 0.0 for v in shares.values()))
    test.assertTrue(math.isclose(sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-12))


class UniformFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.creatives = load_creatives()
        self.ids = [str(c["creative_id"]) for c in self.creatives]
        self.policy = Policy(self.creatives, daily_budget=2000.0, seed=7)

    def test_empty_history_is_uniform(self) -> None:
        shares = self.policy.allocate(61, [])
        assert_valid_shares(self, shares, self.ids)
        each = 1.0 / len(self.ids)
        for cid in self.ids:
            self.assertAlmostEqual(shares[cid], each, places=12)

    def test_allocate_is_deterministic(self) -> None:
        a = self.policy.allocate(61, [])
        b = self.policy.allocate(61, [])
        self.assertEqual(a, b)


class RankingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use n > 5 so effective cap stays at 20% and ranking can differentiate.
        self.ids = ["STRONG", "WEAK", "NEW", "MID_A", "MID_B", "MID_C"]
        self.creatives = toy_creatives(*self.ids)
        self.policy = Policy(
            self.creatives,
            daily_budget=2000.0,
            seed=7,
            shrink_spend=200.0,
            test_budget_fraction=0.10,
            low_spend_threshold=50.0,
        )

    def _history(self) -> list[dict]:
        return [
            {
                "day": 1,
                "creative_id": "STRONG",
                "budget": 500.0,
                "spend": 500.0,
                "impressions": 10000,
                "clicks": 100,
                "purchases": 20,
                "revenue": 2000.0,
            },
            {
                "day": 1,
                "creative_id": "WEAK",
                "budget": 500.0,
                "spend": 500.0,
                "impressions": 10000,
                "clicks": 50,
                "purchases": 2,
                "revenue": 100.0,
            },
            {
                "day": 1,
                "creative_id": "MID_A",
                "budget": 200.0,
                "spend": 200.0,
                "impressions": 4000,
                "clicks": 40,
                "purchases": 4,
                "revenue": 400.0,
            },
            {
                "day": 1,
                "creative_id": "MID_B",
                "budget": 200.0,
                "spend": 200.0,
                "impressions": 4000,
                "clicks": 40,
                "purchases": 4,
                "revenue": 400.0,
            },
            {
                "day": 1,
                "creative_id": "MID_C",
                "budget": 200.0,
                "spend": 200.0,
                "impressions": 4000,
                "clicks": 40,
                "purchases": 4,
                "revenue": 400.0,
            },
        ]

    def test_valid_shares_with_history(self) -> None:
        shares = self.policy.allocate(2, self._history())
        assert_valid_shares(self, shares, self.ids)
        effective_cap = max(0.20, 1.0 / len(self.ids))
        for share in shares.values():
            self.assertLessEqual(share, effective_cap + 1e-12)

    def test_strong_beats_weak(self) -> None:
        shares = self.policy.allocate(2, self._history())
        self.assertGreater(shares["STRONG"], shares["WEAK"])

    def test_new_creative_gets_positive_test_spend(self) -> None:
        shares = self.policy.allocate(2, self._history())
        dollars = shares["NEW"] * self.policy.daily_budget
        self.assertGreater(shares["NEW"], 0.0)
        self.assertGreaterEqual(dollars, 1.0)

    def test_revised_history_row_updates_allocation(self) -> None:
        history = self._history()
        before = self.policy.allocate(2, history)

        revised = [dict(row) for row in history]
        for row in revised:
            if row["creative_id"] == "WEAK":
                row["revenue"] = 5000.0
                row["purchases"] = 50

        after = self.policy.allocate(2, revised)
        self.assertGreater(after["WEAK"], before["WEAK"])
        self.assertLess(after["STRONG"], before["STRONG"])
        assert_valid_shares(self, after, self.ids)

    def test_future_day_rows_are_ignored(self) -> None:
        history = self._history() + [
            {
                "day": 5,
                "creative_id": "WEAK",
                "budget": 500.0,
                "spend": 500.0,
                "impressions": 1,
                "clicks": 1,
                "purchases": 50,
                "revenue": 10000.0,
            }
        ]
        shares = self.policy.allocate(2, history)
        # Day-5 row must not flip the ranking for a day-2 decision.
        self.assertGreater(shares["STRONG"], shares["WEAK"])

    def test_zero_spend_history_falls_back_to_uniform(self) -> None:
        history = [
            {
                "day": 1,
                "creative_id": "STRONG",
                "budget": 0.0,
                "spend": 0.0,
                "impressions": 0,
                "clicks": 0,
                "purchases": 0,
                "revenue": 0.0,
            }
        ]
        shares = self.policy.allocate(2, history)
        for cid in self.ids:
            self.assertAlmostEqual(shares[cid], 1.0 / len(self.ids), places=12)


class RecentRowWeightTests(unittest.TestCase):
    def test_recent_rows_half_weight_scales_spend_and_revenue(self) -> None:
        creatives = toy_creatives("A", "B")
        policy = Policy(
            creatives,
            daily_budget=2000.0,
            seed=0,
            recent_day_weight=0.5,
            recent_day_horizon=3,
            test_budget_fraction=0.0,
            low_spend_threshold=0.0,
        )
        history = [
            {"day": 10, "creative_id": "A", "spend": 100.0, "revenue": 200.0},  # age 1
            {"day": 5, "creative_id": "B", "spend": 100.0, "revenue": 200.0},  # age 6
        ]
        raw, weighted_spend, weighted_rev = policy._aggregate(11, history)
        self.assertAlmostEqual(raw["A"], 100.0)
        self.assertAlmostEqual(raw["B"], 100.0)
        self.assertAlmostEqual(weighted_spend["A"], 50.0)
        self.assertAlmostEqual(weighted_rev["A"], 100.0)
        self.assertAlmostEqual(weighted_spend["B"], 100.0)
        self.assertAlmostEqual(weighted_rev["B"], 200.0)

    def test_default_weight_matches_unweighted(self) -> None:
        creatives = toy_creatives("A")
        policy = Policy(creatives, daily_budget=2000.0, seed=0)
        history = [{"day": 10, "creative_id": "A", "spend": 80.0, "revenue": 160.0}]
        raw, weighted_spend, weighted_rev = policy._aggregate(11, history)
        self.assertEqual(raw, weighted_spend)
        self.assertAlmostEqual(weighted_rev["A"], 160.0)


class SafetyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        # n=6 keeps effective cap at 20% so ranking claims stay testable.
        self.ids = ["STRONG", "WEAK", "NEW", "MID_A", "MID_B", "MID_C"]
        self.creatives = toy_creatives(*self.ids)
        self.policy = Policy(
            self.creatives,
            daily_budget=2000.0,
            seed=7,
            shrink_spend=200.0,
            test_budget_fraction=0.10,
            low_spend_threshold=50.0,
            max_share=0.20,
        )

    def _valid_history(self) -> list[dict]:
        return [
            {
                "day": 1,
                "creative_id": "STRONG",
                "budget": 500.0,
                "spend": 500.0,
                "impressions": 10000,
                "clicks": 100,
                "purchases": 20,
                "revenue": 2000.0,
            },
            {
                "day": 1,
                "creative_id": "WEAK",
                "budget": 500.0,
                "spend": 500.0,
                "impressions": 10000,
                "clicks": 50,
                "purchases": 2,
                "revenue": 100.0,
            },
            {
                "day": 1,
                "creative_id": "MID_A",
                "spend": 200.0,
                "revenue": 400.0,
            },
            {
                "day": 1,
                "creative_id": "MID_B",
                "spend": 200.0,
                "revenue": 400.0,
            },
            {
                "day": 1,
                "creative_id": "MID_C",
                "spend": 200.0,
                "revenue": 400.0,
            },
        ]

    def test_malformed_rows_are_skipped_valid_rows_used(self) -> None:
        history: list[object] = [
            "not-a-row",
            {"day": "bad", "creative_id": "STRONG", "spend": 500.0, "revenue": 2000.0},
            {"creative_id": "STRONG", "spend": 500.0, "revenue": 2000.0},  # missing day
            {"day": 1, "spend": 500.0, "revenue": 2000.0},  # missing creative_id
            {"day": 1, "creative_id": "STRONG", "spend": "nope", "revenue": 2000.0},
            *self._valid_history(),
        ]
        shares = self.policy.allocate(2, history)  # type: ignore[arg-type]
        assert_valid_shares(self, shares, self.ids)
        self.assertGreater(shares["STRONG"], shares["WEAK"])
        self.assertGreater(shares["NEW"], 0.0)

    def test_nonfinite_rows_are_skipped(self) -> None:
        history = [
            {
                "day": 1,
                "creative_id": "STRONG",
                "spend": float("nan"),
                "revenue": 2000.0,
            },
            {
                "day": 1,
                "creative_id": "WEAK",
                "spend": 500.0,
                "revenue": float("inf"),
            },
            {
                "day": 1,
                "creative_id": "STRONG",
                "spend": float("-inf"),
                "revenue": 100.0,
            },
            *self._valid_history(),
        ]
        shares = self.policy.allocate(2, history)
        assert_valid_shares(self, shares, self.ids)
        self.assertGreater(shares["STRONG"], shares["WEAK"])

    def test_only_unusable_rows_fall_back_to_uniform(self) -> None:
        history: list[object] = [
            {"day": 1, "creative_id": "STRONG", "spend": float("nan"), "revenue": 10.0},
            {"day": "x", "creative_id": "WEAK", "spend": 10.0, "revenue": 10.0},
            None,
        ]
        shares = self.policy.allocate(2, history)  # type: ignore[arg-type]
        for cid in self.ids:
            self.assertAlmostEqual(shares[cid], 1.0 / len(self.ids), places=12)

    def test_single_exploiter_respects_effective_cap(self) -> None:
        # Small fixture: only STRONG is above the spend threshold.
        ids = ["STRONG", "WEAK", "NEW"]
        policy = Policy(
            toy_creatives(*ids),
            daily_budget=2000.0,
            seed=7,
            test_budget_fraction=0.10,
            low_spend_threshold=50.0,
            max_share=0.20,
        )
        history = [
            {
                "day": 1,
                "creative_id": "STRONG",
                "budget": 1000.0,
                "spend": 1000.0,
                "impressions": 20000,
                "clicks": 200,
                "purchases": 40,
                "revenue": 8000.0,
            }
        ]
        shares = policy.allocate(2, history)
        effective_cap = max(0.20, 1.0 / len(ids))
        assert_valid_shares(self, shares, ids)
        for share in shares.values():
            self.assertLessEqual(share, effective_cap + 1e-12)
        self.assertGreater(shares["NEW"], 0.0)
        self.assertGreater(shares["WEAK"], 0.0)
        # With n=3 the floor is 1/3, so the lone exploiter is clipped there.
        self.assertAlmostEqual(shares["STRONG"], effective_cap, places=12)

    def test_fifty_creative_cap_is_twenty_percent(self) -> None:
        creatives = toy_creatives(*[f"C{i:02d}" for i in range(50)])
        policy = Policy(creatives, daily_budget=2000.0, seed=7)
        history = [
            {
                "day": 1,
                "creative_id": "C00",
                "spend": 1000.0,
                "revenue": 10000.0,
            },
            {
                "day": 1,
                "creative_id": "C01",
                "spend": 1000.0,
                "revenue": 100.0,
            },
        ]
        shares = policy.allocate(2, history)
        assert_valid_shares(self, shares, [str(c["creative_id"]) for c in creatives])
        for share in shares.values():
            self.assertLessEqual(share, 0.20 + 1e-12)
        self.assertAlmostEqual(shares["C00"], 0.20, places=12)


if __name__ == "__main__":
    unittest.main()
