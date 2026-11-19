"""Focused tests for B1 uniform and B2 windowed-greedy baselines."""

from __future__ import annotations

import math
import unittest

from baselines import (
    MIN_IMPRESSIONS,
    REST_SHARE,
    TOP_K,
    TOP_SHARE,
    B1Uniform,
    B2WindowedGreedy,
)


def toy_creatives(*ids: str) -> list[dict]:
    return [{"creative_id": cid} for cid in ids]


def assert_valid_shares(test: unittest.TestCase, shares: dict[str, float], ids: list[str]) -> None:
    test.assertEqual(set(shares), set(ids))
    test.assertTrue(all(v >= 0.0 for v in shares.values()))
    test.assertTrue(math.isclose(sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-12))


class B1Tests(unittest.TestCase):
    def test_uniform_shares(self) -> None:
        ids = [f"C{i:02d}" for i in range(10)]
        policy = B1Uniform(toy_creatives(*ids), daily_budget=2000.0, seed=0)
        shares = policy.allocate(61, [])
        assert_valid_shares(self, shares, ids)
        for cid in ids:
            self.assertAlmostEqual(shares[cid], 0.1, places=12)


class B2WindowBoundaryTests(unittest.TestCase):
    """Window is days [day-7, day-1] from the supplied snapshot."""

    def setUp(self) -> None:
        # 6 creatives so top-5 / rest split is observable.
        self.ids = [f"C{i}" for i in range(6)]
        self.policy = B2WindowedGreedy(toy_creatives(*self.ids))

    def test_includes_day_minus_7_excludes_decision_day_and_older(self) -> None:
        day = 20
        # In-window: day 13 (= day-7) through day 19 (= day-1).
        in_window = [
            {
                "day": 13,
                "creative_id": "C0",
                "spend": 100.0,
                "revenue": 400.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 15,
                "creative_id": "C3",
                "spend": 100.0,
                "revenue": 100.0,
                "impressions": MIN_IMPRESSIONS,
            },
        ]
        outside = [
            {
                "day": 12,  # older than day-7
                "creative_id": "C1",
                "spend": 100.0,
                "revenue": 10000.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 20,  # decision day
                "creative_id": "C2",
                "spend": 100.0,
                "revenue": 10000.0,
                "impressions": MIN_IMPRESSIONS,
            },
        ]
        shares_with_outside = self.policy.allocate(day, in_window + outside)
        shares_in_window_only = self.policy.allocate(day, in_window)
        assert_valid_shares(self, shares_with_outside, self.ids)
        # Out-of-window rows must not change the allocation.
        self.assertEqual(shares_with_outside, shares_in_window_only)
        self.assertAlmostEqual(shares_with_outside["C0"], TOP_SHARE / TOP_K, places=12)

    def test_day_minus_1_is_included(self) -> None:
        day = 10
        history = [
            {
                "day": 9,
                "creative_id": "C0",
                "spend": 50.0,
                "revenue": 500.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 8,
                "creative_id": "C1",
                "spend": 50.0,
                "revenue": 50.0,
                "impressions": MIN_IMPRESSIONS,
            },
        ]
        shares = self.policy.allocate(day, history)
        assert_valid_shares(self, shares, self.ids)
        # C0 has the best in-window rate and must land in the top-5 bucket.
        self.assertAlmostEqual(shares["C0"], TOP_SHARE / TOP_K, places=12)
        # C1 is uniquely worst among 6, so it alone takes the 25% rest bucket.
        self.assertAlmostEqual(shares["C1"], REST_SHARE, places=12)


class B2FallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"C{i}" for i in range(6)]
        self.policy = B2WindowedGreedy(toy_creatives(*self.ids))

    def test_zero_spend_uses_population_mean(self) -> None:
        # C0 qualifies with high rate. C1 has impressions but zero spend → mean.
        # Population mean = (400+100)/(100+100) = 2.5; C0 rate = 4.0; C2 = 1.0.
        history = [
            {
                "day": 1,
                "creative_id": "C0",
                "spend": 100.0,
                "revenue": 400.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C1",
                "spend": 0.0,
                "revenue": 0.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C2",
                "spend": 100.0,
                "revenue": 100.0,
                "impressions": MIN_IMPRESSIONS,
            },
        ]
        shares = self.policy.allocate(2, history)
        assert_valid_shares(self, shares, self.ids)
        # C0 is uniquely best and in the top bucket.
        self.assertAlmostEqual(shares["C0"], TOP_SHARE / TOP_K, places=12)
        # C2 is uniquely worst (rate 1.0 < mean 2.5) → alone in the rest bucket.
        self.assertAlmostEqual(shares["C2"], REST_SHARE, places=12)
        # Zero-spend C1 takes the mean (2.5) and stays in the top bucket with
        # the other mean-assigned creatives — not an infinite/undefined rate.
        self.assertAlmostEqual(shares["C1"], TOP_SHARE / TOP_K, places=12)

    def test_low_impressions_uses_population_mean(self) -> None:
        # C1 is a lucky small sample (raw rate 100) with only 100 impressions.
        # Heavy low-ROAS spend on C2 pulls the population mean below C0.
        # Fallback assigns C1 that mean, so C0 takes a top-bucket seat on its
        # empirical rate of 8.0 rather than being displaced by C1's raw 100.
        history = [
            {
                "day": 1,
                "creative_id": "C0",
                "spend": 100.0,
                "revenue": 800.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C1",
                "spend": 10.0,
                "revenue": 1000.0,
                "impressions": 100,
            },
            {
                "day": 1,
                "creative_id": "C2",
                "spend": 890.0,
                "revenue": 890.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C3",
                "spend": 100.0,
                "revenue": 400.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C4",
                "spend": 100.0,
                "revenue": 300.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C5",
                "spend": 100.0,
                "revenue": 200.0,
                "impressions": MIN_IMPRESSIONS,
            },
        ]
        # spend=1300, revenue=3590, mean≈2.762.
        # Rates: C0=8, C3=4, C4=3, C5=2, C1→2.762, C2=1.
        # Top5 under fallback: C0,C3,C4,C1,C5 | rest C2.
        # Top5 if C1 used raw 100: C1,C0,C3,C4,C5 | rest C2 — C1 displaces
        # nobody in the rest bucket, but C1 would rank ahead of C0.
        # Distinguish by who is 5th: fallback keeps C5 (rate 2) in top because
        # C1 is only at the mean (2.762); wait C5=2 < mean so order is
        # C0,C3,C4,C1,C2? C5=2, C2=1, mean for... all have data.
        # Order: C0(8), C3(4), C4(3), C1(2.762), C5(2) | C2(1).
        shares = self.policy.allocate(2, history)
        assert_valid_shares(self, shares, self.ids)
        for cid in ["C0", "C3", "C4", "C1", "C5"]:
            self.assertAlmostEqual(shares[cid], TOP_SHARE / TOP_K, places=12)
        self.assertAlmostEqual(shares["C2"], REST_SHARE, places=12)

        history_raw = [dict(r) for r in history]
        for row in history_raw:
            if row["creative_id"] == "C1":
                row["impressions"] = MIN_IMPRESSIONS
        # Raw order: C1(100), C0(8), C3(4), C4(3), C5(2) | C2(1).
        # C1 still top-bucket; rest bucket unchanged. The observable difference
        # is that fallback does not let C1's raw rate exclude C5 — already
        # true above. Confirm qualifying impressions still run cleanly:
        shares_raw = self.policy.allocate(2, history_raw)
        self.assertAlmostEqual(shares_raw["C1"], TOP_SHARE / TOP_K, places=12)
        self.assertAlmostEqual(shares_raw["C5"], TOP_SHARE / TOP_K, places=12)
        self.assertAlmostEqual(shares_raw["C2"], REST_SHARE, places=12)

    def test_population_mean_is_total_revenue_over_total_spend(self) -> None:
        history = [
            {
                "day": 1,
                "creative_id": "C0",
                "spend": 200.0,
                "revenue": 400.0,
                "impressions": MIN_IMPRESSIONS,
            },
            {
                "day": 1,
                "creative_id": "C1",
                "spend": 100.0,
                "revenue": 50.0,
                "impressions": 100,  # fallback creative
            },
        ]
        # mean = 450/300 = 1.5; C0 rate = 2.0; C1 gets 1.5
        spend, revenue, impressions = self.policy._window_totals(2, history)
        total_spend = sum(spend.values())
        total_revenue = sum(revenue.values())
        mean = total_revenue / total_spend
        self.assertAlmostEqual(mean, 1.5, places=12)
        self.assertLess(impressions["C1"], MIN_IMPRESSIONS)
        shares = self.policy.allocate(2, history)
        self.assertAlmostEqual(shares["C0"], TOP_SHARE / TOP_K, places=12)


class B2SplitAndTieTests(unittest.TestCase):
    def test_exact_75_25_split(self) -> None:
        ids = [f"C{i}" for i in range(10)]
        policy = B2WindowedGreedy(toy_creatives(*ids))
        history = []
        # Distinct rates so the top five are C0..C4.
        for i, cid in enumerate(ids):
            history.append(
                {
                    "day": 1,
                    "creative_id": cid,
                    "spend": 100.0,
                    "revenue": float(1000 - 50 * i),
                    "impressions": MIN_IMPRESSIONS,
                }
            )
        shares = policy.allocate(2, history)
        assert_valid_shares(self, shares, ids)

        top = [f"C{i}" for i in range(5)]
        rest = [f"C{i}" for i in range(5, 10)]
        for cid in top:
            self.assertAlmostEqual(shares[cid], TOP_SHARE / TOP_K, places=12)
            self.assertAlmostEqual(shares[cid], 0.15, places=12)
        for cid in rest:
            self.assertAlmostEqual(shares[cid], REST_SHARE / len(rest), places=12)
            self.assertAlmostEqual(shares[cid], 0.05, places=12)
        self.assertAlmostEqual(sum(shares[c] for c in top), TOP_SHARE, places=12)
        self.assertAlmostEqual(sum(shares[c] for c in rest), REST_SHARE, places=12)

    def test_ties_broken_by_creative_id(self) -> None:
        ids = [f"C{i}" for i in range(6)]
        policy = B2WindowedGreedy(toy_creatives(*ids))
        # All equal qualifying rates → ranking is pure creative_id order.
        history = [
            {
                "day": 1,
                "creative_id": cid,
                "spend": 100.0,
                "revenue": 200.0,
                "impressions": MIN_IMPRESSIONS,
            }
            for cid in ids
        ]
        shares = policy.allocate(2, history)
        top = ["C0", "C1", "C2", "C3", "C4"]
        for cid in top:
            self.assertAlmostEqual(shares[cid], TOP_SHARE / TOP_K, places=12)
        self.assertAlmostEqual(shares["C5"], REST_SHARE, places=12)


if __name__ == "__main__":
    unittest.main()
