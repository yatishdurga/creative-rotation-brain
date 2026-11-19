"""Evaluation baselines: B1 uniform and B2 windowed greedy.

Population mean rate (B2)
-------------------------
Over the trailing window [day-7, day-1], let R be the sum of reported
revenue and S the sum of reported spend across every creative in the
snapshot (including creatives that later fall back). The population mean
rate is R/S when S > 0, otherwise 0.0.

Creatives with zero trailing spend or fewer than 2,000 trailing
impressions are assigned that population mean instead of their own
revenue/spend. Ranking then uses those assigned rates.
"""

from __future__ import annotations

from typing import Any


TOP_K = 5
TOP_SHARE = 0.75
REST_SHARE = 0.25
MIN_IMPRESSIONS = 2000
WINDOW_DAYS = 7


def _creative_ids(creatives: list[dict]) -> list[str]:
    return [str(c["creative_id"]) for c in creatives]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


class B1Uniform:
    """Equal share to every creative, every day."""

    def __init__(self, creatives: list[dict], daily_budget: float = 2000.0, seed: int = 0):
        if not creatives:
            raise ValueError("creatives must be non-empty")
        self.creatives = list(creatives)
        self.creative_ids = _creative_ids(self.creatives)
        self.daily_budget = float(daily_budget)
        self.seed = int(seed)

    def allocate(self, day: int, history: list[dict]) -> dict[str, float]:
        _ = day, history
        n = len(self.creative_ids)
        share = 1.0 / n
        return {cid: share for cid in self.creative_ids}


class B2WindowedGreedy:
    """Trailing-7-day revenue/spend ranking with 75/25 top-5 split."""

    def __init__(self, creatives: list[dict], daily_budget: float = 2000.0, seed: int = 0):
        if not creatives:
            raise ValueError("creatives must be non-empty")
        self.creatives = list(creatives)
        self.creative_ids = _creative_ids(self.creatives)
        self.daily_budget = float(daily_budget)
        self.seed = int(seed)

    def allocate(self, day: int, history: list[dict]) -> dict[str, float]:
        ids = self.creative_ids
        n = len(ids)
        spend, revenue, impressions = self._window_totals(day, history)

        total_spend = sum(spend.values())
        total_revenue = sum(revenue.values())
        population_mean = (total_revenue / total_spend) if total_spend > 0.0 else 0.0

        rates: dict[str, float] = {}
        for cid in ids:
            trailing_spend = spend[cid]
            trailing_impr = impressions[cid]
            if trailing_spend <= 0.0 or trailing_impr < MIN_IMPRESSIONS:
                rates[cid] = population_mean
            else:
                rates[cid] = revenue[cid] / trailing_spend

        # Deterministic ties: higher rate first, then creative_id ascending.
        ranked = sorted(ids, key=lambda cid: (-rates[cid], cid))
        top_n = min(TOP_K, n)
        top = ranked[:top_n]
        rest = ranked[top_n:]

        shares = {cid: 0.0 for cid in ids}
        if top:
            # If fewer than TOP_K creatives exist, they still split the top mass.
            each_top = TOP_SHARE / len(top)
            for cid in top:
                shares[cid] = each_top
        if rest:
            each_rest = REST_SHARE / len(rest)
            for cid in rest:
                shares[cid] = each_rest
        elif top:
            # n <= TOP_K: everyone already received TOP_SHARE/n; renormalize to 1.
            each = 1.0 / len(top)
            for cid in top:
                shares[cid] = each

        return shares

    def _window_totals(
        self, day: int, history: list[dict]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Aggregate spend, revenue, impressions for days in [day-7, day-1]."""
        lo = day - WINDOW_DAYS
        hi = day - 1
        known = set(self.creative_ids)
        spend = {cid: 0.0 for cid in self.creative_ids}
        revenue = {cid: 0.0 for cid in self.creative_ids}
        impressions = {cid: 0.0 for cid in self.creative_ids}

        for row in history:
            if not isinstance(row, dict):
                continue
            try:
                row_day = int(float(row["day"]))
            except (KeyError, TypeError, ValueError):
                continue
            if row_day < lo or row_day > hi:
                continue
            cid = str(row.get("creative_id", ""))
            if cid not in known:
                continue
            spend[cid] += _finite_float(row.get("spend", 0.0))
            revenue[cid] += _finite_float(row.get("revenue", 0.0))
            impressions[cid] += _finite_float(row.get("impressions", 0.0))

        return spend, revenue, impressions
