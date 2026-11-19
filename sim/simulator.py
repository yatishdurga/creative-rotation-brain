"""Config-driven simulator. Does not import policy internals beyond allocate()."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from sim.rng import noise_multiplier, unit_draw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


class AllocatingPolicy(Protocol):
    def allocate(self, day: int, history: list[dict]) -> dict[str, float]: ...


@dataclass
class RunResult:
    name: str
    seed: int
    n_decisions: int
    total_allocated: float
    total_spend: float
    total_revenue: float
    mean_daily_top5_share: float = 0.0
    days: list[int] = field(default_factory=list)
    daily_top5_shares: list[float] = field(default_factory=list)
    spend_by_creative: dict[str, float] = field(default_factory=dict)
    allocated_by_creative: dict[str, float] = field(default_factory=dict)


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open() as f:
        return json.load(f)


def load_creatives(path: Path | None = None) -> list[dict]:
    creatives_path = path or (ROOT / "data" / "creatives.csv")
    with creatives_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_reported_snapshot(path: Path | None = None) -> list[dict]:
    """Day-60 reported export: history visible on the morning of day 61."""
    reported_path = path or (ROOT / "data" / "daily_reported.csv")
    rows: list[dict] = []
    with reported_path.open(newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "day": int(raw["day"]),
                    "creative_id": str(raw["creative_id"]),
                    "budget": float(raw["budget"]),
                    "spend": float(raw["spend"]),
                    "impressions": float(raw["impressions"]),
                    "clicks": float(raw["clicks"]),
                    "purchases": float(raw["purchases"]),
                    "revenue": float(raw["revenue"]),
                }
            )
    return rows


def normalize_shares(shares: dict[str, float], creative_ids: list[str]) -> dict[str, float]:
    cleaned = {cid: max(float(shares.get(cid, 0.0)), 0.0) for cid in creative_ids}
    total = sum(cleaned.values())
    if total <= 0.0:
        n = len(creative_ids)
        return {cid: 1.0 / n for cid in creative_ids}
    return {cid: value / total for cid, value in cleaned.items()}


class Simulator:
    def __init__(
        self,
        config: dict[str, Any],
        creatives: list[dict],
        seed: int,
        *,
        saturation_k: float | None = None,
    ):
        self.config = config
        self.creatives = list(creatives)
        self.creative_ids = [str(c["creative_id"]) for c in self.creatives]
        self.seed = int(seed)
        self.daily_budget = float(config["daily_budget"])
        self.start_day = int(config["start_day"])
        self.end_day = int(config["end_day"])
        self.min_deliver = float(config["min_deliver_dollars"])
        self.delivery_rate = float(config["delivery_rate"])
        self.roas_noise_cv = float(config["roas_noise_cv"])
        self.delivery_noise_cv = float(config["delivery_noise_cv"])
        self.impression_noise_cv = float(config["impression_noise_cv"])
        self.lag_days = int(config["attribution_lag_days"])
        self.frac_by_age = {
            int(k): float(v) for k, v in config["reported_fraction_by_age"].items()
        }
        self.creative_params = config["creatives"]
        # None / <=0 => linear returns. Positive k => revenue *= k/(k+spend).
        if saturation_k is None:
            saturation_k = config.get("saturation_k")
        self.saturation_k = (
            float(saturation_k) if saturation_k is not None else None
        )
        if self.saturation_k is not None and self.saturation_k <= 0.0:
            self.saturation_k = None

    def run(self, name: str, policy_factory: Callable[[], AllocatingPolicy]) -> RunResult:
        policy = policy_factory()
        # Truth rows for simulated days; reported view is derived each morning.
        baseline_history = load_reported_snapshot()
        truth_rows: list[dict] = []

        total_allocated = 0.0
        total_spend = 0.0
        total_revenue = 0.0
        days_run: list[int] = []
        daily_top5_shares: list[float] = []
        spend_by_creative = {cid: 0.0 for cid in self.creative_ids}
        allocated_by_creative = {cid: 0.0 for cid in self.creative_ids}

        for day in range(self.start_day, self.end_day + 1):
            visible = self._visible_history(baseline_history, truth_rows, day)
            raw_shares = policy.allocate(day, visible)
            shares = normalize_shares(raw_shares, self.creative_ids)
            top5 = sorted(shares.values(), reverse=True)[:5]
            daily_top5_shares.append(float(sum(top5)))

            for cid in self.creative_ids:
                budget = shares[cid] * self.daily_budget
                outcome = self._deliver(day, cid, budget)
                truth_rows.append(
                    {
                        "day": day,
                        "creative_id": cid,
                        "budget": budget,
                        "spend": outcome["spend"],
                        "impressions": outcome["impressions"],
                        "clicks": outcome["clicks"],
                        "purchases": outcome["purchases"],
                        "revenue": outcome["revenue"],
                    }
                )
                total_allocated += budget
                total_spend += outcome["spend"]
                total_revenue += outcome["revenue"]
                spend_by_creative[cid] += outcome["spend"]
                allocated_by_creative[cid] += budget

            days_run.append(day)

        mean_top5 = (
            sum(daily_top5_shares) / len(daily_top5_shares) if daily_top5_shares else 0.0
        )
        return RunResult(
            name=name,
            seed=self.seed,
            n_decisions=len(days_run),
            total_allocated=total_allocated,
            total_spend=total_spend,
            total_revenue=total_revenue,
            mean_daily_top5_share=mean_top5,
            days=days_run,
            daily_top5_shares=daily_top5_shares,
            spend_by_creative=spend_by_creative,
            allocated_by_creative=allocated_by_creative,
        )

    def _visible_history(
        self,
        baseline: list[dict],
        truth_rows: list[dict],
        decision_day: int,
    ) -> list[dict]:
        """Rows with day < decision_day, revenue/purchases as currently reported."""
        visible: list[dict] = []
        for row in baseline:
            if row["day"] >= decision_day:
                continue
            visible.append(self._reported_view(row, decision_day, lag_applies=False))
        for row in truth_rows:
            if row["day"] >= decision_day:
                continue
            visible.append(self._reported_view(row, decision_day, lag_applies=True))
        return visible

    def _reported_view(self, row: dict, decision_day: int, *, lag_applies: bool) -> dict:
        out = deepcopy(row)
        if not lag_applies:
            # Baseline CSV is already a morning-of-61 reported snapshot.
            return out
        age = decision_day - int(row["day"])
        frac = self._reported_fraction(age)
        out["revenue"] = float(row["revenue"]) * frac
        out["purchases"] = float(row["purchases"]) * frac
        return out

    def _reported_fraction(self, age: int) -> float:
        if age >= self.lag_days:
            return 1.0
        if age <= 0:
            return 0.0
        if age in self.frac_by_age:
            return self.frac_by_age[age]
        # Linear bridge between configured ages if a key is missing.
        return min(1.0, age / float(self.lag_days))

    def _deliver(self, day: int, creative_id: str, budget: float) -> dict[str, float]:
        if budget < self.min_deliver:
            return {
                "spend": 0.0,
                "impressions": 0.0,
                "clicks": 0.0,
                "purchases": 0.0,
                "revenue": 0.0,
            }

        params = self.creative_params[creative_id]
        base_roas = float(params["roas"])
        ips = float(params["impressions_per_dollar"])
        ctr = float(params["ctr"])
        aov = float(params["aov"])

        delivery_u = unit_draw(self.seed, day, creative_id, "delivery")
        roas_u = unit_draw(self.seed, day, creative_id, "roas")
        impr_u = unit_draw(self.seed, day, creative_id, "impressions")

        delivery_mult = noise_multiplier(delivery_u, self.delivery_noise_cv)
        spend = min(budget, budget * self.delivery_rate * delivery_mult)
        spend = max(0.0, spend)

        roas = max(0.0, base_roas * noise_multiplier(roas_u, self.roas_noise_cv))
        if self.saturation_k is None:
            revenue = spend * roas
        else:
            k = self.saturation_k
            revenue = spend * roas * (k / (k + spend)) if spend > 0.0 else 0.0
        impressions = max(0.0, spend * ips * noise_multiplier(impr_u, self.impression_noise_cv))
        clicks = max(0.0, impressions * ctr)
        purchases = (revenue / aov) if aov > 0.0 else 0.0

        return {
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "purchases": purchases,
            "revenue": revenue,
        }
