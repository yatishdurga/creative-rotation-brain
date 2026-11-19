"""Performance-based budget-split policy.

Ranks creatives by revenue per delivered dollar, shrinks low-spend estimates
toward the account average, and reserves a bounded test budget for creatives
with little or no spend. Recomputes from the supplied history on every call.
"""

from __future__ import annotations

import math
from typing import Any


class Policy:
    # Defaults are overridable via __init__ kwargs.
    # shrink_spend=0 keeps the empirical revenue/spend rate.
    # Pass shrink_spend=200.0 to pull low-spend rates toward the account average.
    DEFAULT_SHRINK_SPEND = 0.0
    DEFAULT_TEST_BUDGET_FRACTION = 0.10
    DEFAULT_LOW_SPEND_THRESHOLD = 50.0
    DEFAULT_MAX_SHARE = 0.20
    DEFAULT_ALLOCATION_MODE = "proportional"
    DEFAULT_TOP_K = 5
    DEFAULT_TOP_SHARE = 0.75
    DEFAULT_RECENT_DAY_WEIGHT = 1.0
    DEFAULT_RECENT_DAY_HORIZON = 3

    def __init__(
        self,
        creatives: list[dict],
        daily_budget: float,
        seed: int,
        *,
        shrink_spend: float | None = None,
        test_budget_fraction: float | None = None,
        low_spend_threshold: float | None = None,
        max_share: float | None = None,
        allocation_mode: str | None = None,
        top_k: int | None = None,
        top_share: float | None = None,
        recent_day_weight: float | None = None,
        recent_day_horizon: int | None = None,
    ):
        """creatives: rows of creatives.csv, one dict per row.

        allocation_mode:
          - "proportional": exploit mass ∝ shrunk revenue/spend (default).
          - "concentrated_top5": top-k eligible creatives split top_share of
            the daily budget equally; remaining exploit mass is split equally
            among the other eligible creatives.

        recent_day_weight / recent_day_horizon:
          Rows with age in 1..horizon (age = decision_day - row_day) scale
          both revenue and spend by recent_day_weight before aggregation.
          Default weight 1.0 leaves behavior unchanged.
        """
        if not creatives:
            raise ValueError("creatives must be non-empty")

        self.creatives = list(creatives)
        self.creative_ids = [str(c["creative_id"]) for c in self.creatives]
        self.daily_budget = float(daily_budget)
        self.seed = int(seed)

        self.shrink_spend = float(
            self.DEFAULT_SHRINK_SPEND if shrink_spend is None else shrink_spend
        )
        self.test_budget_fraction = float(
            self.DEFAULT_TEST_BUDGET_FRACTION
            if test_budget_fraction is None
            else test_budget_fraction
        )
        self.low_spend_threshold = float(
            self.DEFAULT_LOW_SPEND_THRESHOLD
            if low_spend_threshold is None
            else low_spend_threshold
        )
        self.max_share = float(self.DEFAULT_MAX_SHARE if max_share is None else max_share)
        self.allocation_mode = (
            self.DEFAULT_ALLOCATION_MODE if allocation_mode is None else str(allocation_mode)
        )
        self.top_k = int(self.DEFAULT_TOP_K if top_k is None else top_k)
        self.top_share = float(self.DEFAULT_TOP_SHARE if top_share is None else top_share)
        self.recent_day_weight = float(
            self.DEFAULT_RECENT_DAY_WEIGHT
            if recent_day_weight is None
            else recent_day_weight
        )
        self.recent_day_horizon = int(
            self.DEFAULT_RECENT_DAY_HORIZON
            if recent_day_horizon is None
            else recent_day_horizon
        )

        if self.shrink_spend < 0:
            raise ValueError("shrink_spend must be >= 0")
        if not 0.0 <= self.test_budget_fraction <= 1.0:
            raise ValueError("test_budget_fraction must be in [0, 1]")
        if self.low_spend_threshold < 0:
            raise ValueError("low_spend_threshold must be >= 0")
        if not 0.0 < self.max_share <= 1.0:
            raise ValueError("max_share must be in (0, 1]")
        if self.allocation_mode not in {"proportional", "concentrated_top5"}:
            raise ValueError(
                "allocation_mode must be 'proportional' or 'concentrated_top5'"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 < self.top_share <= 1.0:
            raise ValueError("top_share must be in (0, 1]")
        if self.recent_day_weight < 0.0:
            raise ValueError("recent_day_weight must be >= 0")
        if self.recent_day_horizon < 0:
            raise ValueError("recent_day_horizon must be >= 0")

    def allocate(self, day: int, history: list[dict]) -> dict[str, float]:
        """Returns {creative_id: share} from history rows with day < decision day."""
        ids = self.creative_ids
        n = len(ids)
        uniform = {cid: 1.0 / n for cid in ids}
        effective_cap = max(self.max_share, 1.0 / n)

        spend_raw, spend, revenue = self._aggregate(day, history)
        total_spend = sum(spend.values())
        if total_spend <= 0.0:
            return uniform

        total_revenue = sum(revenue.values())
        if not math.isfinite(total_revenue):
            return uniform
        account_avg = total_revenue / total_spend
        if not math.isfinite(account_avg):
            return uniform

        rates = {
            cid: self._shrunk_rate(spend[cid], revenue[cid], account_avg) for cid in ids
        }

        # Explorer eligibility uses unweighted delivered spend.
        explorers = [cid for cid in ids if spend_raw[cid] < self.low_spend_threshold]
        exploiters = [cid for cid in ids if spend_raw[cid] >= self.low_spend_threshold]

        # Assign exploration first so the reserve survives later cap redistribution.
        shares = {cid: 0.0 for cid in ids}
        reserved = {cid: 0.0 for cid in ids}

        if explorers and self.test_budget_fraction > 0.0:
            raw_each = self.test_budget_fraction / len(explorers)
            for cid in explorers:
                reserved[cid] = min(raw_each, effective_cap)
                shares[cid] = reserved[cid]
            assigned_test = sum(reserved[cid] for cid in explorers)
        else:
            assigned_test = 0.0

        exploit_frac = max(0.0, 1.0 - assigned_test)

        if exploiters and exploit_frac > 0.0:
            if self.allocation_mode == "concentrated_top5":
                self._add_concentrated_top_shares(
                    shares, exploiters, rates, exploit_frac, assigned_test
                )
            else:
                self._add_weighted_shares(shares, exploiters, rates, exploit_frac)
        elif exploit_frac > 0.0 and explorers:
            each = exploit_frac / len(explorers)
            for cid in explorers:
                shares[cid] += each

        shares = self._normalize(shares)
        shares = self._apply_share_cap(shares, reserved=reserved, cap=effective_cap)
        return self._normalize(shares)

    def _add_concentrated_top_shares(
        self,
        shares: dict[str, float],
        eligible: list[str],
        rates: dict[str, float],
        exploit_frac: float,
        assigned_test: float,
    ) -> None:
        """Give top-k eligible creatives up to top_share of the daily budget."""
        ranked = sorted(eligible, key=lambda cid: (-rates[cid], cid))
        top = ranked[: min(self.top_k, len(ranked))]
        rest = ranked[len(top) :]

        # Subject to reserve: cannot spend more than remaining mass on the top tier.
        top_mass = min(self.top_share, exploit_frac)
        rest_mass = max(0.0, exploit_frac - top_mass)
        # If reserve already consumed more than (1 - top_share), top_mass may be
        # below top_share; that is intentional ("subject to the reserve").

        if top:
            each_top = top_mass / len(top)
            for cid in top:
                shares[cid] += each_top
        if rest and rest_mass > 0.0:
            each_rest = rest_mass / len(rest)
            for cid in rest:
                shares[cid] += each_rest
        elif rest_mass > 0.0 and top:
            each_top = rest_mass / len(top)
            for cid in top:
                shares[cid] += each_top
        _ = assigned_test  # documented constraint via exploit_frac only

    def _aggregate(
        self, day: int, history: list[dict]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Return (raw_spend, weighted_spend, weighted_revenue)."""
        known = set(self.creative_ids)
        spend_raw = {cid: 0.0 for cid in self.creative_ids}
        spend = {cid: 0.0 for cid in self.creative_ids}
        revenue = {cid: 0.0 for cid in self.creative_ids}
        for row in history:
            parsed = self._parse_history_row(row, decision_day=day, known=known)
            if parsed is None:
                continue
            cid, row_spend, row_revenue, row_day = parsed
            weight = self._row_weight(day, row_day)
            spend_raw[cid] += row_spend
            spend[cid] += row_spend * weight
            revenue[cid] += row_revenue * weight
        return spend_raw, spend, revenue

    def _row_weight(self, decision_day: int, row_day: int) -> float:
        age = decision_day - row_day
        if self.recent_day_horizon > 0 and 1 <= age <= self.recent_day_horizon:
            return self.recent_day_weight
        return 1.0

    def _parse_history_row(
        self,
        row: Any,
        *,
        decision_day: int,
        known: set[str],
    ) -> tuple[str, float, float, int] | None:
        """Return (creative_id, spend, revenue, row_day) or None if unusable."""
        if not isinstance(row, dict):
            return None
        try:
            day_val = self._finite_number(row.get("day"))
            if day_val is None:
                return None
            row_day = int(day_val)
        except (TypeError, ValueError, OverflowError):
            return None
        if row_day >= decision_day:
            return None

        cid_raw = row.get("creative_id")
        if cid_raw is None:
            return None
        cid = str(cid_raw)
        if cid not in known:
            return None

        row_spend = self._finite_number(row.get("spend", 0.0))
        row_revenue = self._finite_number(row.get("revenue", 0.0))
        if row_spend is None or row_revenue is None:
            return None
        if row_spend < 0.0 or row_revenue < 0.0:
            return None
        return cid, row_spend, row_revenue, row_day

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        # bool is a subclass of int; reject it as malformed input.
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def _shrunk_rate(self, spend: float, revenue: float, account_avg: float) -> float:
        """Empirical revenue/spend pulled toward account_avg when spend is small."""
        if spend <= 0.0:
            return account_avg
        empirical = revenue / spend
        if not math.isfinite(empirical):
            return account_avg
        tau = self.shrink_spend
        if tau <= 0.0:
            return empirical
        shrunk = (spend * empirical + tau * account_avg) / (spend + tau)
        if not math.isfinite(shrunk):
            return account_avg
        return shrunk

    @staticmethod
    def _add_weighted_shares(
        shares: dict[str, float],
        creatives: list[str],
        rates: dict[str, float],
        mass: float,
    ) -> None:
        weights = [max(rates[cid], 0.0) for cid in creatives]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            each = mass / len(creatives)
            for cid in creatives:
                shares[cid] += each
            return
        for cid, weight in zip(creatives, weights):
            shares[cid] += mass * (weight / weight_sum)

    def _apply_share_cap(
        self,
        shares: dict[str, float],
        *,
        reserved: dict[str, float],
        cap: float,
    ) -> dict[str, float]:
        """Clip shares above cap and redistribute excess to creatives with room.

        Creatives keep at least their exploration reserve when that reserve is
        itself at or under the cap; excess is never taken from that floor.
        """
        out = {cid: max(float(v), 0.0) for cid, v in shares.items()}
        floors = {
            cid: min(max(float(reserved.get(cid, 0.0)), 0.0), cap) for cid in out
        }

        for _ in range(len(out) + 5):
            excess = 0.0
            for cid, value in list(out.items()):
                if value > cap:
                    excess += value - cap
                    out[cid] = cap
            if excess <= 1e-15:
                break

            # Prefer filling creatives still below their exploration floor.
            below_floor = [
                cid for cid, value in out.items() if value + 1e-15 < floors[cid]
            ]
            if below_floor:
                recipients = below_floor
                room = {cid: floors[cid] - out[cid] for cid in recipients}
            else:
                recipients = [cid for cid, value in out.items() if value + 1e-15 < cap]
                room = {cid: cap - out[cid] for cid in recipients}

            room_sum = sum(room.values())
            if not recipients or room_sum <= 1e-15:
                # No eligible recipient; dump residual on anyone under floor==0 path.
                # With cap >= 1/n this should not happen for a normalized vector.
                break

            for cid in recipients:
                out[cid] += excess * (room[cid] / room_sum)

        # Restore floors if redistribution somehow dipped below them.
        shortfall = 0.0
        for cid, floor in floors.items():
            if out[cid] + 1e-15 < floor:
                shortfall += floor - out[cid]
                out[cid] = floor
        if shortfall > 1e-15:
            donors = [
                cid
                for cid, value in out.items()
                if value > floors[cid] + 1e-15 and value > 0.0
            ]
            donor_mass = sum(out[cid] - floors[cid] for cid in donors)
            if donor_mass > 1e-15:
                take = min(shortfall, donor_mass)
                for cid in donors:
                    available = out[cid] - floors[cid]
                    out[cid] -= take * (available / donor_mass)

        return out

    @staticmethod
    def _normalize(shares: dict[str, float]) -> dict[str, float]:
        cleaned = {cid: max(float(v), 0.0) for cid, v in shares.items()}
        total = sum(cleaned.values())
        if total <= 0.0:
            n = len(cleaned)
            return {cid: 1.0 / n for cid in cleaned}
        return {cid: value / total for cid, value in cleaned.items()}
