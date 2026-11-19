"""Clairvoyant allocator that may read simulator ground-truth config.

What it can see
---------------
- Per-creative latent expected ROAS from ``sim/config.json`` (``creatives.*.roas``).
- Simulator delivery rules from the same config: ``min_deliver_dollars`` and
  ``delivery_rate``. Expected delivered dollars for an allocation ``b`` are
  ``0`` if ``b < min_deliver_dollars``, else ``b * delivery_rate`` (noise is
  mean-one, so it drops out of the expectation used for ranking).
- Optional saturation curve for stress scenarios:
  ``revenue = roas * spend * (k / (k + spend))`` with the scenario's ``k``.

What it cannot see / does not use
---------------------------------
- The reporting history passed to ``allocate`` (ignored).
- Realized noise draws for the day.
- Anything inside ``policy.py``, ``baselines.py``, or the CSV history files
  beyond the creative ID list supplied at construction.

Policy, B1, and B2 must not import this module or the ground-truth ROAS map.
"""

from __future__ import annotations

from typing import Any


def expected_revenue(
    allocation: float,
    *,
    roas: float,
    delivery_rate: float,
    min_deliver: float,
    saturation_k: float | None,
) -> float:
    if allocation < min_deliver:
        return 0.0
    spend = allocation * delivery_rate
    if spend <= 0.0:
        return 0.0
    if saturation_k is None:
        return roas * spend
    return roas * spend * (saturation_k / (saturation_k + spend))


class Clairvoyant:
    """Allocate using ground-truth expected returns and delivery rules."""

    def __init__(
        self,
        creatives: list[dict],
        daily_budget: float,
        seed: int,
        *,
        config: dict[str, Any],
        saturation_k: float | None = None,
    ):
        if not creatives:
            raise ValueError("creatives must be non-empty")
        self.creatives = list(creatives)
        self.creative_ids = [str(c["creative_id"]) for c in self.creatives]
        self.daily_budget = float(daily_budget)
        self.seed = int(seed)
        self.min_deliver = float(config["min_deliver_dollars"])
        self.delivery_rate = float(config["delivery_rate"])
        if saturation_k is None:
            saturation_k = config.get("saturation_k")
        self.saturation_k = (
            float(saturation_k) if saturation_k is not None else None
        )
        if self.saturation_k is not None and self.saturation_k <= 0.0:
            self.saturation_k = None

        params = config["creatives"]
        self.base_roas = {
            cid: float(params[cid]["roas"]) for cid in self.creative_ids
        }
        self._shares = self._plan_shares()

    def allocate(self, day: int, history: list[dict]) -> dict[str, float]:
        _ = day, history
        return dict(self._shares)

    def _plan_shares(self) -> dict[str, float]:
        ranked = sorted(
            self.creative_ids,
            key=lambda cid: (-self.base_roas[cid], cid),
        )
        if self.saturation_k is None:
            shares = {cid: 0.0 for cid in self.creative_ids}
            shares[ranked[0]] = 1.0
            return shares

        # Restricted benchmark: only equal splits over the top-m creatives by
        # base ROAS are considered (m = 1..10). This is not an unrestricted
        # water-filling optimum under the saturation curve.
        best_m = 1
        best_value = -1.0
        best_shares = {cid: 0.0 for cid in self.creative_ids}
        best_shares[ranked[0]] = 1.0
        max_m = min(10, len(ranked))
        for m in range(1, max_m + 1):
            chosen = ranked[:m]
            each = 1.0 / m
            value = 0.0
            trial = {cid: 0.0 for cid in self.creative_ids}
            for cid in chosen:
                trial[cid] = each
                value += expected_revenue(
                    each * self.daily_budget,
                    roas=self.base_roas[cid],
                    delivery_rate=self.delivery_rate,
                    min_deliver=self.min_deliver,
                    saturation_k=self.saturation_k,
                )
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12 and m < best_m
            ):
                best_value = value
                best_m = m
                best_shares = trial
        return best_shares
