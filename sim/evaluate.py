#!/usr/bin/env python3
"""Paired multi-seed evaluation of Policy, B1, B2, and a clairvoyant.

The clairvoyant may read sim/config.json. Policy, B1, and B2 do not.

Examples
--------
    python3 sim/evaluate.py --out examples/sample_evaluation.txt
    python3 sim/evaluate.py --shrink-spend 200
    python3 sim/evaluate.py --saturation-k 400
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import B1Uniform, B2WindowedGreedy
from policy import Policy
from sim.clairvoyant import Clairvoyant
from sim.simulator import Simulator, load_config, load_creatives

# Fixed paired seeds: every policy is evaluated on this exact list.
PAIRED_SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260318


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_diffs(primary: list[float], baseline: list[float]) -> list[float]:
    if len(primary) != len(baseline):
        raise ValueError("paired series must have equal length")
    return [p - b for p, b in zip(primary, baseline)]


def bootstrap_mean_ci(
    diffs: list[float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample whole seed-level paired differences; return mean and percentile CI."""
    point = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return point, means[lo_idx], means[hi_idx]


def headroom_ratio(policy_mean: float, b1_mean: float, clair_mean: float) -> str:
    denom = clair_mean - b1_mean
    numer = policy_mean - b1_mean
    if denom <= 0.0:
        return (
            f"undefined (clairvoyant-B1={denom:.2f} <= 0; "
            f"policy-B1={numer:.2f})"
        )
    return f"{numer / denom:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "sim" / "config.json",
        help="Simulator ground-truth config (clairvoyant-only; not used by Policy/B1/B2)",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--shrink-spend",
        type=float,
        default=None,
        help="Override Policy shrink_spend (default: Policy.DEFAULT_SHRINK_SPEND). "
        "Use 200 to pull low-spend rates toward the account average.",
    )
    parser.add_argument(
        "--saturation-k",
        type=float,
        default=None,
        help="If set, run the saturated-return scenario with this k "
        "(e.g. 250). Default is linear returns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the full printed transcript.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    creatives = load_creatives()
    daily_budget = float(config["daily_budget"])
    shrink = (
        Policy.DEFAULT_SHRINK_SPEND if args.shrink_spend is None else float(args.shrink_spend)
    )

    revenues: dict[str, list[float]] = {
        "Policy": [],
        "B1_uniform": [],
        "B2_windowed_greedy": [],
        "Clairvoyant": [],
    }
    top5: dict[str, list[float]] = {k: [] for k in revenues}

    lines: list[str] = []

    class Tee:
        def write(self, s: str) -> int:
            lines.append(s)
            sys.__stdout__.write(s)
            return len(s)

        def flush(self) -> None:
            sys.__stdout__.flush()

    sys.stdout = Tee()  # type: ignore[assignment]
    try:
        print("paired_seeds=", PAIRED_SEEDS)
        print(
            f"n_seeds={len(PAIRED_SEEDS)} bootstrap={args.bootstrap} "
            f"bootstrap_seed={BOOTSTRAP_SEED}"
        )
        try:
            config_display = args.config.resolve().relative_to(ROOT)
        except ValueError:
            config_display = args.config
        print(f"config={config_display}")
        print(f"Policy shrink_spend={shrink} (DEFAULT_SHRINK_SPEND={Policy.DEFAULT_SHRINK_SPEND})")
        print(f"Policy test_budget_fraction={Policy.DEFAULT_TEST_BUDGET_FRACTION}")
        if args.saturation_k is None:
            print("returns: linear")
        else:
            print(f"returns: saturated k={args.saturation_k}")
        print()

        for seed in PAIRED_SEEDS:
            sim = Simulator(
                config, creatives, seed=seed, saturation_k=args.saturation_k
            )
            factories = {
                "Policy": lambda s=seed: Policy(
                    creatives,
                    daily_budget=daily_budget,
                    seed=s,
                    shrink_spend=shrink,
                ),
                "B1_uniform": lambda s=seed: B1Uniform(
                    creatives, daily_budget=daily_budget, seed=s
                ),
                "B2_windowed_greedy": lambda s=seed: B2WindowedGreedy(
                    creatives, daily_budget=daily_budget, seed=s
                ),
                "Clairvoyant": lambda s=seed: Clairvoyant(
                    creatives,
                    daily_budget=daily_budget,
                    seed=s,
                    config=config,
                    saturation_k=args.saturation_k,
                ),
            }
            for name, factory in factories.items():
                result = sim.run(name, factory)
                revenues[name].append(result.total_revenue)
                top5[name].append(result.mean_daily_top5_share)

        print("Mean 30-day revenue and mean daily top-5 allocation share")
        print(f"{'policy':<20} {'mean_revenue':>14} {'mean_top5_share':>16}")
        for name in revenues:
            print(
                f"{name:<20} {mean(revenues[name]):14.2f} {mean(top5[name]):16.4f}"
            )

        print()
        print("Paired revenue differences (seed-level bootstrap 95% CI)")
        print(f"{'contrast':<28} {'mean_diff':>12} {'ci_low':>12} {'ci_high':>12}")

        contrasts = [
            ("Policy - B1", revenues["Policy"], revenues["B1_uniform"]),
            ("Policy - B2", revenues["Policy"], revenues["B2_windowed_greedy"]),
            ("B2 - B1", revenues["B2_windowed_greedy"], revenues["B1_uniform"]),
            ("Clairvoyant - B1", revenues["Clairvoyant"], revenues["B1_uniform"]),
            ("Clairvoyant - Policy", revenues["Clairvoyant"], revenues["Policy"]),
        ]
        for label, left, right in contrasts:
            diffs = paired_diffs(left, right)
            point, lo, hi = bootstrap_mean_ci(
                diffs, n_boot=args.bootstrap, seed=BOOTSTRAP_SEED
            )
            print(f"{label:<28} {point:12.2f} {lo:12.2f} {hi:12.2f}")

        print()
        ratio = headroom_ratio(
            mean(revenues["Policy"]),
            mean(revenues["B1_uniform"]),
            mean(revenues["Clairvoyant"]),
        )
        if args.saturation_k is None:
            print(
                "headroom_ratio_vs_clairvoyant "
                "(Policy-B1)/(clairvoyant-B1):",
                ratio,
            )
            print(
                "Clairvoyant (linear): 100% to the best expected-return ad "
                "by config creatives.*.roas after delivery rules; ignores "
                "history. See sim/clairvoyant.py."
            )
        else:
            print(
                "headroom_ratio_vs_restricted_clairvoyant "
                "(Policy-B1)/(restricted_clairvoyant-B1):",
                ratio,
            )
            print(
                "Restricted clairvoyant (saturation): equal-split top-m "
                "search over m=1..10 by base ROAS—not unrestricted "
                "water-filling. Visibility: creatives.*.roas, "
                "delivery_rate, min_deliver_dollars; ignores history. "
                "See sim/clairvoyant.py."
            )
    finally:
        sys.stdout = sys.__stdout__

    if args.out is not None:
        args.out.write_text("".join(lines))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
