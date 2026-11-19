#!/usr/bin/env python3
"""Run Policy, B1, and B2 once on the shared simulator (fixed seed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import B1Uniform, B2WindowedGreedy
from policy import Policy
from sim.simulator import Simulator, load_config, load_creatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "sim" / "config.json",
        help="Simulator ground-truth config (not read by Policy)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    creatives = load_creatives()
    daily_budget = float(config["daily_budget"])
    sim = Simulator(config, creatives, seed=args.seed)

    runs = [
        (
            "Policy",
            lambda: Policy(creatives, daily_budget=daily_budget, seed=args.seed),
        ),
        (
            "B1_uniform",
            lambda: B1Uniform(creatives, daily_budget=daily_budget, seed=args.seed),
        ),
        (
            "B2_windowed_greedy",
            lambda: B2WindowedGreedy(creatives, daily_budget=daily_budget, seed=args.seed),
        ),
    ]

    print(f"seed={args.seed}")
    print(f"config={args.config}")
    print(f"days={config['start_day']}..{config['end_day']} daily_budget={daily_budget}")
    print()
    print(
        f"{'policy':<20} {'decisions':>10} {'allocated':>12} {'spend':>12} {'revenue':>12}"
    )

    expected_decisions = int(config["end_day"]) - int(config["start_day"]) + 1
    expected_allocated = expected_decisions * daily_budget

    for name, factory in runs:
        result = sim.run(name, factory)
        print(
            f"{result.name:<20} {result.n_decisions:10d} "
            f"{result.total_allocated:12.2f} {result.total_spend:12.2f} "
            f"{result.total_revenue:12.2f}"
        )
        if result.n_decisions != expected_decisions:
            raise SystemExit(
                f"{name}: expected {expected_decisions} decisions, got {result.n_decisions}"
            )
        if abs(result.total_allocated - expected_allocated) > 0.01:
            raise SystemExit(
                f"{name}: expected allocated ${expected_allocated:.2f}, "
                f"got ${result.total_allocated:.2f}"
            )

    print()
    print(f"ok: each policy made {expected_decisions} decisions")
    print(f"ok: each policy allocated ${expected_allocated:.2f}")


if __name__ == "__main__":
    main()
