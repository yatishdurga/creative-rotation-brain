#!/usr/bin/env python3
"""Generate a synthetic example account and rebuild sim/config.json from it.

The creative list and day-1..60 history are invented here. Per-creative
rates, the delivery ratio, and the reporting-lag curve written into
sim/config.json are estimated from those files. Structural knobs that
cannot be read off a single history (noise scales, horizon, minimum
delivery) are recorded in data/example_settings.json and copied through
by the fit.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.rng import noise_multiplier, unit_draw

N_CREATIVES = 50
N_SERVED = 40
HISTORY_DAYS = 60
DECISION_MORNING = 61
SETTLED_THROUGH = 45
GENERATIVE_SEED = 20260318

FORMATS = ("image", "short_video", "carousel")
HOOKS = ("question", "price", "how_to")
CTAS = ("Learn more", "Get offer", "See details")

# Reporting lag applied when building the morning-of-61 snapshot.
# The fitter re-estimates these from the two CSVs; it does not read this map.
LAG_FRACTIONS = {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.65, 5: 0.78, 6: 0.90, 7: 1.0}

SETTINGS = {
    "daily_budget": 2000.0,
    "start_day": 61,
    "end_day": 90,
    "history_days": HISTORY_DAYS,
    "settled_through_day": SETTLED_THROUGH,
    "attribution_lag_days": 7,
    "min_deliver_dollars": 1.0,
    "generative_delivery_rate": 0.91,
    "delivery_noise_cv": 0.04,
    "impression_noise_cv": 0.08,
    "roas_noise_cv": 0.15,
    "generative_seed": GENERATIVE_SEED,
    "n_creatives": N_CREATIVES,
    "n_served": N_SERVED,
}


def creative_id(index: int) -> str:
    return f"AD{index:02d}"


def build_creatives(rng: random.Random) -> tuple[list[dict], dict[str, dict]]:
    """Return catalog rows and the latent rates used only to simulate history."""
    rows: list[dict] = []
    latent: dict[str, dict] = {}
    fixed_roas = {
        1: 3.4,
        2: 2.7,
        3: 2.2,
        4: 0.40,
        5: 0.55,
        6: 0.70,
    }
    for index in range(1, N_CREATIVES + 1):
        cid = creative_id(index)
        served = index <= N_SERVED
        launch = "" if not served else str(1 + ((index - 1) % 6) * 7)
        rows.append(
            {
                "creative_id": cid,
                "format": FORMATS[(index - 1) % len(FORMATS)],
                "hook_type": HOOKS[(index - 1) % len(HOOKS)],
                "headline": f"Example headline {index:02d}",
                "primary_text": f"Synthetic example creative {index:02d}.",
                "cta": CTAS[(index - 1) % len(CTAS)],
                "launch_day": launch,
            }
        )
        if not served:
            continue
        latent[cid] = {
            "roas": fixed_roas.get(index, round(rng.uniform(0.85, 1.90), 4)),
            "aov": round(rng.uniform(28.0, 90.0), 2),
            "ctr": round(rng.uniform(0.008, 0.025), 4),
            "impressions_per_dollar": round(rng.uniform(55.0, 100.0), 2),
        }
    return rows, latent


def deliver(
    day: int,
    creative_id_value: str,
    budget: float,
    params: dict,
) -> dict[str, float]:
    minimum = float(SETTINGS["min_deliver_dollars"])
    if budget < minimum:
        return {
            "spend": 0.0,
            "impressions": 0.0,
            "clicks": 0.0,
            "purchases": 0.0,
            "revenue": 0.0,
        }
    delivery_u = unit_draw(GENERATIVE_SEED, day, creative_id_value, "delivery")
    roas_u = unit_draw(GENERATIVE_SEED, day, creative_id_value, "roas")
    impr_u = unit_draw(GENERATIVE_SEED, day, creative_id_value, "impressions")
    spend = budget * float(SETTINGS["generative_delivery_rate"])
    spend *= noise_multiplier(delivery_u, float(SETTINGS["delivery_noise_cv"]))
    spend = min(budget, max(0.0, spend))
    roas = max(
        0.0,
        float(params["roas"])
        * noise_multiplier(roas_u, float(SETTINGS["roas_noise_cv"])),
    )
    revenue = spend * roas
    impressions = max(
        0.0,
        spend
        * float(params["impressions_per_dollar"])
        * noise_multiplier(impr_u, float(SETTINGS["impression_noise_cv"])),
    )
    clicks = max(0.0, impressions * float(params["ctr"]))
    aov = float(params["aov"])
    purchases = (revenue / aov) if aov > 0.0 else 0.0
    return {
        "spend": spend,
        "impressions": impressions,
        "clicks": clicks,
        "purchases": purchases,
        "revenue": revenue,
    }


def simulate_history(catalog: list[dict], latent: dict[str, dict]) -> list[dict]:
    """Equal split across creatives already launched. This is example history, not the policy."""
    daily_budget = float(SETTINGS["daily_budget"])
    rows: list[dict] = []
    for day in range(1, HISTORY_DAYS + 1):
        live = [
            row
            for row in catalog
            if row["launch_day"] != "" and int(row["launch_day"]) <= day
        ]
        if not live:
            continue
        each = daily_budget / len(live)
        for row in live:
            cid = row["creative_id"]
            outcome = deliver(day, cid, each, latent[cid])
            rows.append(
                {
                    "day": day,
                    "creative_id": cid,
                    "budget": each,
                    "spend": outcome["spend"],
                    "impressions": outcome["impressions"],
                    "clicks": outcome["clicks"],
                    "purchases": outcome["purchases"],
                    "revenue": outcome["revenue"],
                }
            )
    return rows


def reported_snapshot(truth_rows: list[dict]) -> list[dict]:
    reported: list[dict] = []
    lag_days = int(SETTINGS["attribution_lag_days"])
    for row in truth_rows:
        age = DECISION_MORNING - int(row["day"])
        frac = 1.0 if age >= lag_days else LAG_FRACTIONS[age]
        reported.append(
            {
                **row,
                "revenue": float(row["revenue"]) * frac,
                "purchases": float(row["purchases"]) * frac,
            }
        )
    return reported


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take a median of an empty sample")
    return float(median(values))


def fit_config(catalog: list[dict], truth_rows: list[dict], reported_rows: list[dict]) -> dict:
    settled = [row for row in truth_rows if int(row["day"]) <= SETTLED_THROUGH]
    by_id: dict[str, dict[str, float]] = {}
    for row in settled:
        bucket = by_id.setdefault(
            row["creative_id"],
            {"spend": 0.0, "revenue": 0.0, "impressions": 0.0, "clicks": 0.0, "purchases": 0.0},
        )
        bucket["spend"] += float(row["spend"])
        bucket["revenue"] += float(row["revenue"])
        bucket["impressions"] += float(row["impressions"])
        bucket["clicks"] += float(row["clicks"])
        bucket["purchases"] += float(row["purchases"])

    fitted: dict[str, dict] = {}
    for cid, totals in by_id.items():
        if totals["spend"] <= 0.0:
            continue
        fitted[cid] = {
            "roas": totals["revenue"] / totals["spend"],
            "ctr": (totals["clicks"] / totals["impressions"]) if totals["impressions"] > 0.0 else 0.0,
            "impressions_per_dollar": totals["impressions"] / totals["spend"],
            "aov": (totals["revenue"] / totals["purchases"]) if totals["purchases"] > 0.0 else None,
            "source": f"settled_days_1_{SETTLED_THROUGH}",
        }

    aov_samples = [float(row["aov"]) for row in fitted.values() if row["aov"] is not None]
    prior = {
        "roas": _median([float(row["roas"]) for row in fitted.values()]),
        "ctr": _median([float(row["ctr"]) for row in fitted.values()]),
        "impressions_per_dollar": _median(
            [float(row["impressions_per_dollar"]) for row in fitted.values()]
        ),
        "aov": _median(aov_samples),
    }
    for row in fitted.values():
        if row["aov"] is None:
            row["aov"] = prior["aov"]

    creatives_cfg: dict[str, dict] = {}
    for row in catalog:
        cid = row["creative_id"]
        if cid in fitted:
            creatives_cfg[cid] = fitted[cid]
            continue
        creatives_cfg[cid] = {
            "roas": prior["roas"],
            "ctr": prior["ctr"],
            "impressions_per_dollar": prior["impressions_per_dollar"],
            "aov": prior["aov"],
            "source": "population_median_prior",
        }

    budget = sum(float(row["budget"]) for row in truth_rows)
    spend = sum(float(row["spend"]) for row in truth_rows)
    if budget <= 0.0:
        raise ValueError("example history has no allocated budget")
    delivery_rate = spend / budget

    reported_by_key = {(int(row["day"]), row["creative_id"]): row for row in reported_rows}
    lag_days = int(SETTINGS["attribution_lag_days"])
    recovered_lag: dict[str, float] = {}
    for age in range(1, lag_days + 1):
        day = DECISION_MORNING - age
        num = 0.0
        den = 0.0
        for row in truth_rows:
            if int(row["day"]) != day:
                continue
            den += float(row["revenue"])
            num += float(reported_by_key[(day, row["creative_id"])]["revenue"])
        if den <= 0.0:
            raise ValueError(f"no truth revenue on day {day} to estimate lag age {age}")
        recovered_lag[str(age)] = round(num / den, 10)

    return {
        "daily_budget": float(SETTINGS["daily_budget"]),
        "start_day": int(SETTINGS["start_day"]),
        "end_day": int(SETTINGS["end_day"]),
        "min_deliver_dollars": float(SETTINGS["min_deliver_dollars"]),
        "delivery_rate": delivery_rate,
        "delivery_noise_cv": float(SETTINGS["delivery_noise_cv"]),
        "impression_noise_cv": float(SETTINGS["impression_noise_cv"]),
        "roas_noise_cv": float(SETTINGS["roas_noise_cv"]),
        "attribution_lag_days": lag_days,
        "reported_fraction_by_age": recovered_lag,
        "population_prior": prior,
        "creatives": creatives_cfg,
        "notes": [
            "Synthetic example. Not measured on a live account.",
            "Per-creative roas, ctr, impressions_per_dollar, and aov are totals from data/daily_final.csv days 1-45.",
            "Creatives with no settled spend use the median of the served creatives.",
            "delivery_rate is total spend / total budget on data/daily_final.csv.",
            "reported_fraction_by_age is reported revenue / final revenue by age on the morning of day 61.",
            "Noise scales are the generator settings in data/example_settings.json; a single history does not identify them.",
        ],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    rng = random.Random(GENERATIVE_SEED)
    catalog, latent = build_creatives(rng)
    truth_rows = simulate_history(catalog, latent)
    reported_rows = reported_snapshot(truth_rows)
    config = fit_config(catalog, truth_rows, reported_rows)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        data_dir / "creatives.csv",
        catalog,
        ["creative_id", "format", "hook_type", "headline", "primary_text", "cta", "launch_day"],
    )
    history_fields = [
        "day",
        "creative_id",
        "budget",
        "spend",
        "impressions",
        "clicks",
        "purchases",
        "revenue",
    ]
    write_csv(data_dir / "daily_final.csv", truth_rows, history_fields)
    write_csv(data_dir / "daily_reported.csv", reported_rows, history_fields)
    (data_dir / "example_settings.json").write_text(json.dumps(SETTINGS, indent=2) + "\n")
    (ROOT / "sim" / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    served = sum(1 for row in config["creatives"].values() if row["source"].startswith("settled"))
    prior = sum(1 for row in config["creatives"].values() if row["source"] == "population_median_prior")
    print(f"creatives={len(catalog)} history_rows={len(truth_rows)}")
    print(f"config served={served} median_prior={prior}")
    print(f"fitted delivery_rate={config['delivery_rate']:.6f}")
    print(f"recovered lag={config['reported_fraction_by_age']}")
    print(f"wrote {data_dir / 'creatives.csv'}")
    print(f"wrote {data_dir / 'daily_final.csv'}")
    print(f"wrote {data_dir / 'daily_reported.csv'}")
    print(f"wrote {ROOT / 'sim' / 'config.json'}")


if __name__ == "__main__":
    main()
