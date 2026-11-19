# Creative Rotation Brain

A daily budget policy for a fixed set of ad creatives. Each morning it reads the reporting history available so far and returns a share of that day's budget for every creative. The repository also includes two baselines, a config-driven delivery simulator, and a clairvoyant that is allowed to read the simulator's ground truth.

The checked-in account is synthetic. `examples/generate_example.py` invents 50 creatives and 60 days of history, then rebuilds `sim/config.json` from those files. Nothing here is a live-account export.

## Setup

Python 3.11 or newer. The policy, baselines, simulator, and tests use the standard library only.

```bash
python3 -m unittest discover -s tests -v
python3 examples/generate_example.py
python3 sim/run_once.py
python3 sim/evaluate.py --out examples/sample_evaluation.txt
```

`generate_example.py` rewrites `data/` and `sim/config.json`. Run it again to reproduce the example from seed `20260318`.

## Architecture

`policy.py` sees creative metadata and history rows (`day`, `creative_id`, `budget`, `spend`, `impressions`, `clicks`, `purchases`, `revenue`). It ranks creatives by revenue per delivered dollar, keeps a 10% reserve for creatives under $50 of spend, and caps any one creative at 20% of the day (or `1/n` when that is larger). It does not import the simulator and does not read `sim/config.json`.

`baselines.py` provides two comparisons. B1 gives every creative an equal share. B2 ranks by trailing 7-day revenue per dollar, assigns the population mean rate when trailing spend is zero or trailing impressions are under 2,000, gives the top 5 creatives 75% of the budget, and splits the rest evenly.

`sim/simulator.py` loads the morning-of-day-61 snapshot in `data/daily_reported.csv`, then simulates days 61–90. Allocations under $1 do not deliver. Spend, impressions, and revenue use the fitted rates in `sim/config.json`, with noise draws keyed by seed, day, and creative so every policy faces the same world. Recent simulated revenue is scaled by the fitted reporting lag before the next decision.

`sim/clairvoyant.py` may read that config. Under linear returns it assigns the whole budget to the creative with the highest fitted ROAS. It ignores history. Policy and the baselines cannot see it.

`examples/generate_example.py` is the only place the example latents exist. The committed config does not copy them. It estimates, from `data/daily_final.csv` and `data/daily_reported.csv`:

- revenue/spend, clicks/impressions, impressions per dollar, and average order value on days 1–45
- the median of those rates for creatives with no settled spend
- delivery rate as total spend divided by total budget
- reporting fractions as reported revenue divided by final revenue at each age from 1 to 7

Noise scales stay in `data/example_settings.json` because one history does not identify them. The fitter copies those scales into the config.

## Example results

Reproduced with `python3 sim/evaluate.py` on this example: 20 paired seeds, linear returns, 5,000 bootstrap resamples, bootstrap seed `20260318`. Full text is in `examples/sample_evaluation.txt`.

| Contrast | Mean revenue difference | 95% interval |
|---|---:|---:|
| Policy − B1 | +9618.83 | [9544.21, 9690.89] |
| Policy − B2 | −27465.06 | [−27883.67, −27028.25] |
| Clairvoyant − B1 | +106439.07 | [104687.37, 108099.01] |

Mean 30-day revenue: Policy 88847.60, B1 79228.77, B2 116312.66, clairvoyant 185667.85. Headroom `(Policy − B1) / (clairvoyant − B1)` is 0.0904.

`python3 sim/run_once.py` (seed 7 only) scored Policy 88992.96, B1 79134.00, and B2 118104.42. That is one shared world, not the 20-seed mean.

## Simulation limitations

These numbers describe the synthetic generator, not a live account. History before day 61 is an equal split across creatives that had already launched, so it is not a record of this policy or of a human buyer. Days 61–90 are then drawn from the fitted config. Fitted ROAS is close to the generator's latent ROAS, but it is not the same number, and the 10 unserved creatives receive only the population median because the history has no spend for them.

The linear clairvoyant puts 100% on the single highest fitted ROAS. That is an upper bound inside this delivery model, and it ignores the 20% cap the policy uses. Delivery noise, impression noise, and ROAS noise are assumptions from `example_settings.json`. The lag curve is a deterministic function of age. The simulator has no auction, no fatigue unless you pass `--saturation-k`, and no competitor response. A positive gap versus B1 on this example is evidence that the policy can use the history it is given. It is not an estimate of incremental revenue on real spend.
