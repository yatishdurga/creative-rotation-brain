"""The checked-in example is synthetic and matches the fitted simulator config."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from sim.simulator import load_config, load_creatives, load_reported_snapshot

ROOT = Path(__file__).resolve().parents[1]


class ExampleDataTests(unittest.TestCase):
    def test_catalog_config_and_history_use_the_same_ids(self) -> None:
        creatives = load_creatives()
        config = load_config()
        reported = load_reported_snapshot()
        ids = [str(row["creative_id"]) for row in creatives]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(config["creatives"]))
        self.assertGreaterEqual(len(ids), 2)
        reported_ids = {row["creative_id"] for row in reported}
        self.assertTrue(reported_ids <= set(ids))
        self.assertGreater(len(reported), 0)
        self.assertGreater(config["delivery_rate"], 0.0)
        self.assertLessEqual(config["delivery_rate"], 1.0)
        for age, fraction in config["reported_fraction_by_age"].items():
            self.assertGreaterEqual(float(fraction), 0.0)
            self.assertLessEqual(float(fraction), 1.0)
            self.assertGreaterEqual(int(age), 1)

    def test_daily_files_share_keys_and_do_not_include_decision_days(self) -> None:
        def load(name: str) -> list[dict]:
            with (ROOT / "data" / name).open(newline="") as handle:
                return list(csv.DictReader(handle))

        final_rows = load("daily_final.csv")
        reported_rows = load("daily_reported.csv")
        self.assertEqual(len(final_rows), len(reported_rows))
        final_keys = {(row["day"], row["creative_id"]) for row in final_rows}
        reported_keys = {(row["day"], row["creative_id"]) for row in reported_rows}
        self.assertEqual(final_keys, reported_keys)
        self.assertTrue(all(int(row["day"]) < 61 for row in final_rows))


if __name__ == "__main__":
    unittest.main()
