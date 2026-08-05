#!/usr/bin/env python
"""
Diagnostic backtest for NHSNNowcaster: hits real S3 NHSN vintages (anonymous) to check whether
nowcast correction actually reduces error against later-revised "truth" data.

This is a maintainer-run diagnostic, NOT a CI gate -- see the project plan's "Empirical
Validation Results" section for why. Real-data backtesting (this script, run across 25
historical date/location combinations spanning the 2024/25 flu season) found that nowcast
correction does NOT reliably reduce error: it INCREASED total aggregate error under every
max_delay tried. Two root causes are documented in the plan and in `iddata.constants`'
`NHSN_MAX_DELAY_WEEKS` comment: (1) a single pooled/national delay-PMF doesn't transfer to
individual locations, whose completion curves vary far more than a rescaled shared curve; (2)
the completion profile is not stable over time, and its time-variation differs between the
national aggregate and individual states.

Nowcasting remains implemented and opt-in (off by default everywhere) but is NOT recommended for
production use pending a v2. Re-run this script after any v2 change (e.g. per-location fitting)
to check whether it actually improves on these numbers before considering production use.

Usage
-----
    uv run python scripts/backtest_nhsn_nowcast.py
    uv run python scripts/backtest_nhsn_nowcast.py --max-delay-weeks 2
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np
import pandas as pd

from iddata.enums import Disease
from iddata.loader import DiseaseDataLoader
from iddata.nowcast.base import NowcastConfig
from iddata.sources.nhsn import NHSNDataSource

# Historical as_of dates during the confirmed 2024/25 flu season's rise/peak/decline, each
# independently confirmed to have dense, real (non-shutdown-affected) vintage coverage.
_BACKTEST_AS_OF_DATES = [
    datetime.date(2025, 1, 18),
    datetime.date(2025, 2, 8),
    datetime.date(2025, 2, 15),
    datetime.date(2025, 3, 15),
    datetime.date(2025, 4, 12),
]

# National aggregate plus a handful of representative states: two large (more stable, closer to
# national in character) and two small/mid-size (noisier raw counts, where correction behavior
# could plausibly differ most from the smoothed national series).
_LOCATIONS = [
    ("US", "national"),  # national aggregate
    ("06", "state"),      # California -- large
    ("36", "state"),      # New York -- large
    ("34", "state"),      # New Jersey -- mid-size
    ("50", "state"),      # Vermont -- small
]

_TRUTH_LAG_WEEKS = 12
_TRAILING_WEEKS_TO_CHECK = 3  # compare the weeks nearest each as_of, where backfill matters most


def _location_series(df, location, agg_level, as_of_upper_bound):
    sub = df[(df["location"] == location) & (df["agg_level"] == agg_level)].sort_values("wk_end_date")
    sub = sub[sub["wk_end_date"] <= pd.Timestamp(as_of_upper_bound)]
    return sub.tail(_TRAILING_WEEKS_TO_CHECK).set_index("wk_end_date")["inc"]


def _backtest_one_date(disease, as_of, nowcast_config):
    """Returns a list of (location, agg_level, raw_error, corrected_error) tuples, one per _LOCATIONS entry."""
    truth_as_of = as_of + datetime.timedelta(weeks=_TRUTH_LAG_WEEKS)

    raw_df = NHSNDataSource(disease=disease).load(as_of=as_of)
    truth_df = NHSNDataSource(disease=disease).load(as_of=truth_as_of)
    loader = DiseaseDataLoader()
    corrected_df = loader.load(
        sources=[NHSNDataSource(disease=disease)],
        as_of=as_of,
        nowcast=nowcast_config,
    )

    results = []
    for location, agg_level in _LOCATIONS:
        raw_trailing = _location_series(raw_df, location, agg_level, as_of)
        corrected_trailing = _location_series(corrected_df, location, agg_level, as_of)
        truth_trailing = _location_series(truth_df, location, agg_level, as_of)

        common_weeks = raw_trailing.index.intersection(corrected_trailing.index).intersection(truth_trailing.index)
        if len(common_weeks) == 0:
            continue

        raw_error = np.abs(raw_trailing[common_weeks] - truth_trailing[common_weeks]).sum()
        corrected_error = np.abs(corrected_trailing[common_weeks] - truth_trailing[common_weeks]).sum()
        results.append((location, agg_level, raw_error, corrected_error))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-delay-weeks", type=int, default=None,
                         help="Override max_delay_weeks (default: look up the calibrated NHSN_MAX_DELAY_WEEKS constant).")
    parser.add_argument("--disease", type=str, default="flu", choices=["flu", "covid"])
    args = parser.parse_args()

    disease = Disease.FLU if args.disease == "flu" else Disease.COVID
    nowcast_config = NowcastConfig(min_vintages=8, max_delay_weeks=args.max_delay_weeks)

    results = []
    for as_of in _BACKTEST_AS_OF_DATES:
        print(f"backtesting as_of={as_of}...")
        for location, agg_level, raw_error, corrected_error in _backtest_one_date(disease, as_of, nowcast_config):
            results.append((as_of, location, agg_level, raw_error, corrected_error))

    if not results:
        print("no backtest date/location combinations produced usable overlapping trailing weeks")
        return

    print()
    print(f'{"as_of":12s} {"location":10s} {"raw_error":>10s} {"corrected_error":>16s}')
    for as_of, location, agg_level, raw, corrected in results:
        print(f"{str(as_of):12s} {location}/{agg_level:8s} {raw:10.3f} {corrected:16.3f}")

    total_raw_error = sum(r for _, _, _, r, _ in results)
    total_corrected_error = sum(c for _, _, _, _, c in results)
    verdict = "REDUCED" if total_corrected_error < total_raw_error else "INCREASED (worse)"
    print(f"\ntotal_raw_error={total_raw_error:.2f}, total_corrected_error={total_corrected_error:.2f}")
    print(f"nowcast correction {verdict} aggregate error across {len(results)} date/location combinations.")


if __name__ == "__main__":
    main()
