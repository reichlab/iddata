#!/usr/bin/env python
"""
Diagnostic backtest for NHSNNowcaster: hits real S3 NHSN vintages (anonymous) to check whether
nowcast correction actually reduces error against later-revised "truth" data.

This is a maintainer-run diagnostic, NOT a CI gate -- see the project plan's "Empirical
Validation Results" section for why. Real-data backtesting (this script, run across 25
historical date/location combinations spanning the 2024/25 flu season) found that nowcast
correction does NOT reliably reduce error: it INCREASED total aggregate error under every
max_delay tried, including a subsequent per-location max_delay attempt (`NHSN_MAX_DELAY_WEEKS_BY_
LOCATION`), which made it MORE THAN DOUBLE WORSE (30 date/location combinations including PA:
total_raw_error=26.68, total_corrected_error=59.36) rather than better -- see `iddata.constants`'
`NHSN_MAX_DELAY_WEEKS_BY_LOCATION` comment and the plan's "v2 Attempt: Per-Location max_delay"
section for why (short answer: fitting a delay-PMF from one location's own limited training
window is noisier than the pooled fit, and that noise cost more accuracy than the per-location
signal gained back, even for locations whose own fitted max_delay matched independent
ground-truth evidence).

A follow-up v2 attempt (shrinking each location's own fitted delay-PMF toward a live pooled PMF,
weighted by data volume -- `NowcastConfig.pmf_shrinkage_k`, see its docstring) was ALSO found to
depend on a real, separate bug: the pooled PMF itself (`NHSNNowcaster._fit_pooled_pmf`) was being
computed by naively `np.vstack`-ing every location's triangle and calling `estimate_delay` on the
result, which silently corrupts every location's data after the first with computed placeholders
instead of their real observed values (see `iddata.nowcast.delay_model.estimate_delay_pooled`'s
docstring for the mechanism and a concrete before/after repro). Under that bug, shrinkage looked
like it roughly halved the per-location-noise damage (k=0 -> corrected=58.33; k=10000 ->
34.76). Once the pooled fit was fixed to fill each location's triangle independently before
combining (so real observed values are never discarded), shrinkage barely helps at all: sweeping
pmf_shrinkage_k on the same 30 combinations (raw=26.68 throughout) now gives k=0 -> 58.33
(unaffected, as expected -- k=0 never uses the pooled fit); k=2000 -> 54.49; k=10000 (the
default) -> 54.57; k=50000 -> 54.59. The near-total flatness across a 25x range of k is itself
informative: this parameter isn't the lever that matters. Shrinkage is a real (if now much more
modest, ~6%) improvement over pure per-location fitting, and still does NOT flip the sign --
correction stays worse than raw at every k, because the surviving error concentrates specifically
on dates near the 2024/25 season's sharp Feb-2025 peak (e.g. NJ and PA are close to raw on 3 of 5
backtest dates but blow up on 2025-02-08/02-15 -- MORE severely than under the old, buggy pooled
fit). That's the OTHER root cause (time-varying completion near a peak, not location noise),
which shrinkage does not address at all.

Nowcasting remains implemented and opt-in (off by default everywhere) but is NOT recommended for
production use pending a v2 that addresses time-varying completion near a peak. Re-run this
script after any future v2 change to check whether it actually improves on these numbers first.

Usage
-----
    uv run python scripts/backtest_nhsn_nowcast.py
    uv run python scripts/backtest_nhsn_nowcast.py --max-delay-weeks 2
    uv run python scripts/backtest_nhsn_nowcast.py --pmf-shrinkage-k 0
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
# national in character), two small/mid-size (noisier raw counts, where correction behavior
# could plausibly differ most from the smoothed national series), and PA -- included specifically
# to check the per-location NHSN_MAX_DELAY_WEEKS_BY_LOCATION calibration's suspected failure mode
# (see that constant's comment): PA independently shows a genuinely low completion ceiling that
# needs a LONGER max_delay, but the per-location chain-ladder fit recommends a SHORTER one
# (max_delay=2), which -- if that fit is spurious small-sample noise rather than real signal --
# should make PA's corrected error worse, not better.
_LOCATIONS = [
    ("US", "national"),  # national aggregate
    ("06", "state"),      # California -- large
    ("36", "state"),      # New York -- large
    ("34", "state"),      # New Jersey -- mid-size, known fast/complete convergence
    ("50", "state"),      # Vermont -- small
    ("42", "state"),      # Pennsylvania -- suspected per-location max_delay failure case
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
    parser.add_argument("--pmf-shrinkage-k", type=float, default=10_000.0,
                         help="NowcastConfig.pmf_shrinkage_k -- pass 0 to disable shrinkage (pure per-location PMF fit).")
    parser.add_argument("--disease", type=str, default="flu", choices=["flu", "covid"])
    args = parser.parse_args()

    disease = Disease.FLU if args.disease == "flu" else Disease.COVID
    nowcast_config = NowcastConfig(
        min_vintages=8, max_delay_weeks=args.max_delay_weeks, pmf_shrinkage_k=args.pmf_shrinkage_k
    )
    print(f"pmf_shrinkage_k={args.pmf_shrinkage_k}")

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
