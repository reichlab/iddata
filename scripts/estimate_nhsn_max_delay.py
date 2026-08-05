#!/usr/bin/env python
"""
Offline/maintainer-run analysis to (re)derive the static `NHSN_MAX_DELAY_WEEKS` constants in
`iddata.constants`, following the same exploratory approach as the `baselinenowcast` R
package: fit a delay distribution over a wide historical window of NHSN vintages, then find
the delay at which a `completeness_threshold` proportion of eventual cases have been reported.

This is NOT run as part of any automated pipeline or CI job -- run it by hand (e.g. once a
season, or if backtest results suggest NHSN's reporting-delay profile has drifted), review the
printed recommendation, and update `iddata.constants.NHSN_MAX_DELAY_WEEKS` manually.

Usage
-----
    uv run python scripts/estimate_nhsn_max_delay.py
    uv run python scripts/estimate_nhsn_max_delay.py --as-of 2026-07-25 --completeness-threshold 0.95
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

from iddata.constants import NHSN_MAX_DELAY_WEEKS
from iddata.enums import Disease
from iddata.nowcast.delay_model import estimate_delay
from iddata.nowcast.nhsn import cap_training_window_to_cutover
from iddata.nowcast.triangle import build_increment_triangle, estimate_max_delay, stack_triangles, weekly_as_of_dates
from iddata.nowcast.vintage_cache import VintageCache
from iddata.sources.nhsn import NHSNDataSource


def analyze_disease(
    disease: Disease,
    as_of: datetime.date,
    wide_max_delay_weeks: int,
    training_window_weeks: int,
    completeness_threshold: float,
) -> int:
    """
    Fit a single delay distribution pooled across every reporting location (all states plus
    national) for `disease`, rather than relying on the national series alone -- pooling gives
    many more reference-week observations than any one series alone provides, which matters
    given NHSN's short calendar vintage history.
    """
    capped_training_window_weeks = cap_training_window_to_cutover(as_of, wide_max_delay_weeks, training_window_weeks)
    if capped_training_window_weeks < training_window_weeks:
        print(
            f"  requested training_window_weeks={training_window_weeks} would reach before the "
            f"NHSN source cutover; capping to {capped_training_window_weeks} to avoid mixing in "
            "the incompatible legacy HHS archive"
        )
    training_window_weeks = capped_training_window_weeks

    source = NHSNDataSource(disease=disease)
    as_of_dates = weekly_as_of_dates(as_of, training_window_weeks + wide_max_delay_weeks)

    cache = VintageCache()
    vintages = cache.get_many(source, as_of_dates)
    n_distinct = cache.n_distinct_vintages(as_of_dates)
    print(f"  {n_distinct} distinct vintages available (of {len(as_of_dates)} weekly as_of dates requested)")

    latest = vintages[max(vintages)]
    location_groups = latest[["location", "agg_level"]].drop_duplicates().itertuples(index=False)

    triangles = []
    n_skipped = 0
    for location, agg_level in location_groups:
        series_by_vintage = {}
        for v, df in vintages.items():
            sub = df[(df["location"] == location) & (df["agg_level"] == agg_level)]
            if not sub.empty:
                series_by_vintage[v] = sub.set_index("wk_end_date")["inc"]

        matrix, _ = build_increment_triangle(series_by_vintage, as_of, wide_max_delay_weeks, training_window_weeks)
        # Skip locations with insufficient history for the chain-ladder fill's invariant that
        # the oldest row(s) used for delay estimation are fully complete (e.g. a location that
        # was only recently added to reporting).
        if np.isnan(matrix[0, :]).any():
            n_skipped += 1
            continue
        triangles.append(matrix)

    print(f"  pooling {len(triangles)} location/agg_level series ({n_skipped} skipped for insufficient history)")
    pooled = stack_triangles(triangles)
    pmf = estimate_delay(pooled)
    return estimate_max_delay(pmf, completeness_threshold=completeness_threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=datetime.date.fromisoformat, default=None,
                         help="Reference date to analyze from (default: today).")
    parser.add_argument("--wide-max-delay-weeks", type=int, default=12,
                         help="Generous upper bound on delay columns, to observe the full completion curve.")
    parser.add_argument("--training-window-weeks", type=int, default=52,
                         help="Number of trailing reference weeks to fit the delay distribution over.")
    parser.add_argument("--completeness-threshold", type=float, default=0.95,
                         help="Proportion of eventual cases that must be reported by the estimated max delay.")
    args = parser.parse_args()

    as_of = args.as_of or datetime.date.today()
    if as_of.weekday() != 5:  # Monday=0, ..., Saturday=5
        snapped = as_of - datetime.timedelta(days=(as_of.weekday() - 5) % 7)
        print(
            f"--as-of {as_of} is a {as_of.strftime('%A')}, but NHSN's wk_end_date grid is always "
            f"Saturdays; a misaligned as_of silently produces an empty/unusable pooled triangle "
            f"rather than an error. Snapping to the most recent Saturday: {snapped}."
        )
        as_of = snapped

    recommended = {}
    for disease in NHSN_MAX_DELAY_WEEKS:
        print(f"Analyzing NHSN/{disease.value} as of {as_of}...")
        recommended[disease] = analyze_disease(
            disease, as_of, args.wide_max_delay_weeks, args.training_window_weeks, args.completeness_threshold
        )
        current = NHSN_MAX_DELAY_WEEKS[disease]
        flag = "" if recommended[disease] == current else "  <-- differs from current constant"
        print(f"  current={current} weeks, recommended={recommended[disease]} weeks{flag}")

    print("\nReview the above, then update iddata.constants.NHSN_MAX_DELAY_WEEKS if warranted:\n")
    print("NHSN_MAX_DELAY_WEEKS: dict[Disease, int] = {")
    for disease, value in recommended.items():
        print(f"    Disease.{disease.name}: {value},")
    print("}")


if __name__ == "__main__":
    main()
