#!/usr/bin/env python
"""
Offline/maintainer-run analysis to (re)derive the static `NHSN_MAX_DELAY_WEEKS` (and, with
--per-location, `NHSN_MAX_DELAY_WEEKS_BY_LOCATION`) constants in `iddata.constants`, following
the same exploratory approach as the `baselinenowcast` R package: fit a delay distribution over
a wide historical window of NHSN vintages, then find the delay at which a
`completeness_threshold` proportion of eventual cases have been reported.

This is NOT run as part of any automated pipeline or CI job -- run it by hand (e.g. once a
season, or if backtest results suggest NHSN's reporting-delay profile has drifted), review the
printed recommendation, and update `iddata.constants` manually.

Usage
-----
    uv run python scripts/estimate_nhsn_max_delay.py
    uv run python scripts/estimate_nhsn_max_delay.py --as-of 2026-07-25 --completeness-threshold 0.95
    uv run python scripts/estimate_nhsn_max_delay.py --per-location
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


def _fetch_location_triangles(
    disease: Disease,
    as_of: datetime.date,
    wide_max_delay_weeks: int,
    training_window_weeks: int,
) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    """
    Fetch NHSN vintages and build one reporting triangle per (location, agg_level) group.

    Shared by the pooled analysis (which stacks these into one combined fit, giving many more
    reference-week observations than any one series alone provides -- useful given NHSN's short
    calendar vintage history) and the per-location analysis (which fits each one separately, to
    check whether individual locations' own delay distributions differ enough from the pooled
    fit to warrant their own max_delay -- see the project plan's "Empirical Validation Results"
    section, which found real per-location heterogeneity even though `NHSNNowcaster` already
    fits its delay-PMF *shape* per-location at nowcast time; only the `max_delay` threshold
    itself has been a single pooled/global constant so far).
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

    triangles = {}
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
        triangles[(location, agg_level)] = matrix

    return triangles, n_skipped


def analyze_disease(
    disease: Disease,
    as_of: datetime.date,
    wide_max_delay_weeks: int,
    training_window_weeks: int,
    completeness_threshold: float,
) -> int:
    """Fit a single delay distribution pooled across every reporting location for `disease`."""
    triangles, n_skipped = _fetch_location_triangles(disease, as_of, wide_max_delay_weeks, training_window_weeks)
    print(f"  pooling {len(triangles)} location/agg_level series ({n_skipped} skipped for insufficient history)")
    pooled = stack_triangles(list(triangles.values()))
    pmf = estimate_delay(pooled)
    return estimate_max_delay(pmf, completeness_threshold=completeness_threshold)


def analyze_disease_per_location(
    disease: Disease,
    as_of: datetime.date,
    wide_max_delay_weeks: int,
    training_window_weeks: int,
    completeness_threshold: float,
) -> dict[tuple[str, str], int]:
    """
    Fit a separate delay distribution for each (location, agg_level) group, rather than pooling.
    Locations whose fitted PMF doesn't reach `completeness_threshold` within
    `wide_max_delay_weeks` columns are reported but omitted from the returned dict -- they should
    fall back to the pooled/disease-level default (or get a wider `--wide-max-delay-weeks` re-run)
    rather than being force-fit to an unreliable estimate.
    """
    triangles, n_skipped = _fetch_location_triangles(disease, as_of, wide_max_delay_weeks, training_window_weeks)
    print(f"  fitting {len(triangles)} location/agg_level series individually "
          f"({n_skipped} skipped for insufficient history)")

    recommended = {}
    n_incomplete = 0
    for (location, agg_level), matrix in triangles.items():
        pmf = estimate_delay(matrix)
        try:
            recommended[(location, agg_level)] = estimate_max_delay(pmf, completeness_threshold=completeness_threshold)
        except ValueError:
            n_incomplete += 1
    print(f"  {n_incomplete} location/agg_level series did not reach completeness_threshold="
          f"{completeness_threshold} within {wide_max_delay_weeks} delay columns; "
          "these will fall back to the disease-level default")
    return recommended


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
    parser.add_argument("--per-location", action="store_true",
                         help="Also fit a separate max_delay per (location, agg_level) instead of only the "
                              "pooled/disease-level default, and print a NHSN_MAX_DELAY_WEEKS_BY_LOCATION suggestion.")
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
    recommended_by_location = {}
    for disease in NHSN_MAX_DELAY_WEEKS:
        print(f"Analyzing NHSN/{disease.value} as of {as_of}...")
        recommended[disease] = analyze_disease(
            disease, as_of, args.wide_max_delay_weeks, args.training_window_weeks, args.completeness_threshold
        )
        current = NHSN_MAX_DELAY_WEEKS[disease]
        flag = "" if recommended[disease] == current else "  <-- differs from current constant"
        print(f"  current={current} weeks, recommended={recommended[disease]} weeks{flag}")

        if args.per_location:
            print(f"Analyzing NHSN/{disease.value} per-location as of {as_of}...")
            recommended_by_location[disease] = analyze_disease_per_location(
                disease, as_of, args.wide_max_delay_weeks, args.training_window_weeks, args.completeness_threshold
            )
            for (location, agg_level), value in sorted(recommended_by_location[disease].items()):
                flag = "" if value == recommended[disease] else "  <-- differs from pooled recommendation"
                print(f"  {location}/{agg_level}: {value} weeks{flag}")

    print("\nReview the above, then update iddata.constants.NHSN_MAX_DELAY_WEEKS if warranted:\n")
    print("NHSN_MAX_DELAY_WEEKS: dict[Disease, int] = {")
    for disease, value in recommended.items():
        print(f"    Disease.{disease.name}: {value},")
    print("}")

    if args.per_location:
        print("\nAnd NHSN_MAX_DELAY_WEEKS_BY_LOCATION, keeping only entries that meaningfully differ from the "
              "pooled default above (omitted entries fall back to it automatically):\n")
        print("NHSN_MAX_DELAY_WEEKS_BY_LOCATION: dict[Disease, dict[tuple[str, str], int]] = {")
        for disease, by_location in recommended_by_location.items():
            pooled_default = recommended[disease]
            differing = {k: v for k, v in sorted(by_location.items()) if v != pooled_default}
            if not differing:
                continue
            print(f"    Disease.{disease.name}: {{")
            for (location, agg_level), value in differing.items():
                print(f'        ("{location}", "{agg_level}"): {value},')
            print("    },")
        print("}")


if __name__ == "__main__":
    main()
