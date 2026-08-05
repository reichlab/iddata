"""
Reporting-triangle construction for NHSN nowcasting, plus max-delay estimation from a fitted
delay distribution.

Unlike `iddata.nowcast.delay_model`, nothing in this module is a port of `baselinenowcast` (or
of RESPINOW, which it was itself adapted from) -- there is no equivalent function in that R
package to port, because its reporting-triangle constructors (`as_reporting_triangle()`,
`make_test_triangle()`, `get_delays_from_dates()`, etc.) expect either a pre-built matrix or a
long-format table of individual report events with report/reference dates, whereas our input is
repeated `NHSNDataSource.load(as_of=...)` vintage snapshots (each a full historical series as it
looked as of a given date, not a per-event report log). `build_increment_triangle` is new glue
code written specifically to turn that vintage-snapshot shape into the incremental
reference-week x delay matrix that `delay_model.estimate_delay`/`apply_delay` (which ARE
faithful ports) expect as input. `estimate_max_delay` is likewise new code, embodying the same
"delay at which a completeness threshold is reached" idea described in the package's
exploratory-analysis vignette, but is not a port of an exported package function.

The triangle built here holds *increments* between successive vintages of the same reference
week (not raw cumulative values) -- see `build_increment_triangle` -- matching the convention
expected by `iddata.nowcast.delay_model.estimate_delay`/`apply_delay`.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def weekly_as_of_dates(as_of: datetime.date, n_weeks: int) -> list[datetime.date]:
    """The `n_weeks` most recent weekly as_of dates ending at (and including) `as_of`, oldest first."""
    return [as_of - datetime.timedelta(weeks=w) for w in range(n_weeks - 1, -1, -1)]


def build_increment_triangle(
    vintage_series: dict[datetime.date, pd.Series],
    as_of: datetime.date,
    max_delay_weeks: int,
    training_window_weeks: int,
) -> tuple[np.ndarray, list[datetime.date]]:
    """
    Build a reference-week x delay increment matrix for a single (location, agg_level) group.

    Parameters
    ----------
    vintage_series : dict[datetime.date, pd.Series]
        Maps a vintage `as_of` date to a Series of `inc` values indexed by `wk_end_date`
        (cumulative/revised values as reported in that vintage), for one location/agg_level.
        Typically produced by slicing the frames returned by `VintageCache.get_many`.
    as_of : datetime.date
        The current reference date (the vintage we're nowcasting).
    max_delay_weeks : int
        Number of delay columns (0..max_delay_weeks) in the triangle.
    training_window_weeks : int
        Number of trailing reference weeks (rows) to include.

    Returns
    -------
    (matrix, ref_dates) : the (training_window_weeks, max_delay_weeks + 1) increment matrix
        (np.nan where not yet observed, i.e. `wk_end_date + delay weeks > as_of`) and the
        corresponding reference `wk_end_date` values for its rows, oldest first.
    """
    vintage_dates = sorted(vintage_series)
    ref_dates = weekly_as_of_dates(as_of, training_window_weeks)
    n_delays = max_delay_weeks + 1

    def _cumulative_value(target_as_of: datetime.date, wk_end_date: datetime.date) -> float:
        """Cumulative `inc` value for `wk_end_date`, per the latest vintage <= target_as_of."""
        eligible = [v for v in vintage_dates if v <= target_as_of]
        if not eligible:
            return np.nan
        return vintage_series[eligible[-1]].get(pd.Timestamp(wk_end_date), np.nan)

    cum = np.full((len(ref_dates), n_delays), np.nan)
    for i, ref_date in enumerate(ref_dates):
        for d in range(n_delays):
            target_as_of = ref_date + datetime.timedelta(weeks=d)
            if target_as_of > as_of:
                break  # not yet observed at this delay; remaining columns in this row stay NaN
            cum[i, d] = _cumulative_value(target_as_of, ref_date)

    increments = _increments_from_cumulative(cum)
    return increments, ref_dates


def _increments_from_cumulative(cum: np.ndarray) -> np.ndarray:
    """
    Convert a cumulative-value matrix to increments, treating each row's *first* non-NaN delay
    as its base increment (as if diffed from an implicit 0) and computing normal consecutive
    diffs from there. This is NOT equivalent to `np.diff(cum, axis=1, prepend=0.0)`: some
    sources (e.g. NHSN's current reporting pipeline) have a structural minimum publication lag,
    so cum[:, 0] can be NaN not because it's "not yet observed" in the future-cutoff sense but
    because that delay is never populated for ANY reference week. A plain prepend=0.0 diff would
    let that leading NaN poison the very next (otherwise perfectly valid) column, since
    `real_value - NaN` is NaN.

    Leading structural gaps get an increment of exactly 0.0, not NaN: the absence of any report
    at those delays is a *certain* fact (nothing has ever arrived yet), not an unknown value to
    be nowcasted later -- unlike trailing NaN (genuinely not yet observed because as_of hasn't
    reached that delay), which must stay NaN so `apply_delay` knows to fill it in, and which
    still correctly stops the increment computation once encountered.

    A row with NO valid (non-NaN) delay at all (e.g. a location/vintage combination with no
    data whatsoever, not merely an early-delay publication lag) is a different case and is left
    entirely NaN -- there, we have no confirmed information, so claiming a "certain zero" would
    be wrong; this is what lets callers distinguish "genuinely no data" from "fully observed".
    """
    n_dates, n_delays = cum.shape
    increments = np.full((n_dates, n_delays), np.nan)
    for i in range(n_dates):
        valid_delays = np.where(~np.isnan(cum[i, :]))[0]
        if len(valid_delays) == 0:
            continue  # no observations anywhere in this row; leave entirely NaN
        first_valid = valid_delays[0]
        increments[i, :first_valid] = 0.0  # leading structural gap: certainly zero, not unknown

        prev_value = 0.0
        for d in range(first_valid, n_delays):
            if np.isnan(cum[i, d]):
                break  # genuinely not yet observed from here on; stop
            increments[i, d] = cum[i, d] - prev_value
            prev_value = cum[i, d]
    return increments


def stack_triangles(triangles: list[np.ndarray]) -> np.ndarray:
    """
    Vertically stack same-shaped (n_dates, n_delays) increment triangles from multiple
    (location, agg_level) groups into one combined triangle, so a single delay distribution can
    be fit jointly across all of them -- this gives many more reference-week observations than
    any one location/aggregate series alone provides, which matters given NHSN's short
    calendar vintage history. Used by `scripts/estimate_nhsn_max_delay.py` to pool across all
    reporting locations rather than relying on a single (e.g. national-only) series; not used
    by `NHSNNowcaster`, which fits per-(location, agg_level) at nowcast time.
    """
    return np.vstack(triangles)


def estimate_max_delay(delay_pmf: np.ndarray, completeness_threshold: float = 0.95) -> int:
    """
    Find the smallest delay D (0-indexed) at which the cumulative delay distribution reaches
    `completeness_threshold`.

    This is an offline/exploratory analysis step -- see `scripts/estimate_nhsn_max_delay.py`,
    which uses this to derive the static `NHSN_MAX_DELAY_WEEKS` constants in
    `iddata.constants` -- not something called at nowcast-run time (see project plan for why
    max-delay determination is static rather than live-recomputed per run).

    Parameters
    ----------
    delay_pmf : np.ndarray
        A delay probability mass function, e.g. from `delay_model.estimate_delay` fit over a
        wide historical window with a generous number of delay columns.
    completeness_threshold : float
        Proportion of eventual cases that must be reported by the returned delay.

    Returns
    -------
    int : the estimated max delay, in weeks.
    """
    cdf = np.cumsum(delay_pmf)
    idx = np.searchsorted(cdf, completeness_threshold)
    if idx >= len(cdf):
        raise ValueError(
            f"Cumulative delay distribution does not reach completeness_threshold="
            f"{completeness_threshold} within the fitted range (max cdf={cdf[-1]:.4f}); "
            "refit with more delay columns."
        )
    return int(idx)
