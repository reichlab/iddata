"""Unit tests for iddata.nowcast.triangle -- new code (not a port), see that module's docstring."""

import datetime

import numpy as np
import pandas as pd
import pytest

from iddata.nowcast.triangle import (
    _increments_from_cumulative,
    build_increment_triangle,
    estimate_max_delay,
    stack_triangles,
    weekly_as_of_dates,
)


class TestWeeklyAsOfDates:
    def test_returns_n_weeks_oldest_first_ending_at_as_of(self):
        as_of = datetime.date(2026, 1, 24)
        result = weekly_as_of_dates(as_of, 4)
        assert result == [
            datetime.date(2026, 1, 3),
            datetime.date(2026, 1, 10),
            datetime.date(2026, 1, 17),
            datetime.date(2026, 1, 24),
        ]


    def test_single_week_is_just_as_of(self):
        as_of = datetime.date(2026, 1, 24)
        assert weekly_as_of_dates(as_of, 1) == [as_of]


class TestBuildIncrementTriangle:
    """
    Uses a synthetic scenario with known true final values and a known cumulative-completion
    curve to verify build_increment_triangle recovers the expected increments exactly, and
    correctly leaves not-yet-observed cells as NaN.
    """

    def _make_scenario(self):
        as_of = datetime.date(2026, 1, 24)
        ref_dates = weekly_as_of_dates(as_of, 3)  # [rd0 (oldest), rd1, rd2 == as_of]
        true_finals = [100, 200, 300]  # one per ref_date
        fraction_by_delay = [0.5, 0.8, 1.0]  # cumulative proportion reported by delay 0, 1, 2

        # Vintage v can only report cumulative values for reference weeks t <= v, at
        # delay = (v - t) in weeks, i.e. true_final[t] * fraction_by_delay[delay].
        vintage_series = {}
        for v_idx, v in enumerate(ref_dates):
            values = {}
            for t_idx, t in enumerate(ref_dates):
                delay_weeks = (v - t).days // 7
                if delay_weeks < 0:
                    continue
                values[pd.Timestamp(t)] = true_finals[t_idx] * fraction_by_delay[delay_weeks]
            vintage_series[v] = pd.Series(values)

        return as_of, ref_dates, true_finals, vintage_series


    def test_recovers_expected_increments_and_nans(self):
        as_of, ref_dates, true_finals, vintage_series = self._make_scenario()

        matrix, returned_ref_dates = build_increment_triangle(
            vintage_series, as_of, max_delay_weeks=2, training_window_weeks=3
        )

        assert returned_ref_dates == ref_dates
        expected = np.array([
            [50.0, 30.0, 20.0],   # true_final=100: 50, +30 (=.8-.5), +20 (=1.0-.8)
            [100.0, 60.0, np.nan],  # true_final=200: 100, +60; delay=2 not yet observed
            [150.0, np.nan, np.nan],  # true_final=300: 150; delays 1, 2 not yet observed
        ])
        np.testing.assert_allclose(matrix, expected, equal_nan=True)
        # Completed rows sum to the true final value; the incomplete rows' observed-so-far sum
        # is less than the true final (that's exactly what needs nowcasting).
        assert np.nansum(matrix[0, :]) == pytest.approx(true_finals[0])
        assert np.nansum(matrix[1, :]) < true_finals[1]
        assert np.nansum(matrix[2, :]) < true_finals[2]


    def test_missing_location_produces_all_nan_row(self):
        # A vintage_series with no data at all for a group should just produce an all-NaN
        # triangle rather than raising -- callers (NHSNNowcaster) skip such groups.
        matrix, ref_dates = build_increment_triangle({}, datetime.date(2026, 1, 24), 2, 3)
        assert matrix.shape == (3, 3)
        assert np.isnan(matrix).all()


class TestIncrementsFromCumulativeStructuralLag:
    """
    Regression coverage for a real bug found against live NHSN data: the current NHSN
    reporting pipeline has a structural minimum publication lag -- a vintage dated as_of=X
    never covers wk_end_date=X itself, only X-7days or earlier -- so delay=0 is NaN for every
    reference week, always, not just recently-incomplete ones. A naive `np.diff(cum, axis=1,
    prepend=0.0)` lets that leading NaN poison the very next (otherwise valid) column, since
    `real_value - NaN` is NaN, which cascades into every downstream computation (the whole
    delay PMF, and `apply_delay`'s row sums, since summing any NaN propagates NaN for that row).
    """

    def test_leading_structural_gap_becomes_zero_not_nan(self):
        # Row 0: delay=0 structurally missing (leading gap), then real values -- a fully
        # complete row in every practical sense once the structural gap is accounted for.
        # Row 1: delay=0 structurally missing, then a real value, then a genuine trailing gap
        # (not yet observed) -- still incomplete, and must stay NaN so apply_delay fills it in.
        cum = np.array([
            [np.nan, 10.0, 16.0, 20.0],
            [np.nan, 8.0, np.nan, np.nan],
        ])

        increments = _increments_from_cumulative(cum)

        expected = np.array([
            [0.0, 10.0, 6.0, 4.0],
            [0.0, 8.0, np.nan, np.nan],
        ])
        np.testing.assert_allclose(increments, expected, equal_nan=True)
        # The leading gap is a real, non-NaN zero -- not a missing value.
        assert not np.isnan(increments[0, 0])
        assert not np.isnan(increments[1, 0])
        # The genuine trailing gap (not yet observed) must remain NaN so it still gets nowcasted.
        assert np.isnan(increments[1, 2])
        assert np.isnan(increments[1, 3])
        # A fully-complete row (accounting for the structural gap) has no NaN at all.
        assert not np.isnan(increments[0, :]).any()
        assert np.nansum(increments[0, :]) == pytest.approx(20.0)


    def test_build_increment_triangle_with_realistic_publication_lag(self):
        # Mirrors the real NHSN characteristic: a vintage v only ever reports values for
        # reference weeks strictly older than v (delay >= 1 week; delay=0 is never populated).
        as_of = datetime.date(2026, 1, 24)
        ref_dates = weekly_as_of_dates(as_of, 3)  # rd0 (oldest), rd1, rd2 == as_of
        true_finals = [100, 200, 300]
        fraction_by_delay = [0.5, 0.8, 1.0]  # true completion fractions at delay 0, 1, 2

        vintage_dates = weekly_as_of_dates(as_of, 4)  # one extra earlier vintage for range
        vintage_series = {}
        for v in vintage_dates:
            values = {}
            for t_idx, t in enumerate(ref_dates):
                delay_weeks = (v - t).days // 7
                if delay_weeks < 1:  # structural lag: delay=0 is never populated
                    continue
                capped_delay = min(delay_weeks, 2)
                values[pd.Timestamp(t)] = true_finals[t_idx] * fraction_by_delay[capped_delay]
            vintage_series[v] = pd.Series(values)

        matrix, returned_ref_dates = build_increment_triangle(
            vintage_series, as_of, max_delay_weeks=2, training_window_weeks=3
        )

        assert returned_ref_dates == ref_dates
        # Oldest row (rd0) is fully observed by as_of (delay=2 reached) -- no NaN anywhere,
        # including at the structurally-always-missing delay=0 column, and its total matches
        # the true final value exactly.
        assert not np.isnan(matrix[0, :]).any()
        assert np.nansum(matrix[0, :]) == pytest.approx(true_finals[0])
        # Most recent row (rd2 == as_of): no vintage is ever taken *after* its own reference
        # week, so nothing has been observed at any delay yet -- the whole row stays NaN (not
        # even delay=0 becomes a "confirmed zero", since we have no vintage at all to confirm
        # that from; see TestIncrementsFromCumulativeStructuralLag for the all-NaN-row case).
        assert np.isnan(matrix[2, :]).all()


class TestEstimateMaxDelay:
    def test_finds_delay_reaching_threshold(self):
        pmf = np.array([0.5, 0.3, 0.15, 0.05])  # cdf = [0.5, 0.8, 0.95, 1.0]
        assert estimate_max_delay(pmf, completeness_threshold=0.95) == 2
        assert estimate_max_delay(pmf, completeness_threshold=0.8) == 1
        assert estimate_max_delay(pmf, completeness_threshold=0.5) == 0


    def test_raises_if_threshold_never_reached(self):
        pmf = np.array([0.5, 0.3])  # cdf tops out at 0.8
        with pytest.raises(ValueError, match="does not reach completeness_threshold"):
            estimate_max_delay(pmf, completeness_threshold=0.95)


class TestStackTriangles:
    def test_vstacks_same_shaped_triangles(self):
        a = np.array([[1.0, 2.0], [3.0, np.nan]])
        b = np.array([[5.0, 6.0]])
        result = stack_triangles([a, b])
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]), equal_nan=True)
