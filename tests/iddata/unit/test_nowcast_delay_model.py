"""
Unit tests for iddata.nowcast.delay_model, ported from baselinenowcast's own testthat suite
where expressible against our plain-ndarray schema, plus a few Python/iddata-specific tests.
Ported from:
    https://github.com/epinowcast/baselinenowcast/blob/main/tests/testthat/test-estimate_delay.R
    .../test-estimate_and_apply_delay.R
    .../test-validate_delay_and_triangle.R
See the bottom of this file for a rundown of which R tests were NOT ported, and why.
"""

import numpy as np
import pytest

from iddata.nowcast.delay_model import (
    _validate_delay_and_triangle,
    apply_delay,
    estimate_delay,
    estimate_delay_pooled,
)


def _r_all_equal_numeric(target: np.ndarray, current: np.ndarray, tolerance: float) -> bool:
    """
    Approximates base R's `all.equal.numeric()` default tolerance check, which is what
    testthat's `expect_equal(..., tol=...)` uses under the hood: the POOLED relative
    difference across the whole array (mean absolute difference / mean absolute target), not a
    per-element tolerance. `pytest.approx(..., rel=tolerance)` checks every element
    individually, which is stricter than what the R tests this module ports actually assert.
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)
    mean_abs_target = np.mean(np.abs(target))
    mean_abs_diff = np.mean(np.abs(target - current))
    relative_diff = mean_abs_diff / mean_abs_target if mean_abs_target > tolerance else mean_abs_diff
    return relative_diff <= tolerance


def _staircase_nan(matrix: np.ndarray) -> np.ndarray:
    """
    Apply the standard "staircase" reporting structure to a complete matrix: row i (0-indexed,
    oldest first) is missing column d if its age (n_dates - 1 - i) < d -- i.e. the most recent
    row only has its delay-0 value observed, the second-most-recent has delays 0-1, etc.
    Mirrors baselinenowcast's default `apply_reporting_structure()` behavior in its test fixtures.
    """
    result = matrix.astype(float).copy()
    n_dates, n_delays = result.shape
    for i in range(n_dates):
        age = n_dates - 1 - i
        for d in range(n_delays):
            if age < d:
                result[i, d] = np.nan
    return result


# Create a triangle with known delay PMF
SIM_DELAY_PMF = np.array([0.4, 0.3, 0.2, 0.1])

# Generate counts for each reference date
COUNTS = np.array([100, 150, 200, 250, 300])

# Create a complete triangle based on the known delay PMF
COMPLETE_TRIANGLE = np.outer(COUNTS, SIM_DELAY_PMF)

# Create a reporting triangle with NAs in the lower right
REPORTING_TRIANGLE = _staircase_nan(COMPLETE_TRIANGLE)


class TestEstimateDelayPortedFromR:
    """Ported from test-estimate_delay.R."""

    def test_returns_a_valid_probability_mass_function(self):
        result = estimate_delay(REPORTING_TRIANGLE)
        assert isinstance(result, np.ndarray)
        assert len(result) == REPORTING_TRIANGLE.shape[1]
        assert np.all((result >= 0) & (result <= 1))
        assert result.sum() == pytest.approx(1.0, abs=1e-6)
        assert result == pytest.approx(SIM_DELAY_PMF, abs=1e-6)


    def test_works_with_truncated_triangles(self):
        result_full = estimate_delay(REPORTING_TRIANGLE)
        assert len(result_full) == 4
        assert result_full == pytest.approx(SIM_DELAY_PMF, abs=1e-6)

        # truncate_to_delay(reporting_triangle, max_delay = 2) -- drop the last delay column
        truncated_triangle = REPORTING_TRIANGLE[:, :3]
        result_truncated = estimate_delay(truncated_triangle)
        assert len(result_truncated) == 3
        expected_truncated = SIM_DELAY_PMF[:3] / SIM_DELAY_PMF[:3].sum()
        assert result_truncated == pytest.approx(expected_truncated, abs=1e-6)


    def test_handles_custom_n_history_parameter(self):
        result_full = estimate_delay(REPORTING_TRIANGLE, n=5)
        result_partial = estimate_delay(REPORTING_TRIANGLE, n=4)
        assert result_full == pytest.approx(SIM_DELAY_PMF, abs=1e-6)
        assert result_partial == pytest.approx(SIM_DELAY_PMF, abs=1e-6)


    def test_validates_input_parameters_correctly(self):
        # Ported from the single "estimate_delay validates input parameters correctly" test_that
        # block, which loops over a `cases` list of (args, expected regex) pairs. The R version
        # has two additional cases we do NOT port here -- a ragged triangle ("must contain at
        # least one row with no missing") and an all-zeros triangle ("only contain 0s") -- since
        # `estimate_delay` here doesn't port `.validate_for_delay_estimation()`'s corresponding
        # checks (see this module's and delay_model.py's docstrings for why).
        n_rows = REPORTING_TRIANGLE.shape[0]
        cases = [
            (dict(n=0), "greater than or equal to 1"),
            (dict(n=10), f"Reporting triangle has {n_rows} reference times but n = 10 was requested"),
        ]
        for kwargs, regex in cases:
            with pytest.raises(ValueError, match=regex):
                estimate_delay(REPORTING_TRIANGLE, **kwargs)


    def test_calculates_correct_pmf_with_complete_matrix(self):
        # "estimate_delay calculates correct PMF with complete matrix" -- rounded counts, no NaNs
        complete_pmf = np.array([0.45, 0.25, 0.2, 0.1])
        complete_counts = np.array([80, 100, 90, 80, 70])
        full_triangle = np.round(np.outer(complete_counts, complete_pmf))

        delay_pmf = estimate_delay(full_triangle, n=5)

        assert delay_pmf == pytest.approx(complete_pmf, abs=1e-3)


# sim_delay_pmf/counts for the estimate_and_apply_delay fixtures (a separate, longer series from
# the one above, matching test-estimate_and_apply_delay.R's own module-level fixture)
JOINT_SIM_DELAY_PMF = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
JOINT_COUNTS = np.array([30, 40, 50, 60, 70, 50, 40, 50, 80, 40])
JOINT_COMPLETE_TRIANGLE = np.outer(JOINT_COUNTS, JOINT_SIM_DELAY_PMF)
JOINT_REPORTING_TRIANGLE = _staircase_nan(JOINT_COMPLETE_TRIANGLE)


class TestEstimateAndApplyDelayPortedFromR:
    """Ported from test-estimate_and_apply_delay.R (only the part exercising estimate_delay +
    apply_delay directly -- see this file's bottom section for what else that R file covers
    that isn't ported here, since it's a `baselinenowcast`-package-level convenience wrapper we
    don't have)."""

    def test_estimate_and_apply_delay_recovers_complete_triangle(self):
        delay_pmf = estimate_delay(JOINT_REPORTING_TRIANGLE)
        point_nowcast_matrix = apply_delay(JOINT_REPORTING_TRIANGLE, delay_pmf)

        # tol = 0.2 to match the R test exactly; see _r_all_equal_numeric's docstring for why
        # this isn't a plain pytest.approx call.
        assert _r_all_equal_numeric(JOINT_COMPLETE_TRIANGLE, point_nowcast_matrix, tolerance=0.2)


    def test_apply_delay_fills_all_nans(self):
        delay_pmf = estimate_delay(JOINT_REPORTING_TRIANGLE)
        result = apply_delay(JOINT_REPORTING_TRIANGLE, delay_pmf)

        assert not np.isnan(result).any()
        assert result.shape == JOINT_REPORTING_TRIANGLE.shape


    def test_apply_delay_refuses_to_nowcast_a_complete_triangle(self):
        # Not a literal port -- exercises apply_delay's own `n_row_nas == 0` check (outside
        # `_validate_delay_and_triangle`, see TestValidateDelayAndTrianglePortedFromR below),
        # using JOINT_COMPLETE_TRIANGLE, which has no NaNs at all.
        with pytest.raises(ValueError, match="nothing to nowcast"):
            apply_delay(JOINT_COMPLETE_TRIANGLE, JOINT_SIM_DELAY_PMF)


# Shared inputs for TestValidateDelayAndTrianglePortedFromR
VALID_TRIANGLE = np.arange(1, 13).reshape(4, 3).T.astype(float)  # matrix(1:12, nrow=3, ncol=4) is column-major
VALID_DELAY_PMF = np.array([0.4, 0.3, 0.2, 0.1])


class TestValidateDelayAndTrianglePortedFromR:
    """
    Ported from test-validate_delay_and_triangle.R, exercising `_validate_delay_and_triangle`
    directly (length mismatch, the delay=0-with-insufficient-data case, and the negative-
    first-entry case) -- NOT the separate `n_row_nas == 0` "nothing to nowcast" check, which R
    also keeps outside `.validate_delay_and_triangle`, directly in `apply_delay`'s own body (see
    `test_apply_delay_refuses_to_nowcast_a_complete_triangle` above for that one). The R file's
    other cases (non-matrix triangle, non-numeric PMF, non-integer values accepted, empty
    triangle, empty PMF) test `checkmate`-based R type assertions that don't have a meaningful
    Python/numpy equivalent to port -- see this file's bottom section.
    """

    def test_valid_inputs_pass_validation(self):
        _validate_delay_and_triangle(VALID_TRIANGLE, VALID_DELAY_PMF)  # should not raise


    def test_mismatched_lengths_cause_error(self):
        mismatched_delay = np.array([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="Length of the delay PMF is not the same as the number of delays"):
            _validate_delay_and_triangle(VALID_TRIANGLE, mismatched_delay)


    def test_delay_pmf_0_with_insufficient_triangle_causes_error(self):
        triangle = np.array([
            [10, 5, 5, 5],
            [20, 10, 10, np.nan],
            [40, 20, np.nan, np.nan],
            [1, np.nan, np.nan, np.nan],
        ])
        delay_pmf = np.array([0, 0.2, 0.4, 0.2])
        with pytest.raises(ValueError, match="insufficient information"):
            _validate_delay_and_triangle(triangle, delay_pmf)


    def test_negative_first_pmf_entry_causes_error(self):
        triangle = np.array([
            [10, 5, 5, 5],
            [20, 10, 10, np.nan],
            [40, 20, np.nan, np.nan],
        ])
        delay_pmf = np.array([-0.1, 0.5, 0.4, 0.2])
        with pytest.raises(ValueError, match="First entry of delay PMF .* is negative"):
            _validate_delay_and_triangle(triangle, delay_pmf)


    def test_accepts_negative_pmf_at_later_delays(self):
        triangle = np.array([
            [10, 5, 5, 5],
            [20, 10, 10, np.nan],
            [40, 20, np.nan, np.nan],
        ])
        # Negative at delay 2, not delay 0
        delay_pmf = np.array([0.7, 0.4, -0.1, 0.0])
        _validate_delay_and_triangle(triangle, delay_pmf)  # should not raise


class TestEstimateDelayPooled:
    """
    Not a port of anything in baselinenowcast -- this project-specific function fits one
    delay-PMF from several independent reporting triangles (e.g. different locations, or flu and
    COVID sharing a reporting pipeline), each with its own trailing incomplete rows. See its
    docstring for why naively `np.vstack`-ing raw triangles and calling `estimate_delay` on the
    result is WRONG: `_chainladder_fill_triangle` assumes one single monotonic triangle and
    silently overwrites every row after the first missing one, including later triangles'
    genuinely-observed values, with unrelated computed placeholders.
    """

    def test_matches_estimate_delay_for_a_single_triangle(self):
        # Pooling exactly one triangle must be identical to not pooling at all.
        triangle = np.array([[10.0, 2.0], [8.0, 3.0], [5.0, np.nan]])
        np.testing.assert_allclose(estimate_delay_pooled([triangle]), estimate_delay(triangle))


    def test_does_not_corrupt_a_second_triangles_genuinely_observed_values(self):
        # group_b's ratios are deliberately unlike group_a's, so if group_b's real values were
        # overwritten with group_a-derived placeholders, the result would be detectably wrong.
        group_a = np.array([[10.0, 2.0], [8.0, 3.0], [5.0, np.nan]])
        group_b = np.array([[100.0, 5.0], [80.0, 40.0], [50.0, np.nan]])

        pmf = estimate_delay_pooled([group_a, group_b])

        naive_and_wrong = estimate_delay(np.vstack([group_a, group_b]))
        assert not np.allclose(pmf, naive_and_wrong)

        # Direct check: group_b's real observed delay-1 values (5.0, 40.0) must appear,
        # unmodified, in the totals implied by the pooled fit's own filled-triangle arithmetic.
        # Recompute what estimate_delay_pooled does internally and confirm group_b's rows 0-1
        # weren't touched by group_a's chain-ladder fill.
        from iddata.nowcast.delay_model import _chainladder_fill_triangle

        filled_b = _chainladder_fill_triangle(group_b)
        np.testing.assert_array_equal(filled_b[:2], group_b[:2])  # already-complete rows untouched


    def test_pooling_two_identical_triangles_gives_the_same_pmf_as_one(self):
        triangle = np.array([[10.0, 2.0], [8.0, 3.0], [5.0, np.nan]])
        pmf_single = estimate_delay(triangle)
        pmf_pooled = estimate_delay_pooled([triangle, triangle.copy()])
        np.testing.assert_allclose(pmf_pooled, pmf_single)


# ---------------------------------------------------------------------------------------------
# R tests NOT ported here (see PR/plan discussion for the full rundown)
# ---------------------------------------------------------------------------------------------
#
# From test-estimate_delay.R:
#   - "errors when NAs are in upper part of reporting triangle", "errors if not passed a
#     matrix", "handles diagonal reporting triangles" (structure=1 ragged), "works with every
#     other day reporting of daily data" (structure=2 ragged), and all four
#     preserves/handles-negative-values tests -- these exercise ragged-reporting-structure
#     support and negative-PMF/downward-correction handling, which this port explicitly doesn't
#     implement (see delay_model.py's module docstring).
#
# From test-estimate_and_apply_delay.R:
#   - "estimate_and_apply_delay errors when n_history_delay is misspecified", "works with every
#     other day reporting of daily data (ragged triangle test)", "messages if max delay is
#     specified as higher than reporting triangle", "works with custom delay_pmf", "works with
#     different n values", "errors when n is too large" -- ALL of these test
#     `estimate_and_apply_delay()`, a `baselinenowcast`-package convenience wrapper that calls
#     `estimate_delay()` then `apply_delay()` together and also preserves the R
#     `reporting_triangle` S3 class on its result. We don't have that wrapper (callers here just
#     call `estimate_delay` then `apply_delay` directly, as `NHSNNowcaster` and the tests above
#     do), so most of these don't have a direct target to port to. The "custom delay_pmf" case
#     is the exception worth noting explicitly: `apply_delay` here already always takes an
#     externally-supplied `delay_pmf` (there's no "estimate internally" mode to contrast it
#     against), so that scenario is inherently covered by every `apply_delay` call in this file.
#   - "errors when no complete rows" is a REAL GAP, not just a skipped test: this exercises
#     `.validate_for_delay_estimation()`'s "has_complete_row" check, which `estimate_delay` here
#     does not port (see delay_model.py's docstring). Without it, calling `estimate_delay` on a
#     triangle whose training window contains no fully-observed row won't raise a clear error --
#     `_chainladder_fill_triangle`'s block-sum computations would silently degrade (e.g. an
#     empty `block_top`/`block_top_left` sum) rather than failing loudly. This should be added as
#     a follow-up validation check before this code is trusted on arbitrary/adversarial inputs;
#     it isn't purely a missing-test issue.
#
# From test-validate_delay_and_triangle.R: "accepts non-integer values", "non-matrix triangle
#   causes error", "non-numeric delay PMF causes error", "empty triangle causes error", "empty
#   delay PMF causes error" -- all test R/checkmate-specific type assertions (matrix class,
#   numeric class, non-empty-ness) that don't map onto a meaningful Python/numpy check; a numpy
#   array is always "matrix-like" and empty PMF/triangle inputs already fail one of our other
#   ported checks (e.g. length mismatch) rather than needing a dedicated type-assertion test.
