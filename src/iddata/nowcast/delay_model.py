"""
Chain-ladder delay-distribution estimation and point-nowcast formula.

Ported as closely as possible (same function/variable names, same helper decomposition, same
comments, same file-internal ordering) from the R package `baselinenowcast`
(https://github.com/epinowcast/baselinenowcast, MIT licensed):
    https://github.com/epinowcast/baselinenowcast/blob/main/R/estimate_delay.R
    https://github.com/epinowcast/baselinenowcast/blob/main/R/apply_delay.R
whose `estimate_delay()`/`apply_delay()` were themselves adapted from code written by the
Karlsruhe Institute of Technology RESPINOW German Hospitalization Nowcasting Hub (MIT licensed):
https://github.com/KITmetricslab/RESPINOW-Hub/blob/7cce3ae2728116e8c8cc0e4ab29074462c24650e/code/baseline/functions.R#L55

The functions below are grouped in two clusters, one per source file, each in the same order as
the R source: `estimate_delay` and its private helpers (mirroring `estimate_delay.R`), followed
by `apply_delay` and its private helpers (mirroring `apply_delay.R`). `_extract_block_bottom_left`
is shared by both clusters, exactly as in the R source, and is defined once, in the cluster
where `estimate_delay.R` defines it.

R is 1-indexed and this module is 0-indexed, so specific integer index values differ by one
from the R source in places -- but the control flow, helper decomposition, and variable/function
names are kept as close to the original as possible so the two can be diffed side by side. R's
`reporting_triangle` is a matrix wrapped in a custom S3 class carrying extra attributes
(reference dates, max delay, etc.); this port has no equivalent class -- `reporting_triangle`
here is always a plain `np.ndarray` -- so `apply_delay()`'s final
`.update_triangle_matrix(reporting_triangle, point_nowcast_matrix)` step (which exists purely to
restore that R class/attributes after `Reduce()` operates on a bare matrix) has no Python
counterpart and is intentionally not ported; there is no class/attribute state to lose in the
first place.

Also NOT ported here (out of scope for NHSN weekly counts, which don't need them): the R
package's ragged/every-other-day reporting structure support, negative-PMF/downward-correction
handling, and several of `.validate_for_delay_estimation()`'s more exotic input-shape checks
(zero-only columns, ragged triangles, NA-in-upper-triangle detection). `estimate_delay`/
`apply_delay` below validate only what's needed for the NHSN use case (see their docstrings).

`iddata.nowcast.triangle` is NOT a port of anything in the R package -- see that module's
docstring for why, and for how it builds the incremental reporting-triangle matrix these
functions expect as input out of repeated `NHSNDataSource.load(as_of=...)` vintage snapshots.

Both functions here operate on an *incremental* reporting triangle (rows = reference weeks,
columns = delay in weeks, cell = the amount added to the cumulative value between successive
delays), matching the RESPINOW/baselinenowcast convention, which is also the right
representation for NHSN hospitalization counts: a value per reference week that gets revised
upward over time.

This module ports only the point-nowcast path (chain-ladder delay estimation + the
`apply_delay` completion formula), not `baselinenowcast`'s negative-binomial uncertainty/
sample-draw layer -- see the project plan for why that's deferred.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# estimate_delay.R
# ---------------------------------------------------------------------------------------------


def estimate_delay(reporting_triangle: np.ndarray, n: int | None = None) -> np.ndarray:
    """
    Estimate the reporting-delay probability mass function from an incremental reporting
    triangle via the chain-ladder method. Ports `baselinenowcast::estimate_delay()`.

    Parameters
    ----------
    reporting_triangle : np.ndarray
        (n_dates, n_delays) matrix of increments between successive vintages of the same
        reference week; NaN marks cells not yet observed (see `triangle.build_increment_triangle`).
    n : int | None
        Number of most recent reference weeks (rows) to use, always starting from the most
        recent. Defaults to the whole triangle (`nrow(reporting_triangle)` in the R source).

    Returns
    -------
    np.ndarray of length n_delays: the estimated probability mass on each delay (0-indexed),
    summing to approximately 1.
    """
    n_dates = reporting_triangle.shape[0]
    if n is None:
        n = n_dates

    # Ports the relevant parts of .validate_for_delay_estimation()'s checks (see module
    # docstring for which of its checks are NOT ported).
    if n < 1:
        raise ValueError("Insufficient `n`, must be greater than or equal to 1.")
    if n_dates < n:
        raise ValueError(f"Reporting triangle has {n_dates} reference times but n = {n} was requested.")

    # Truncate to last n rows
    trunc_triangle = reporting_triangle[n_dates - n:, :]

    # Convert to matrix for chainladder fill -- a no-op here since reporting_triangle is
    # already a plain np.ndarray (see module docstring on why there's no R-style class to strip).
    trunc_matrix = trunc_triangle

    # Fill in missing values in the triangle
    expectation = _chainladder_fill_triangle(trunc_matrix)

    # Calculate probability mass function from filled triangle
    pmf = _calculate_pmf(expectation)

    return pmf


def estimate_delay_pooled(triangles: list[np.ndarray]) -> np.ndarray:
    """
    Fit ONE delay-PMF jointly from multiple independent reporting triangles (e.g. different
    locations, or -- per this project's cross-disease pooling -- flu and COVID sharing the same
    reporting pipeline for one location), each with its own trailing incomplete rows.

    NOT a port of anything in `baselinenowcast`, which only ever fits one series at a time -- this
    project-specific extension exists because naively `np.vstack`-ing several independent
    triangles and calling `estimate_delay` on the result is WRONG: `_chainladder_fill_triangle`
    assumes its input is a single monotonic reporting triangle, where a column's missing cells
    form one contiguous block running to the very last row. That assumption holds for one
    series' own triangle (more recent reference weeks are never MORE complete than older ones),
    but breaks the moment a second triangle's rows are appended below the first, since it finds
    the FIRST missing row across the WHOLE stack and overwrites every row from there to the very
    end of the stack in that column -- silently discarding the second (and any later) triangle's
    genuinely-observed values in the rows straddling that column's missing/present boundary, not
    just the cells that were actually NaN. Confirmed directly: stacking a second, fully-observed
    triangle after a first with a trailing NaN caused `_chainladder_fill_triangle` to replace the
    second triangle's real values (e.g. 5.0, 40.0) with unrelated computed placeholders (27.8,
    22.2) that were never seen in its actual data.

    The fix: fill each triangle independently first (using ONLY its own, correctly-scoped
    chain-ladder ratios -- exactly how `estimate_delay` already handles a single triangle
    correctly), so every input triangle is fully complete (no remaining NaN) before combining.
    Stacking already-filled triangles and summing is then just `_calculate_pmf`'s
    colsum-over-total formula -- no further filling step, and no way for one triangle's
    completeness pattern to bleed into another's.
    """
    filled = [_chainladder_fill_triangle(t) for t in triangles]
    return _calculate_pmf(np.vstack(filled))


def _chainladder_fill_triangle(rep_tri_mat: np.ndarray) -> np.ndarray:
    """
    Fill in missing values in the reporting triangle using the iterative "chainladder" method.

    Ports .chainladder_fill_triangle().
    """
    n_delays = rep_tri_mat.shape[1]
    n_dates = rep_tri_mat.shape[0]
    expectation = rep_tri_mat.copy()

    # Find the column to start filling in
    na_cols = np.where(np.sum(np.isnan(rep_tri_mat), axis=0) > 0)[0]
    start_col = na_cols[0] if len(na_cols) > 0 else None

    # Only fill in reporting triangle if it is incomplete
    if start_col is not None:
        for co in range(start_col, n_delays):
            start_row_candidates = np.where(np.isnan(rep_tri_mat[:, co]))[0]
            if len(start_row_candidates) == 0:
                continue
            start_row = start_row_candidates[0]

            # Extract relevant blocks of the triangle
            block_top_left = _extract_block_top_left(rep_tri_mat, co, n_dates, start_row)
            block_top = _extract_block_top(rep_tri_mat, co, n_dates, start_row)

            # Calculate multiplication factor
            mult_factor = _calculate_mult_factor(block_top, block_top_left)

            # Extract block bottom left
            block_bottom_left = _extract_block_bottom_left(expectation, co, n_dates, start_row)

            # Compute expectations for bottom right
            expectation[start_row:n_dates, co] = _compute_expectations(mult_factor, block_bottom_left)

    return expectation


def _extract_block_top_left(rep_tri: np.ndarray, co: int, n_dates: int, start_row: int) -> np.ndarray:
    """Extract the top left block of the triangle. Ports .extract_block_top_left()."""
    # n_dates is unused, kept only for parity with the R signature
    # (`return(rep_tri[1:(start_row - 1), 1:(co - 1), drop = FALSE])`). numpy slicing never
    # silently drops dimensions the way R matrix indexing can, so no drop=FALSE equivalent
    # is needed here.
    return rep_tri[0:start_row, 0:co]


def _extract_block_top(rep_tri: np.ndarray, co: int, n_dates: int, start_row: int) -> np.ndarray:
    """Extract the top block of the triangle. Ports .extract_block_top()."""
    return rep_tri[0:start_row, co]


def _extract_block_bottom_left(expectation: np.ndarray, co: int, n_dates: int, start_row: int) -> np.ndarray:
    """
    Extract the bottom left block of the triangle. Ports .extract_block_bottom_left().

    Shared by both `_chainladder_fill_triangle` above and `_calc_expectation` below, same as in
    the R source.
    """
    return expectation[start_row:n_dates, 0:co]


def _calculate_mult_factor(block_top: np.ndarray, block_top_left: np.ndarray) -> float:
    """Calculate multiplication factor. Ports .calculate_mult_factor()."""
    return np.sum(block_top) / max(np.sum(block_top_left), 1.0)


def _compute_expectations(mult_factor: float, block_bottom_left: np.ndarray) -> np.ndarray:
    """Compute expectations for the bottom right part. Ports .compute_expectations()."""
    return mult_factor * np.sum(block_bottom_left, axis=1)


def _calculate_pmf(expectation: np.ndarray) -> np.ndarray:
    """Calculate the probability mass function from the filled triangle. Ports .calculate_pmf()."""
    return np.sum(expectation, axis=0) / np.sum(expectation)


# ---------------------------------------------------------------------------------------------
# apply_delay.R
# ---------------------------------------------------------------------------------------------


def apply_delay(reporting_triangle: np.ndarray, delay_pmf: np.ndarray) -> np.ndarray:
    """
    Apply the delay to generate a point nowcast: fill in NaN (not-yet-observed) cells of an
    incremental reporting triangle with point-nowcast estimates, given a fitted delay PMF.
    Ports `baselinenowcast::apply_delay()`.

    Parameters
    ----------
    reporting_triangle : np.ndarray
        (n_dates, n_delays) incremental triangle to complete; must contain at least one NaN.
    delay_pmf : np.ndarray
        Length-n_delays probability mass function, indexed the same as the triangle's columns.

    Returns
    -------
    np.ndarray, same shape as `reporting_triangle`, with NaN cells replaced by point estimates.
    `result.sum(axis=1)` gives the nowcasted final cumulative value per reference week.
    """
    _validate_delay_and_triangle(reporting_triangle, delay_pmf)
    n_delays = len(delay_pmf)
    n_rows = reporting_triangle.shape[0]

    n_row_nas = np.isnan(reporting_triangle).any(axis=1).sum()
    if n_row_nas == 0:
        raise ValueError(
            "`reporting_triangle` doesn't contain any missing values, there is nothing to nowcast."
        )

    # Precompute CDFs for the delay PMF
    delay_cdf = np.cumsum(delay_pmf)

    # Convert to plain matrix for efficiency in Reduce iterations -- a no-op here since
    # reporting_triangle is already a plain np.ndarray.
    init_matrix = reporting_triangle.copy()

    # Iterates through each column (delay) and adds entries to the reporting matrix to nowcast.
    # R loops `2:n_delays` (1-indexed, via Reduce); the 0-indexed equivalent is
    # `range(1, n_delays)`. `index` here is 0-indexed and corresponds to R's `index - 1`.
    point_nowcast_matrix = init_matrix
    for index in range(1, n_delays):
        point_nowcast_matrix = _calc_expectation(
            index,
            point_nowcast_matrix,
            delay_pmf[index],
            delay_cdf[index - 1],
            n_rows,
        )

    # Preserve reporting_triangle class and attributes -- N/A here; see module docstring.

    return point_nowcast_matrix


def _validate_delay_and_triangle(triangle: np.ndarray, delay_pmf: np.ndarray) -> None:
    """
    Various checks to make sure that the reporting triangle and the delay PMF passed in to
    `apply_delay()` are formatted properly and compatible. Ports .validate_delay_and_triangle().

    Not ported: R's `checkmate`-based type assertions (triangle must inherit from class
    "matrix", delay_pmf must inherit from class "numeric", triangle must not be all-missing) --
    these don't have a meaningful Python/numpy equivalent (a numpy array is always
    "matrix-like"), and empty/malformed inputs already fail one of the checks below instead.
    """
    if triangle.shape[1] != len(delay_pmf):
        raise ValueError(
            "Length of the delay PMF is not the same as the number of delays in the triangle "
            "to be nowcasted."
        )

    if np.isnan(triangle[triangle.shape[0] - 1, 1]) and delay_pmf[0] == 0:
        raise ValueError(
            "Value of delay PMF at delay = 0 is 0, and the latest reference time in the "
            "reporting matrix only contains a value at delay = 0. There is insufficient "
            "information to generate a point nowcast for the latest reference time."
        )

    if delay_pmf[0] < 0:
        raise ValueError(f"First entry of delay PMF (delay = 0) is negative ({delay_pmf[0]}).")


def _calc_expectation(
    index: int,
    expectation: np.ndarray,
    delay_prob: float,
    delay_cdf_prev: float,
    n_rows: int,
) -> np.ndarray:
    """Calculate the updated rows of the expected nowcasted triangle. Ports .calc_expectation()."""
    # Find rows with NA in this column that need to be filled
    na_rows = _where_is_na_in_col(expectation, index)
    if len(na_rows) == 0:
        return expectation

    # Start with the first row that has NA
    row_start = na_rows[0]

    # Extract the left block for these rows
    block_bottom_left = _extract_block_bottom_left(expectation, index, n_rows, row_start)

    # Calculate row sums for the extracted block
    x = np.sum(block_bottom_left, axis=1)

    # Calculate expectations with support for zero values
    exp_n = _calc_modified_expectation(x, delay_cdf_prev)

    # Update only the NA rows in the column
    expectation[row_start:n_rows, index] = exp_n * delay_prob

    return expectation


def _where_is_na_in_col(expectation: np.ndarray, co: int) -> np.ndarray:
    """Ports .where_is_na_in_col()."""
    return np.where(np.isnan(expectation[:, co]))[0]


def _calc_modified_expectation(x: np.ndarray, delay_cdf_prev: float) -> np.ndarray:
    """
    Ports .calc_modified_expectation().

    This "+1-cdf" adjustment is a Bayesian-derived smoothing term (assuming binomial
    subsampling) that specifically tames the near-zero-denominator blowup at small delays,
    rather than naive division by the raw completion proportion.
    """
    return (x + 1 - delay_cdf_prev) / delay_cdf_prev
