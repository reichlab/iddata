"""NHSNNowcaster: orchestrates vintage fetching and the chain-ladder point-nowcast formula
against NHSNDataSource."""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd

from iddata.constants import NHSN_MAX_DELAY_WEEKS, NHSN_MAX_DELAY_WEEKS_BY_LOCATION, NHSN_SOURCE_CUTOVER_DATE
from iddata.enums import SourceType
from iddata.nowcast.base import Nowcaster, register_nowcaster
from iddata.nowcast.delay_model import apply_delay, estimate_delay, estimate_delay_pooled
from iddata.nowcast.triangle import build_increment_triangle, weekly_as_of_dates
from iddata.nowcast.vintage_cache import VintageCache
from iddata.sources.base import DataSource

# baselinenowcast's own "V=3xD reference times for total training volume" convention assumes D
# is measured in days, where D is typically large enough (e.g. ~25 days in its own examples)
# that 3xD is already a reasonable sample size. Our max_delay is in WEEKS and, per the
# empirically-calibrated NHSN_MAX_DELAY_WEEKS, can be as small as 1 -- a bare "3x" multiplier
# would then default training_window to just 3 weeks, which is *below* NowcastConfig's own
# default min_vintages (10), silently making every default-configured NowcastConfig() skip
# correction entirely. This floor keeps the default training window large enough regardless of
# how small max_delay turns out to be.
_MIN_DEFAULT_TRAINING_WINDOW_WEEKS = 12


def cap_training_window_to_cutover(as_of: datetime.date, max_delay: int, training_window: int) -> int:
    """
    Shrink training_window so that the full vintage-fetch window (training_window + max_delay
    weeks back from as_of) never reaches before NHSN_SOURCE_CUTOVER_DATE. The legacy HHS
    archive and the current NHSN source have non-overlapping week coverage, so a vintage window
    straddling the cutover would silently mix an incompatible data regime into the reporting
    triangle -- a legacy vintage that doesn't cover a requested week returns NaN,
    indistinguishable from "not yet reported", corrupting the whole delay-PMF fit rather than
    just the genuinely-incomplete trailing rows. Shared by `NHSNNowcaster.correct()` and
    `scripts/estimate_nhsn_max_delay.py`, which can otherwise request an even wider window.
    """
    max_total_weeks = max(1, (as_of - NHSN_SOURCE_CUTOVER_DATE).days // 7 + 1)
    total_weeks = min(training_window + max_delay, max_total_weeks)
    return max(1, total_weeks - max_delay)


class NHSNNowcaster(Nowcaster):
    source_name = SourceType.NHSN


    def __init__(self, config):
        super().__init__(config)
        self._cache = VintageCache()


    def correct(self, latest_df: pd.DataFrame, as_of: datetime.date, source: DataSource) -> pd.DataFrame:
        # Resolve each group's own max_delay up front (NHSN_MAX_DELAY_WEEKS_BY_LOCATION overrides
        # NHSN_MAX_DELAY_WEEKS[disease] for specific locations whose own fitted delay
        # distribution converges at a meaningfully different delay -- see that constant's
        # comment). The vintage-fetch window below has to be wide enough for the largest of
        # these, even though most groups' own triangle only uses a narrower slice of it.
        groups = list(latest_df.groupby(["location", "agg_level"]).groups.keys())
        max_delays = {
            (location, agg_level): self._resolve_max_delay(source, location, agg_level)
            for location, agg_level in groups
        }
        wide_max_delay = max(max_delays.values(), default=self._resolve_max_delay(source, None, None))

        training_window = self.config.training_window_weeks or max(
            3 * wide_max_delay, _MIN_DEFAULT_TRAINING_WINDOW_WEEKS
        )
        training_window = cap_training_window_to_cutover(as_of, wide_max_delay, training_window)
        as_of_dates = weekly_as_of_dates(as_of, training_window + wide_max_delay)

        vintages = self._cache.get_many(source, as_of_dates)
        n_distinct = self._cache.n_distinct_vintages(as_of_dates)
        if n_distinct < self.config.min_vintages:
            msg = (
                f"Only {n_distinct} distinct NHSN vintages available for as_of={as_of} "
                f"(need >= {self.config.min_vintages}); skipping nowcast correction."
            )
            if self.config.on_insufficient_data == "raise":
                raise ValueError(msg)
            warnings.warn(msg)
            return latest_df.copy()

        warnings.warn(
            "NHSN nowcast correction is experimental and has not been shown to reliably reduce "
            "error in real-data backtesting -- see iddata.constants.NHSN_MAX_DELAY_WEEKS's "
            "comment and the project plan's 'Empirical Validation Results' section for known "
            "limitations (a single delay distribution doesn't transfer across locations, and "
            "completion isn't stable enough over time to trust a fixed historical curve) before "
            "relying on this in production.",
            UserWarning,
        )

        # Build every group's own triangle at the WIDE width up front. This is needed both to
        # correct that group (sliced down to its own max_delay below) and to fit the pooled PMF
        # that group's own fit gets shrunk toward -- see _shrink_toward_pooled's docstring for
        # why a pure per-location fit was tried and found to make things worse, not better.
        wide_triangles = {}
        for location, agg_level in max_delays:
            vintage_series = {}
            for v, df in vintages.items():
                sub = df[(df["location"] == location) & (df["agg_level"] == agg_level)]
                if not sub.empty:
                    vintage_series[v] = sub.set_index("wk_end_date")["inc"]
            wide_triangles[(location, agg_level)] = build_increment_triangle(
                vintage_series, as_of, wide_max_delay, training_window
            )

        pooled_pmf = self._fit_pooled_pmf(wide_triangles)

        result = latest_df.copy()
        for (location, agg_level), max_delay in max_delays.items():
            wide_matrix, ref_dates = wide_triangles[(location, agg_level)]
            self._correct_group(result, wide_matrix, ref_dates, location, agg_level, max_delay, pooled_pmf)

        return result


    def _resolve_max_delay(self, source: DataSource, location: str | None, agg_level: str | None) -> int:
        if self.config.max_delay_weeks is not None:
            return self.config.max_delay_weeks
        disease = getattr(source, "disease", None)
        if disease not in NHSN_MAX_DELAY_WEEKS:
            raise ValueError(
                f"No static NHSN_MAX_DELAY_WEEKS default for disease={disease!r}; "
                "pass NowcastConfig(max_delay_weeks=...) explicitly."
            )
        per_location = NHSN_MAX_DELAY_WEEKS_BY_LOCATION.get(disease, {})
        return per_location.get((location, agg_level), NHSN_MAX_DELAY_WEEKS[disease])


    def _fit_pooled_pmf(self, wide_triangles: dict[tuple[str, str], tuple[np.ndarray, list]]) -> np.ndarray | None:
        """Fit one delay-PMF pooled across every (location, agg_level) group's own WIDE-width
        triangle, for _shrink_toward_pooled to blend individual groups' own fits toward. Mirrors
        scripts/estimate_nhsn_max_delay.py's pooling, but computed live from whatever vintages
        this correct() call already fetched, rather than a separate offline analysis."""
        poolable = []
        for matrix, _ in wide_triangles.values():
            # Same two guards as _correct_group: skip groups with insufficient history for the
            # oldest row (this pooled fit's own chain-ladder invariant), and trim/skip rows or
            # groups with zero observations at every delay (e.g. the current, not-yet-reported
            # reference week) -- otherwise those NaN/all-zero rows corrupt estimate_delay the
            # same way they would in a single-group fit.
            if np.isnan(matrix[0, :]).any():
                continue
            has_any_observation = ~np.isnan(matrix).all(axis=1)
            trimmed = matrix[has_any_observation]
            if trimmed.shape[0] == 0 or np.nansum(trimmed) == 0:
                continue
            poolable.append(trimmed)
        if not poolable:
            return None
        return estimate_delay_pooled(poolable)


    def _shrink_toward_pooled(self, own_pmf: np.ndarray, matrix: np.ndarray, pooled_pmf: np.ndarray | None) -> np.ndarray:
        """
        Blend a group's own fitted delay-PMF with the pooled PMF, weighted by that group's own
        data volume: `w = n / (n + pmf_shrinkage_k)`, `n = total case count in its own triangle`.

        A prior v2 attempt (`NHSN_MAX_DELAY_WEEKS_BY_LOCATION`, see its comment in
        `iddata.constants`) fit each location's delay distribution purely from its own ~17-week
        training window and found this made backtest error more than double, including for New
        Jersey -- a location whose own max_delay happened to match independent ground truth.
        Fitting a chain-ladder ratio estimator from one location's own limited history is
        apparently noisy enough that the estimation error costs more accuracy than correctly
        capturing genuine per-location structure gains back. Shrinking toward the pooled fit
        (which has far more effective observations, since `estimate_delay`'s ratios are sums
        across every pooled group's reference weeks) is the standard fix for exactly this
        small-sample chain-ladder problem.
        """
        if pooled_pmf is None:
            return own_pmf
        pooled_pmf = pooled_pmf[: len(own_pmf)]
        pooled_pmf = pooled_pmf / pooled_pmf.sum()
        n = np.nansum(matrix)
        weight = n / (n + self.config.pmf_shrinkage_k)
        return weight * own_pmf + (1 - weight) * pooled_pmf


    def _correct_group(
        self,
        result: pd.DataFrame,
        wide_matrix: np.ndarray,
        wide_ref_dates: list[datetime.date],
        location: str,
        agg_level: str,
        max_delay: int,
        pooled_pmf: np.ndarray | None,
    ) -> None:
        # wide_matrix's increments are computed column-by-column left to right (see
        # build_increment_triangle/_increments_from_cumulative), so truncating to this group's
        # own (possibly narrower) max_delay here gives exactly the same values as building the
        # triangle directly at that width.
        matrix = wide_matrix[:, : max_delay + 1]
        ref_dates = wide_ref_dates

        # Reference weeks with zero observations at any delay (e.g. the most recent week,
        # which -- given NHSN's structural ~1-week minimum publication lag -- no vintage has
        # been taken far enough past to report on at all yet) can't be nowcasted from nothing,
        # and `result` doesn't even contain a row for them anyway (NHSNDataSource.load() itself
        # hasn't reported that week yet either). Drop them before fitting so they don't trip
        # apply_delay's "insufficient information" check.
        has_any_observation = ~np.isnan(matrix).all(axis=1)
        matrix = matrix[has_any_observation]
        ref_dates = [rd for rd, keep in zip(ref_dates, has_any_observation) if keep]
        if matrix.shape[0] == 0:
            return

        incomplete_mask = np.isnan(matrix).any(axis=1)
        if not incomplete_mask.any():
            return

        # Small territories (e.g. American Samoa, Guam, Northern Mariana Islands, US Virgin
        # Islands) can report all-zero counts for their entire triangle. There's no delay
        # pattern to estimate from all-zero data, and estimate_delay's colSum/sum division
        # would otherwise be a literal 0/0 -- leave these groups uncorrected instead.
        if np.nansum(matrix) == 0:
            return

        own_pmf = estimate_delay(matrix)
        pmf = self._shrink_toward_pooled(own_pmf, matrix, pooled_pmf)
        filled = apply_delay(matrix, pmf)
        nowcasted_totals = filled.sum(axis=1)

        for i, ref_date in enumerate(ref_dates):
            if not incomplete_mask[i]:
                continue
            row_mask = (
                (result["location"] == location)
                & (result["agg_level"] == agg_level)
                & (result["wk_end_date"] == pd.Timestamp(ref_date))
            )
            result.loc[row_mask, "inc"] = nowcasted_totals[i]


register_nowcaster(SourceType.NHSN, NHSNNowcaster)
