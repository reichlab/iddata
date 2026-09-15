"""Unit tests for iddata.nowcast.nhsn.NHSNNowcaster: max_delay resolution (per-location override
vs. disease-level default) and correct()'s per-group dispatch. Uses monkeypatched constants and a
stubbed VintageCache so no real network access is needed -- the real-data behavior is covered by
tests/iddata/integration/test_nowcast_nhsn.py and scripts/backtest_nhsn_nowcast.py."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from iddata.enums import Disease
from iddata.nowcast import nhsn
from iddata.nowcast.base import NowcastConfig
from iddata.nowcast.delay_model import estimate_delay, estimate_delay_pooled
from iddata.nowcast.nhsn import NHSNNowcaster
from iddata.nowcast.vintage_cache import VintageCache


def _make_source(disease=Disease.FLU):
    source = MagicMock()
    source.disease = disease
    return source


class TestResolveMaxDelay:
    def test_falls_back_to_disease_default_when_no_per_location_override(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {})
        nowcaster = NHSNNowcaster(NowcastConfig())

        assert nowcaster._resolve_max_delay(_make_source(), "34", "state") == 5


    def test_uses_per_location_override_when_present(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {Disease.FLU: {("34", "state"): 2}})
        nowcaster = NHSNNowcaster(NowcastConfig())

        assert nowcaster._resolve_max_delay(_make_source(), "34", "state") == 2
        # A different location with no override still falls back to the disease default.
        assert nowcaster._resolve_max_delay(_make_source(), "36", "state") == 5


    def test_per_location_override_is_keyed_on_agg_level_too(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {Disease.FLU: {("34", "state"): 2}})
        nowcaster = NHSNNowcaster(NowcastConfig())

        # Same location string, different agg_level -- must not match the "state" override.
        assert nowcaster._resolve_max_delay(_make_source(), "34", "hsa") == 5


    def test_explicit_config_override_bypasses_per_location_lookup(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {Disease.FLU: {("34", "state"): 2}})
        nowcaster = NHSNNowcaster(NowcastConfig(max_delay_weeks=3))

        assert nowcaster._resolve_max_delay(_make_source(), "34", "state") == 3


    def test_raises_for_disease_with_no_static_default(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {})
        nowcaster = NHSNNowcaster(NowcastConfig())

        with pytest.raises(ValueError, match="No static NHSN_MAX_DELAY_WEEKS default"):
            nowcaster._resolve_max_delay(_make_source(disease=Disease.RSV), "34", "state")


class TestCorrectDispatchesPerLocationMaxDelay:
    def _make_latest_df(self):
        return pd.DataFrame({
            "source": ["nhsn"] * 2, "agg_level": ["state", "state"], "location": ["34", "36"],
            "season": ["2025/26"] * 2, "season_week": [10, 10],
            "wk_end_date": [pd.Timestamp("2026-01-03")] * 2, "inc": [5.0, 6.0],
        })


    def test_wide_fetch_window_covers_the_largest_per_group_max_delay(self, monkeypatch):
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS", {Disease.FLU: 5})
        monkeypatch.setattr(nhsn, "NHSN_MAX_DELAY_WEEKS_BY_LOCATION", {Disease.FLU: {("34", "state"): 2}})
        monkeypatch.setattr(VintageCache, "n_distinct_vintages", lambda self, as_of_dates: len(as_of_dates))

        captured_as_of_dates = {}

        def _fake_get_many(self, source, as_of_dates):
            captured_as_of_dates["value"] = as_of_dates
            return {}

        monkeypatch.setattr(VintageCache, "get_many", _fake_get_many)

        recorded_calls = []

        def _fake_correct_group(self, result, wide_matrix, wide_ref_dates, location, agg_level, max_delay, pooled_pmf):
            recorded_calls.append((location, agg_level, max_delay, wide_matrix.shape[1]))

        monkeypatch.setattr(NHSNNowcaster, "_correct_group", _fake_correct_group)

        nowcaster = NHSNNowcaster(NowcastConfig(min_vintages=1))
        as_of = datetime.date(2026, 1, 3)
        with pytest.warns(UserWarning, match="experimental"):
            nowcaster.correct(self._make_latest_df(), as_of, _make_source())

        by_location = {(loc, agg): (max_delay, wide_cols) for loc, agg, max_delay, wide_cols in recorded_calls}
        assert by_location[("34", "state")][0] == 2  # per-location override
        assert by_location[("36", "state")][0] == 5  # disease-level default
        # Both groups receive the SAME wide-width matrix (built once from the WIDE max_delay, 5),
        # even though "34" will slice it down to its own smaller max_delay internally.
        assert by_location[("34", "state")][1] == by_location[("36", "state")][1] == 6  # wide_max_delay(5) + 1

        # The vintage fetch window must be sized off the wide max_delay (5), not the smaller
        # per-location override (2) -- otherwise the "36" group would be starved of vintages.
        # NowcastConfig(min_vintages=1) with no explicit training_window_weeks resolves to
        # max(3 * wide_max_delay(5), _MIN_DEFAULT_TRAINING_WINDOW_WEEKS(12)) == 15, unaffected by
        # the NHSN_SOURCE_CUTOVER_DATE cap since as_of=2026-01-03 is far past it.
        training_window = 15
        assert len(captured_as_of_dates["value"]) == training_window + 5


class TestFitPooledPmf:
    def test_pools_clean_groups_matching_direct_estimate_delay_pooled_computation(self):
        nowcaster = NHSNNowcaster(NowcastConfig())
        m1 = np.array([[10.0, 2.0], [8.0, 3.0], [12.0, 4.0]])
        m2 = np.array([[5.0, 1.0], [6.0, 2.0], [7.0, 1.0]])
        wide_triangles = {("A", "state"): (m1, []), ("B", "state"): (m2, [])}

        result = nowcaster._fit_pooled_pmf(wide_triangles)

        expected = estimate_delay_pooled([m1, m2])
        np.testing.assert_allclose(result, expected)


    def test_pools_multiple_groups_each_with_their_own_trailing_incomplete_row(self):
        """
        Regression test for a real bug: naively `np.vstack`-ing several groups' triangles (each
        with its own trailing incomplete row, the realistic/common shape -- see
        estimate_delay_pooled's docstring) and calling `estimate_delay` directly on the stack
        corrupts every group AFTER the first, silently overwriting its genuinely-observed rows
        with unrelated computed placeholders. _fit_pooled_pmf must fill each group independently
        (via estimate_delay_pooled) before combining, so a second group's real observed values
        are never discarded just because an earlier group also has a trailing NaN.
        """
        nowcaster = NHSNNowcaster(NowcastConfig())
        # group_b's ratios are deliberately very different from group_a's, so if group_b's real
        # values got overwritten with group_a-derived placeholders, the pooled result would be
        # detectably wrong (this is the exact scenario that reproduced the bug).
        group_a = np.array([[10.0, 2.0], [8.0, 3.0], [5.0, np.nan]])
        group_b = np.array([[100.0, 5.0], [80.0, 40.0], [50.0, np.nan]])
        wide_triangles = {("A", "state"): (group_a, []), ("B", "state"): (group_b, [])}

        result = nowcaster._fit_pooled_pmf(wide_triangles)

        expected = estimate_delay_pooled([group_a, group_b])
        np.testing.assert_allclose(result, expected)
        naive_and_wrong = estimate_delay(np.vstack([group_a, group_b]))
        assert not np.allclose(result, naive_and_wrong), (
            "pooled result matches the naive (buggy) np.vstack + estimate_delay computation -- "
            "the fix isn't actually being used"
        )


    def test_skips_group_with_incomplete_oldest_row(self):
        nowcaster = NHSNNowcaster(NowcastConfig())
        clean = np.array([[10.0, 2.0], [8.0, 3.0], [12.0, 4.0]])
        # Oldest row (index 0) has NaN -- insufficient history for this group's own chain-ladder
        # invariant, so it must be excluded from the pool entirely.
        incomplete = np.array([[np.nan, np.nan], [6.0, 2.0], [7.0, 1.0]])
        wide_triangles = {("A", "state"): (clean, []), ("B", "state"): (incomplete, [])}

        result = nowcaster._fit_pooled_pmf(wide_triangles)

        expected = estimate_delay(clean)  # only the clean group contributes
        np.testing.assert_allclose(result, expected)


    def test_trims_all_nan_trailing_row_before_pooling(self):
        nowcaster = NHSNNowcaster(NowcastConfig())
        # Trailing row (the current, not-yet-reported reference week) has zero observations at
        # every delay -- must be dropped before fitting, same as _correct_group does, or it
        # would corrupt/crash estimate_delay the same way the original single-group bug did.
        m1 = np.array([[10.0, 2.0], [8.0, 3.0], [np.nan, np.nan]])
        wide_triangles = {("A", "state"): (m1, [])}

        result = nowcaster._fit_pooled_pmf(wide_triangles)

        expected = estimate_delay(m1[:2])
        np.testing.assert_allclose(result, expected)


    def test_returns_none_when_no_groups_poolable(self):
        nowcaster = NHSNNowcaster(NowcastConfig())
        incomplete = np.array([[np.nan, np.nan], [6.0, 2.0]])
        wide_triangles = {("A", "state"): (incomplete, [])}

        assert nowcaster._fit_pooled_pmf(wide_triangles) is None


class TestShrinkTowardPooled:
    def test_returns_own_pmf_unchanged_when_no_pooled_pmf(self):
        nowcaster = NHSNNowcaster(NowcastConfig())
        own_pmf = np.array([0.7, 0.3])
        matrix = np.array([[10.0, 2.0], [8.0, 3.0]])

        result = nowcaster._shrink_toward_pooled(own_pmf, matrix, pooled_pmf=None)

        np.testing.assert_array_equal(result, own_pmf)


    def test_weight_is_one_half_when_volume_equals_shrinkage_k(self):
        nowcaster = NHSNNowcaster(NowcastConfig(pmf_shrinkage_k=20.0))
        own_pmf = np.array([0.8, 0.2])
        pooled_pmf = np.array([0.4, 0.6])
        matrix = np.array([[10.0, 2.0], [8.0, 0.0]])  # nansum == 20 == pmf_shrinkage_k -> weight=0.5

        result = nowcaster._shrink_toward_pooled(own_pmf, matrix, pooled_pmf)

        np.testing.assert_allclose(result, 0.5 * own_pmf + 0.5 * pooled_pmf)


    def test_weight_approaches_one_for_large_volume(self):
        nowcaster = NHSNNowcaster(NowcastConfig(pmf_shrinkage_k=10.0))
        own_pmf = np.array([0.8, 0.2])
        pooled_pmf = np.array([0.4, 0.6])
        matrix = np.array([[1_000_000.0, 0.0]])

        result = nowcaster._shrink_toward_pooled(own_pmf, matrix, pooled_pmf)

        np.testing.assert_allclose(result, own_pmf, atol=1e-4)


    def test_weight_approaches_zero_for_tiny_volume(self):
        nowcaster = NHSNNowcaster(NowcastConfig(pmf_shrinkage_k=1_000_000.0))
        own_pmf = np.array([0.8, 0.2])
        pooled_pmf = np.array([0.4, 0.6])
        matrix = np.array([[1.0, 0.0]])

        result = nowcaster._shrink_toward_pooled(own_pmf, matrix, pooled_pmf)

        np.testing.assert_allclose(result, pooled_pmf, atol=1e-4)


    def test_truncates_and_renormalizes_pooled_pmf_to_own_width(self):
        nowcaster = NHSNNowcaster(NowcastConfig(pmf_shrinkage_k=10.0))
        own_pmf = np.array([0.9, 0.1])  # this group's own (narrower) max_delay
        pooled_pmf = np.array([0.5, 0.3, 0.2])  # pooled fit at the wider max_delay
        matrix = np.array([[10.0, 0.0]])  # nansum == 10 == pmf_shrinkage_k -> weight=0.5

        result = nowcaster._shrink_toward_pooled(own_pmf, matrix, pooled_pmf)

        truncated_renormalized = np.array([0.5, 0.3]) / 0.8
        np.testing.assert_allclose(result, 0.5 * own_pmf + 0.5 * truncated_renormalized)
