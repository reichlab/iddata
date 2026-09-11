"""Unit tests for iddata.nowcast.nhsn.NHSNNowcaster: max_delay resolution (per-location override
vs. disease-level default) and correct()'s per-group dispatch. Uses monkeypatched constants and a
stubbed VintageCache so no real network access is needed -- the real-data behavior is covered by
tests/iddata/integration/test_nowcast_nhsn.py and scripts/backtest_nhsn_nowcast.py."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from iddata.enums import Disease
from iddata.nowcast import nhsn
from iddata.nowcast.base import NowcastConfig
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

        def _fake_correct_group(self, result, vintages, location, agg_level, as_of, max_delay, training_window):
            recorded_calls.append((location, agg_level, max_delay, training_window))

        monkeypatch.setattr(NHSNNowcaster, "_correct_group", _fake_correct_group)

        nowcaster = NHSNNowcaster(NowcastConfig(min_vintages=1))
        as_of = datetime.date(2026, 1, 3)
        with pytest.warns(UserWarning, match="experimental"):
            nowcaster.correct(self._make_latest_df(), as_of, _make_source())

        by_location = {(loc, agg): (max_delay, tw) for loc, agg, max_delay, tw in recorded_calls}
        assert by_location[("34", "state")][0] == 2  # per-location override
        assert by_location[("36", "state")][0] == 5  # disease-level default
        # Both groups share the same training_window, computed once from the WIDE max_delay (5).
        assert by_location[("34", "state")][1] == by_location[("36", "state")][1]

        # The vintage fetch window must be sized off the wide max_delay (5), not the smaller
        # per-location override (2) -- otherwise the "36" group would be starved of vintages.
        training_window = by_location[("34", "state")][1]
        assert len(captured_as_of_dates["value"]) == training_window + 5
