"""Unit tests for iddata.nowcast.base: NowcastConfig, Nowcaster, NowcastedDataSource, wrap_sources()."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from iddata.enums import SourceType
from iddata.loader import DiseaseDataLoader
from iddata.nowcast import base
from iddata.nowcast.base import NowcastConfig, NowcastedDataSource, Nowcaster, wrap_sources


class _StubNowcaster(Nowcaster):
    """A minimal Nowcaster used to test wrap_sources()/NowcastedDataSource wiring without
    needing the real (NHSN-specific) vintage-fetching machinery."""

    source_name = SourceType.FLUSURVNET  # arbitrary SourceType, monkeypatched into the registry per-test


    def correct(self, latest_df, as_of, source):
        result = latest_df.copy()
        result["inc"] = result["inc"] * 100  # obviously-different marker value
        return result


class TestNowcastConfig:
    def test_defaults(self):
        config = NowcastConfig()
        assert config.max_delay_weeks is None
        assert config.training_window_weeks is None
        assert config.min_vintages == 10
        assert config.on_insufficient_data == "passthrough"
        assert config.sources == (SourceType.NHSN,)


    def test_rejects_max_delay_weeks_below_one(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            NowcastConfig(max_delay_weeks=0)


    def test_rejects_invalid_on_insufficient_data(self):
        with pytest.raises(ValueError, match="'passthrough' or 'raise'"):
            NowcastConfig(on_insufficient_data="ignore")


class TestNowcastedDataSource:
    def test_source_name_proxies_wrapped_source(self):
        src = MagicMock()
        src.source_name = SourceType.NHSN
        wrapped = NowcastedDataSource(src, _StubNowcaster(NowcastConfig()))
        assert wrapped.source_name == SourceType.NHSN


    def test_load_calls_nowcaster_correct_with_the_loaded_frame(self):
        src = MagicMock()
        src.source_name = SourceType.NHSN
        src.load.return_value = pd.DataFrame({"inc": [1.0, 2.0]})
        nowcaster = _StubNowcaster(NowcastConfig())
        wrapped = NowcastedDataSource(src, nowcaster)

        result = wrapped.load(as_of=datetime.date(2026, 1, 1))

        src.load.assert_called_once_with(as_of=datetime.date(2026, 1, 1))
        assert list(result["inc"]) == [100.0, 200.0]  # _StubNowcaster's marker transform applied


class TestWrapSources:
    def test_wraps_sources_with_a_registered_nowcaster_and_in_config_sources(self, monkeypatch):
        monkeypatch.setitem(base._NOWCASTER_REGISTRY, SourceType.FLUSURVNET, _StubNowcaster)
        src = MagicMock()
        src.source_name = SourceType.FLUSURVNET
        config = NowcastConfig(sources=(SourceType.FLUSURVNET,))

        result = wrap_sources([src], config)

        assert len(result) == 1
        assert isinstance(result[0], NowcastedDataSource)


    def test_does_not_wrap_sources_not_in_config_sources(self, monkeypatch):
        monkeypatch.setitem(base._NOWCASTER_REGISTRY, SourceType.FLUSURVNET, _StubNowcaster)
        src = MagicMock()
        src.source_name = SourceType.FLUSURVNET
        config = NowcastConfig(sources=(SourceType.NHSN,))  # FLUSURVNET not requested

        result = wrap_sources([src], config)

        assert result == [src]


    def test_does_not_wrap_sources_with_no_registered_nowcaster(self, monkeypatch):
        monkeypatch.delitem(base._NOWCASTER_REGISTRY, SourceType.ILINET, raising=False)
        src = MagicMock()
        src.source_name = SourceType.ILINET  # never registered
        config = NowcastConfig(sources=(SourceType.NHSN, SourceType.ILINET))

        with pytest.warns(UserWarning, match="no Nowcaster is registered"):
            result = wrap_sources([src], config)

        assert result == [src]


    def test_warns_when_config_requests_an_unregistered_source_type(self, monkeypatch):
        # Guards against the silent-no-op footgun: requesting nowcasting for a SourceType with
        # no registered Nowcaster (e.g. NSSP today, before a v2 NSSPNowcaster exists) should be
        # loud, not silently do nothing.
        monkeypatch.delitem(base._NOWCASTER_REGISTRY, SourceType.NSSP, raising=False)
        with pytest.warns(UserWarning, match=r"requested nowcasting for \['nssp'\]"):
            wrap_sources([], NowcastConfig(sources=(SourceType.NSSP,)))


    def test_no_warning_when_all_requested_sources_are_registered(self, recwarn):
        # NHSN is registered via the real package import (not a monkeypatched stub).
        wrap_sources([], NowcastConfig(sources=(SourceType.NHSN,)))
        assert len(recwarn) == 0


class TestDiseaseDataLoaderNowcastWiring:
    """Mirrors the existing MagicMock-based DiseaseDataLoader tests in test_sources.py."""

    def _make_source_df(self, source_value: str):
        return pd.DataFrame({
            "source": [source_value], "agg_level": ["state"], "location": ["01"],
            "season": ["2023/24"], "season_week": [15],
            "wk_end_date": [pd.Timestamp("2024-01-06")], "inc": [0.5],
        })


    def test_nowcast_none_is_byte_identical_to_omitting_the_argument(self):
        # Regression guard: the default nowcast=None must not change any existing behavior.
        src = MagicMock()
        src.source_name = SourceType.NHSN
        src.load.return_value = self._make_source_df("nhsn")
        loader = DiseaseDataLoader()
        as_of = datetime.date(2024, 1, 6)

        df_omitted = loader.load(sources=[src], as_of=as_of)
        df_explicit_none = loader.load(sources=[src], as_of=as_of, nowcast=None)

        pd.testing.assert_frame_equal(df_omitted, df_explicit_none)


    def test_nowcast_config_wraps_registered_sources_before_loading(self, monkeypatch):
        monkeypatch.setitem(base._NOWCASTER_REGISTRY, SourceType.FLUSURVNET, _StubNowcaster)
        src = MagicMock()
        src.source_name = SourceType.FLUSURVNET
        src.load.return_value = self._make_source_df("flusurvnet")
        loader = DiseaseDataLoader()

        df = loader.load(
            sources=[src], as_of=datetime.date(2024, 1, 6),
            nowcast=NowcastConfig(sources=(SourceType.FLUSURVNET,)),
        )

        # _StubNowcaster.correct() multiplies inc by 100 -- confirms wrap_sources really ran
        assert df["inc"].iloc[0] == pytest.approx(50.0)


    def test_nowcast_config_leaves_unregistered_sources_untouched(self, monkeypatch):
        monkeypatch.delitem(base._NOWCASTER_REGISTRY, SourceType.ILINET, raising=False)
        src = MagicMock()
        src.source_name = SourceType.ILINET
        src.load.return_value = self._make_source_df("ilinet")
        loader = DiseaseDataLoader()

        with pytest.warns(UserWarning, match=r"requested nowcasting for .*'ilinet'"):
            df = loader.load(
                sources=[src], as_of=datetime.date(2024, 1, 6),
                nowcast=NowcastConfig(sources=(SourceType.NHSN, SourceType.ILINET)),
            )

        assert df["inc"].iloc[0] == pytest.approx(0.5)  # unchanged, no nowcaster registered for ILINET
