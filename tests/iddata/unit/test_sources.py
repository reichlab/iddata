"""Unit tests for iddata DataSource classes (constructor, source_name, validation)."""

import datetime
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from iddata.enums import Disease, SourceType
from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource


class TestNHSNDataSource:
    def test_source_name(self):
        assert NHSNDataSource.source_name == SourceType.NHSN


    def test_default_disease(self):
        src = NHSNDataSource()
        assert src.disease == Disease.FLU


    def test_custom_disease(self):
        src = NHSNDataSource(disease=Disease.COVID)
        assert src.disease == Disease.COVID


    def test_raises_if_as_of_none(self):
        src = NHSNDataSource()
        with pytest.raises((ValueError, TypeError)):
            src.load(as_of=None)


class TestNSSPDataSource:
    def test_source_name(self):
        assert NSSPDataSource.source_name == SourceType.NSSP


    def test_default_disease(self):
        assert NSSPDataSource().disease == Disease.FLU


    def test_custom_disease(self):
        src = NSSPDataSource(disease=Disease.COVID)
        assert src.disease == Disease.COVID


    def test_raises_if_as_of_none(self):
        src = NSSPDataSource()
        with pytest.raises((ValueError, TypeError)):
            src.load(as_of=None)


class TestILINetDataSource:
    def test_source_name(self):
        assert ILINetDataSource.source_name == SourceType.ILINET


    def test_default_scale_to_positive(self):
        assert ILINetDataSource().scale_to_positive is True


    def test_custom_scale(self):
        src = ILINetDataSource(scale_to_positive=False)
        assert src.scale_to_positive is False


class TestFluSurvNetDataSource:
    def test_source_name(self):
        assert FluSurvNetDataSource.source_name == SourceType.FLUSURVNET


    def test_default_burden_adj(self):
        assert FluSurvNetDataSource().burden_adj is True


    def test_custom_locations(self):
        locs = ["California", "Colorado"]
        src = FluSurvNetDataSource(locations=locs)
        assert src.locations == locs


    def test_default_locations_not_empty(self):
        assert len(FluSurvNetDataSource().locations) > 0


class TestDiseaseDataLoaderMerge:
    """Tests for DiseaseDataLoader merge logic using mocked sources."""


    def _make_source_df(self, source_name: str, locations=("01", "06")) -> pd.DataFrame:
        rows = []
        for loc in locations:
            rows.append({
                "source": source_name,
                "agg_level": "state",
                "location": loc,
                "season": "2023/24",
                "season_week": 15,
                "wk_end_date": pd.Timestamp("2024-01-06"),
                "inc": 0.5,
            })
        return pd.DataFrame(rows)


    def _make_mock_source(self, source_value: str):
        src = MagicMock()
        src.load.return_value = self._make_source_df(source_value)
        return src


    def test_load_combines_sources(self):
        src1 = self._make_mock_source("nhsn")
        src2 = self._make_mock_source("ilinet")
        loader = DiseaseDataLoader()
        as_of = datetime.date(2024, 1, 6)

        df = loader.load(sources=[src1, src2], as_of=as_of)

        assert set(df["source"].unique()) == {"nhsn", "ilinet"}


    def test_load_merges_ancillary(self):
        src = self._make_mock_source("nhsn")
        pop_data = pd.DataFrame({
            "location": ["01", "06"],
            "pop": [5_000_000.0, 40_000_000.0],
            "log_pop": [np.log(5_000_000), np.log(40_000_000)],
        })
        anc = MagicMock()
        anc.load.return_value = pop_data

        loader = DiseaseDataLoader()
        df = loader.load(sources=[src], as_of=datetime.date(2024, 1, 6), ancillary=[anc])

        assert "pop" in df.columns
        assert "log_pop" in df.columns
        assert df["pop"].notna().all()


    def test_load_passes_pop_for_hsa_when_ancillary_has_it(self):
        rows = self._make_source_df("nhsn")
        rows.loc[0, "agg_level"] = "hsa"
        src = MagicMock()
        src.load.return_value = rows

        pop_data = pd.DataFrame({
            "location": ["01", "06"],
            "pop": [5_000_000.0, 40_000_000.0],
            "log_pop": [np.log(5_000_000), np.log(40_000_000)],
        })
        anc = MagicMock()
        anc.load.return_value = pop_data

        loader = DiseaseDataLoader()
        df = loader.load(sources=[src], as_of=datetime.date(2024, 1, 6), ancillary=[anc])

        hsa_rows = df[df["agg_level"] == "hsa"]
        assert hsa_rows["pop"].notna().all()
        assert hsa_rows["log_pop"].notna().all()


    def test_warns_when_nhsn_hhs_and_drop_pandemic_false(self):
        src = self._make_mock_source("nhsn")
        src.source_name = SourceType.NHSN
        loader = DiseaseDataLoader()
        with pytest.warns(UserWarning, match="does not contain complete data during pandemic seasons"):
            loader.load(sources=[src], as_of=datetime.date(2023, 1, 1), drop_pandemic_seasons=False)


    def test_no_warn_when_drop_pandemic_true_or_recent_as_of(self):
        loader = DiseaseDataLoader()

        src = self._make_mock_source("nhsn")
        src.source_name = SourceType.NHSN
        # no warning when drop_pandemic_seasons=True (default), even with old as_of
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            loader.load(sources=[src], as_of=datetime.date(2023, 1, 1), drop_pandemic_seasons=True)

        # no warning when as_of >= 2024-11-15, even with drop_pandemic_seasons=False
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            loader.load(sources=[src], as_of=datetime.date(2024, 11, 15), drop_pandemic_seasons=False)


    def test_warns_when_flusurvnet_burden_adj_and_drop_pandemic_false(self):
        src = self._make_mock_source("flusurvnet")
        src.source_name = SourceType.FLUSURVNET
        src.burden_adj = True
        loader = DiseaseDataLoader()
        with pytest.warns(UserWarning, match="burden adjustment estimates do not exist for pandemic seasons"):
            loader.load(sources=[src], as_of=datetime.date(2023, 1, 1), drop_pandemic_seasons=False)


    def test_no_warn_when_flusurvnet_drop_pandemic_true_or_no_burden_adj(self):
        loader = DiseaseDataLoader()

        src = self._make_mock_source("flusurvnet")
        src.source_name = SourceType.FLUSURVNET
        src.burden_adj = True
        # no warning when drop_pandemic_seasons=True, even with burden_adj=True
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            loader.load(sources=[src], as_of=datetime.date(2023, 1, 1), drop_pandemic_seasons=True)

        src.burden_adj = False
        # no warning when burden_adj=False, even with drop_pandemic_seasons=False
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            loader.load(sources=[src], as_of=datetime.date(2023, 1, 1), drop_pandemic_seasons=False)
