"""Unit tests for iddata DataSource classes (constructor, source_name, validation)."""

import datetime
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from iddata.enums import DataBackend, Disease, SourceType
from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from iddata.sources.smh import SMHDataSource


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


    def test_default_backend(self):
        src = NHSNDataSource()
        assert src.backend == DataBackend.S3
        assert src.epidata_ctx is None


    def test_backend_string_is_coerced(self):
        assert NHSNDataSource(backend="epidata").backend == DataBackend.EPIDATA


    def test_epidata_backend_rejects_rsv(self):
        with pytest.raises(ValueError, match="epidata"):
            NHSNDataSource(disease=Disease.RSV, backend=DataBackend.EPIDATA)


    @pytest.mark.parametrize("backend, as_of, expected", [
        (DataBackend.S3, "2024-11-14", "_load_from_hhs"),
        (DataBackend.S3, "2024-11-15", "_load_from_nhsn"),
        (DataBackend.S3, "2024-11-19", "_load_from_nhsn"),
        (DataBackend.EPIDATA, "2024-11-14", "_load_from_hhs"),
        (DataBackend.EPIDATA, "2024-11-15", "_load_from_nhsn"),
        (DataBackend.EPIDATA, "2024-11-18", "_load_from_nhsn"),
        (DataBackend.EPIDATA, "2024-11-19", "_load_from_epidata"),
    ])
    def test_loader_for(self, backend, as_of, expected):
        src = NHSNDataSource(backend=backend)
        assert src._loader_for(datetime.date.fromisoformat(as_of)) == getattr(src, expected)


    @pytest.mark.parametrize("backend", [DataBackend.S3, DataBackend.EPIDATA])
    def test_covid_before_cutover_raises(self, backend):
        src = NHSNDataSource(disease=Disease.COVID, backend=backend)
        with pytest.raises(NotImplementedError, match="2024-11-15"):
            src.load(as_of=datetime.date(2024, 11, 14))


    @staticmethod
    def _fips_mappings() -> pd.DataFrame:
        return pd.DataFrame({"abbreviation": ["US", "MA"], "location": ["US", "25"]})


    @staticmethod
    def _mock_epidata_ctx(df: pd.DataFrame) -> MagicMock:
        ctx = MagicMock()
        ctx.epidata_snapshot.return_value.df.return_value = df
        return ctx


    def _epidata_snapshot_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "signal": pd.array(["confirmed_admissions_flu_ew"] * 4, dtype="string"),
            "geo_type": pd.array(["nation", "nation", "state", "state"], dtype="string"),
            "geo_value": pd.array(["us", "us", "ma", "ma"], dtype="string"),
            "reference_time": pd.to_datetime(["2025-01-04", "2025-01-11"] * 2).astype("datetime64[us]"),
            "value": pd.array([100.0, 200.0, 10.0, None], dtype="Float64"),
        })


    @pytest.mark.parametrize("disease, signal", [
        (Disease.FLU, "confirmed_admissions_flu_ew"),
        (Disease.COVID, "confirmed_admissions_covid_ew"),
    ])
    def test_load_from_epidata(self, disease, signal):
        ctx = self._mock_epidata_ctx(self._epidata_snapshot_df())
        src = NHSNDataSource(disease=disease, backend=DataBackend.EPIDATA, epidata_ctx=ctx)
        as_of = datetime.date(2025, 1, 15)
        with patch("iddata.sources.nhsn.load_fips_mappings", return_value=self._fips_mappings()):
            result = src._load_from_epidata(as_of)

        ctx.epidata_snapshot.assert_called_once_with(
            source="nhsn", signals=signal, geo_type=["nation", "state"], snapshot_date=as_of, fill_method="source")
        assert list(result.columns) == ["location", "wk_end_date", "inc"]
        assert result["location"].tolist() == ["US", "US", "25", "25"]
        assert result["wk_end_date"].tolist() == ["2025-01-04", "2025-01-11"] * 2
        assert result["inc"].dtype == np.float64
        assert result["inc"].tolist()[:3] == [100.0, 200.0, 10.0]
        assert np.isnan(result["inc"].iloc[3])


    def test_load_through_epidata(self):
        ctx = self._mock_epidata_ctx(self._epidata_snapshot_df())
        src = NHSNDataSource(backend=DataBackend.EPIDATA, rates=False, epidata_ctx=ctx)
        with patch("iddata.sources.nhsn.load_fips_mappings", return_value=self._fips_mappings()):
            result = src.load(as_of="2025-01-15")

        assert list(result.columns) == ["agg_level", "location", "season", "season_week", "wk_end_date", "inc",
                                        "source"]
        assert pd.api.types.is_datetime64_any_dtype(result["wk_end_date"])
        assert result["inc"].dtype == np.float64
        assert set(result["season"]) == {"2024/25"}
        assert (result["source"] == "nhsn").all()
        agg_by_loc = result.drop_duplicates("location").set_index("location")["agg_level"].to_dict()
        assert agg_by_loc == {"US": "national", "25": "state"}


    def test_load_from_epidata_empty_raises(self):
        ctx = self._mock_epidata_ctx(self._epidata_snapshot_df().iloc[0:0])
        src = NHSNDataSource(backend=DataBackend.EPIDATA, epidata_ctx=ctx)
        with pytest.raises(ValueError, match="no NHSN rows"):
            src._load_from_epidata(datetime.date(2025, 1, 15))


    def test_epidata_requires_key(self, monkeypatch):
        monkeypatch.delenv("DELPHI_EPIDATA_KEY", raising=False)
        src = NHSNDataSource(backend=DataBackend.EPIDATA, rates=False)
        with pytest.raises(RuntimeError, match="DELPHI_EPIDATA_KEY"):
            src.load(as_of=datetime.date(2025, 1, 15))


    @pytest.mark.parametrize("as_of, s3_loader", [
        (datetime.date(2024, 3, 2), "_load_from_hhs"),
        (datetime.date(2024, 11, 18), "_load_from_nhsn"),
    ])
    def test_epidata_fallback_needs_no_key(self, monkeypatch, as_of, s3_loader):
        monkeypatch.delenv("DELPHI_EPIDATA_KEY", raising=False)
        src = NHSNDataSource(backend=DataBackend.EPIDATA, rates=False)
        raw = pd.DataFrame({"location": ["US", "25"], "wk_end_date": ["2024-02-24", "2024-02-24"], "inc": [1.0, 2.0]})
        with patch.object(src, s3_loader, return_value=raw) as mock_loader:
            result = src.load(as_of=as_of)

        mock_loader.assert_called_once_with(as_of)
        assert len(result) == 2
        assert src.epidata_ctx is None


    def test_load_from_nhsn_drops_region_rows(self):
        raw = pd.DataFrame({
            "Geographic aggregation": ["USA", "MA", "Region 1", "Region 10"],
            "Week Ending Date": ["2026-10-10"] * 4,
            "Total Influenza Admissions": [100.0, 10.0, 20.0, 30.0],
            "Total COVID-19 Admissions": [50.0, 5.0, 6.0, 7.0],
        })
        src = NHSNDataSource()
        with patch("iddata.sources.nhsn.get_versioned_file_path", return_value="nhsn-2026-10-14.csv"), \
                patch("iddata.sources.nhsn.pd.read_csv", return_value=raw), \
                patch("iddata.sources.nhsn.load_fips_mappings", return_value=self._fips_mappings()):
            result = src._load_from_nhsn(datetime.date(2026, 10, 14))

        assert result["location"].tolist() == ["US", "25"]
        assert result["inc"].tolist() == [100.0, 10.0]


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


class TestSMHDataSource:
    def test_source_name(self):
        assert SMHDataSource.source_name == SourceType.SMH


    def test_default_disease(self):
        assert SMHDataSource().disease == Disease.FLU


    def test_load_requires_as_of(self):
        with pytest.raises(ValueError, match="as_of"):
            SMHDataSource().load(as_of=None)


    @pytest.mark.parametrize("disease", [Disease.COVID, Disease.RSV])
    def test_load_rejects_unsupported_disease(self, disease):
        with pytest.raises(ValueError, match="SMH supports"):
            SMHDataSource(disease=disease).load(as_of=datetime.date(2023, 12, 30))


    @pytest.mark.parametrize("null_col", ["model_id", "scenario_id", "output_type_id"])
    def test_load_rejects_null_identifiers(self, null_col):
        round_df = pd.DataFrame({
            "model_id": ["m1", "m1"],
            "scenario_id": ["A-2023", "A-2023"],
            "location": ["01", "US"],
            "output_type_id": ["1", "1"],
            "value": [1.0, 2.0],
            "origin_date": ["2023-09-03", "2023-09-03"],
            "horizon": [1, 1],
        })
        round_df.loc[0, null_col] = None

        with patch("iddata.sources.smh.pd.read_parquet", return_value=round_df), \
                pytest.raises(ValueError, match=null_col):
            SMHDataSource(rates=False).load(as_of=datetime.date(2023, 12, 30))


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


    def test_load_requires_a_source(self):
        with pytest.raises(ValueError, match="at least one source"):
            DiseaseDataLoader().load(sources=[], as_of=datetime.date(2024, 1, 6))


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


    @pytest.mark.parametrize("ancillary", [None, []])
    def test_load_without_ancillary_skips_population_merge(self, ancillary):
        src = self._make_mock_source("nhsn")
        loader = DiseaseDataLoader()

        df = loader.load(sources=[src], as_of=datetime.date(2024, 1, 6), ancillary=ancillary)

        assert "pop" not in df.columns
        assert "log_pop" not in df.columns


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


    @pytest.mark.parametrize("drop_pandemic, expect_na", [
        (True, True),
        (False, False),
    ])
    def test_drop_pandemic_seasons_sets_inc_na(self, drop_pandemic, expect_na):
        """The season strings here are deliberately literal rather than read from the constant in iddata.constants.

        Deriving them from that constant would make the test tautological: it would keep passing if the constant
        drifted away from the "YYYY/YY" format that add_season_columns actually produces, silently disabling the
        NaN assignment for every real source. One literal is taken from each pandemic pair.
        """
        rows = self._make_source_df("ilinet", locations=("01", "06", "13"))
        rows.loc[0, "season"] = "2009/10"
        rows.loc[1, "season"] = "2020/21"
        src = MagicMock()
        src.source_name = SourceType.ILINET
        src.load.return_value = rows

        loader = DiseaseDataLoader()
        df = loader.load(sources=[src], as_of=datetime.date(2024, 1, 6), drop_pandemic_seasons=drop_pandemic)

        pandemic_rows = df[df["season"].isin(["2009/10", "2020/21"])]
        assert len(pandemic_rows) == 2
        assert bool(pandemic_rows["inc"].isna().all()) is expect_na
        # non-pandemic seasons are untouched either way
        assert df[df["season"] == "2023/24"]["inc"].notna().all()


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
