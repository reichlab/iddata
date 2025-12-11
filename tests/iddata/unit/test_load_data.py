import datetime

import numpy as np
import pytest
from iddata.loader import DiseaseDataLoader


def test_load_data_sources():
    fdl = DiseaseDataLoader()
    
    sources_options = [
        ["nhsn"],
        ["nhsn", "ilinet"],
        ["flusurvnet"],
        ["flusurvnet", "nhsn", "ilinet"],
        ["nssp"]
    ]
    for sources in sources_options:
        df = fdl.load_data(sources=sources)
        assert set(df["source"].unique()) == set(sources)
    
    df = fdl.load_data()
    assert set(df["source"].unique()) == {"flusurvnet", "nhsn", "ilinet"}


def test_nssp_columns():
    fdl = DiseaseDataLoader()
    
    nhsn_df = fdl.load_data(sources=["nhsn"])
    nssp_df = fdl.load_data(sources=["nssp"])
    assert set(nssp_df.columns) == set(nhsn_df.columns)


@pytest.mark.parametrize("select_date, select_locations, expected_agg_levels", [
    ("2025-09-06", ["US", "01", "25", "25"], ["national", "state", "state", "hsa"])
])
def test_nssp_locations(select_date, select_locations, expected_agg_levels):
    fdl = DiseaseDataLoader()
    df = fdl.load_data(sources=["nssp"])
    subset_df = df.loc[(df["wk_end_date"] == select_date) & (df["location"].isin(select_locations))]
    
    # Get actual aggregation levels as a sorted list to preserve duplicates
    actual_agg_levels = sorted(subset_df["agg_level"].tolist())
    expected_agg_levels_sorted = sorted(expected_agg_levels)
    
    assert actual_agg_levels == expected_agg_levels_sorted


@pytest.mark.parametrize("test_kwargs, season_expected, wk_end_date_expected", [
    (None, "2022/23", "2023-12-23"),
    # ({"drop_pandemic_seasons": False}, "2019/20", "2023-12-23"),
    ({"drop_pandemic_seasons": True, "as_of": datetime.date.fromisoformat("2023-12-30")},
        "2022/23", "2023-12-23")
])
def test_load_data_nhsn_kwargs(test_kwargs, season_expected, wk_end_date_expected):
    fdl = DiseaseDataLoader()
    df = fdl.load_data(sources=["nhsn"], nhsn_kwargs=test_kwargs)
    
    assert df.dropna()["season"].min() == season_expected
    
    # data is snapshotted on wednesday -> as_of will be previous saturday until following wednesday
    wk_end_date_actual = str(df["wk_end_date"].max())[:10]
    if test_kwargs is not None and "as_of" in test_kwargs:
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual > wk_end_date_expected


@pytest.mark.parametrize("test_kwargs, expect_all_na", [
    (None, True),
    ({"drop_pandemic_seasons": False}, False),
    ({"drop_pandemic_seasons": True}, True)
])
def test_load_data_ilinet_kwargs(test_kwargs, expect_all_na):
    fdl = DiseaseDataLoader()
    
    df = fdl.load_data(sources=["ilinet"], ilinet_kwargs=test_kwargs)
    
    if expect_all_na:
        assert np.all(df.loc[df["season"].isin(["2008/09", "2009/10", "2020/21", "2021/22"]), "inc"].isna())
    else:
        # expect some non-NA values in pandemic seasons
        assert np.any(~df.loc[df["season"].isin(["2008/09", "2009/10", "2020/21", "2021/22"]), "inc"].isna())


@pytest.mark.parametrize("test_kwargs", [
    (None),
    ({"locations": ["California", "Colorado", "Connecticut"]})
])
def test_load_data_flusurvnet_kwargs(test_kwargs):
    fdl = DiseaseDataLoader()
    
    #flusurv_kwargs
    df = fdl.load_data(sources=["flusurvnet"], flusurvnet_kwargs=test_kwargs)
    
    if test_kwargs is None:
        assert len(df["location"].unique()) > 3
    else:
        assert len(df["location"].unique()) == len(test_kwargs["locations"])

@pytest.mark.parametrize("test_kwargs, season_expected, wk_end_date_expected", [
    (None, "2022/23", "2025-09-06"),
    ({"drop_pandemic_seasons": True, "as_of": datetime.date.fromisoformat("2025-09-10")},
        "2022/23", "2025-09-06")
])
def test_load_data_nssp_kwargs(test_kwargs, season_expected, wk_end_date_expected):
    fdl = DiseaseDataLoader()
    df = fdl.load_data(sources=["nssp"], nssp_kwargs=test_kwargs)

    assert df["season"].min() == season_expected
    
    # data is snapshotted on wednesday -> as_of will be previous saturday until following wednesday
    wk_end_date_actual = str(df["wk_end_date"].max())[:10]
    if test_kwargs is not None and "as_of" in test_kwargs:
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual > wk_end_date_expected


@pytest.mark.parametrize("disease", ["flu", "covid", "rsv"])
def test_load_nssp_from_epidata_diseases(disease):
    """Test that load_nssp_from_epidata works for all supported diseases."""
    fdl = DiseaseDataLoader()

    # Use an as_of date when epidata data is available
    df = fdl.load_nssp_from_epidata(disease=disease, as_of=datetime.date.fromisoformat("2024-06-01"))

    assert len(df) > 0
    assert "inc" in df.columns
    assert df["source"].unique()[0] == "nssp"


def test_load_nssp_from_epidata_agg_levels():
    """Test that epidata returns expected aggregation levels."""
    fdl = DiseaseDataLoader()

    df = fdl.load_nssp_from_epidata(as_of=datetime.date.fromisoformat("2024-06-01"))

    agg_levels = set(df["agg_level"].unique())
    assert "national" in agg_levels
    assert "state" in agg_levels
    assert "hsa" in agg_levels


def test_load_nssp_from_epidata_columns():
    """Test that epidata returns expected columns."""
    fdl = DiseaseDataLoader()

    df = fdl.load_nssp_from_epidata(as_of=datetime.date.fromisoformat("2024-06-01"))

    expected_cols = {"agg_level", "location", "fips_code", "season", "season_week", "wk_end_date", "inc", "source"}
    assert set(df.columns) == expected_cols


def test_load_nssp_from_epidata_wk_end_date_is_saturday():
    """Test that wk_end_date values are all Saturdays."""
    fdl = DiseaseDataLoader()

    df = fdl.load_nssp_from_epidata(as_of=datetime.date.fromisoformat("2024-06-01"))

    # Saturday is weekday 5
    assert all(df["wk_end_date"].dt.weekday == 5)


def test_load_nssp_from_epidata_drop_pandemic_seasons():
    """Test that pandemic seasons are dropped when requested."""
    fdl = DiseaseDataLoader()

    df = fdl.load_nssp_from_epidata(as_of=datetime.date.fromisoformat("2024-06-01"), drop_pandemic_seasons=True)

    pandemic_data = df[df["season"].isin(["2020/21", "2021/22"])]
    if len(pandemic_data) > 0:
        assert pandemic_data["inc"].isna().all()
