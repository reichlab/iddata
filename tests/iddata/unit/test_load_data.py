import datetime

import numpy as np
import pytest

from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource


_DEFAULT_AS_OF = datetime.date.fromisoformat("2023-12-30")
_NSSP_AS_OF = datetime.date.fromisoformat("2025-09-20")


def test_load_data_sources():
    loader = DiseaseDataLoader()

    cases = [([NHSNDataSource()], {"nhsn"}),
             ([NHSNDataSource(), ILINetDataSource()], {"nhsn", "ilinet"}),
             ([FluSurvNetDataSource()], {"flusurvnet"}),
             ([FluSurvNetDataSource(), NHSNDataSource(), ILINetDataSource()], {"flusurvnet", "nhsn", "ilinet"})]
    for sources, expected_source_values in cases:
        df = loader.load(sources=sources, as_of=_DEFAULT_AS_OF)
        assert set(df["source"].unique()) == expected_source_values


def test_nssp_columns():
    loader = DiseaseDataLoader()

    nhsn_df = loader.load(sources=[NHSNDataSource()], as_of=_DEFAULT_AS_OF)
    nssp_df = loader.load(sources=[NSSPDataSource()], as_of=_NSSP_AS_OF)
    assert set(nssp_df.columns) == set(nhsn_df.columns)


@pytest.mark.parametrize("select_date, select_locations, expected_agg_levels", [
    ("2025-09-06", ["US", "01", "25", "25"], ["national", "state", "state", "hsa"])
])
def test_nssp_locations(select_date, select_locations, expected_agg_levels):
    loader = DiseaseDataLoader()
    df = loader.load(sources=[NSSPDataSource()], as_of=_NSSP_AS_OF)
    subset_df = df.loc[(df["wk_end_date"] == select_date) & (df["location"].isin(select_locations))]

    actual_agg_levels = sorted(subset_df["agg_level"].tolist())
    expected_agg_levels_sorted = sorted(expected_agg_levels)

    assert actual_agg_levels == expected_agg_levels_sorted


@pytest.mark.parametrize("drop_pandemic, as_of, season_expected, wk_end_date_expected", [
    (True, datetime.date.today(), "2022/23", "2023-12-23"),
    (True, datetime.date.fromisoformat("2023-12-30"), "2022/23", "2023-12-23"),
])
def test_load_data_nhsn_kwargs(drop_pandemic, as_of, season_expected, wk_end_date_expected):
    loader = DiseaseDataLoader()

    df = loader.load(
        sources=[NHSNDataSource(drop_pandemic_seasons=drop_pandemic)],
        as_of=as_of,
    )

    assert df.dropna()["season"].min() == season_expected
    wk_end_date_actual = str(df["wk_end_date"].max())[:10]
    if as_of == datetime.date.fromisoformat("2023-12-30"):
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual >= wk_end_date_expected


@pytest.mark.parametrize("drop_pandemic, expect_all_na", [
    (True, True),
    (False, False),
])
def test_load_data_ilinet_kwargs(drop_pandemic, expect_all_na):
    loader = DiseaseDataLoader()

    df = loader.load(
        sources=[ILINetDataSource(drop_pandemic_seasons=drop_pandemic)],
        as_of=_DEFAULT_AS_OF,
    )

    if expect_all_na:
        assert np.all(df.loc[df["season"].isin(["2008/09", "2009/10", "2020/21", "2021/22"]), "inc"].isna())
    else:
        assert np.any(~df.loc[df["season"].isin(["2008/09", "2009/10", "2020/21", "2021/22"]), "inc"].isna())


@pytest.mark.parametrize("locations", [
    None,
    ["California", "Colorado", "Connecticut"],
])
def test_load_data_flusurvnet_kwargs(locations):
    loader = DiseaseDataLoader()

    df = loader.load(
        sources=[FluSurvNetDataSource(locations=locations) if locations else FluSurvNetDataSource()],
        as_of=_DEFAULT_AS_OF,
    )

    if locations is None:
        assert len(df["location"].unique()) > 3
    else:
        assert len(df["location"].unique()) == len(locations)


@pytest.mark.parametrize("drop_pandemic, as_of, season_expected, wk_end_date_expected", [
    (True, datetime.date.today(), "2022/23", "2025-09-06"),
    (True, datetime.date.fromisoformat("2025-09-20"), "2022/23", "2025-09-06"),
])
def test_load_data_nssp_kwargs(drop_pandemic, as_of, season_expected, wk_end_date_expected):
    loader = DiseaseDataLoader()

    df = loader.load(
        sources=[NSSPDataSource(drop_pandemic_seasons=drop_pandemic)],
        as_of=as_of,
    )

    assert df["season"].min() == season_expected
    wk_end_date_actual = str(df["wk_end_date"].max())[:10]
    if as_of == datetime.date.fromisoformat("2025-09-20"):
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual >= wk_end_date_expected
