"""End-to-end DiseaseDataLoader tests against real data. These require network access to S3 and CDC endpoints.

Fast, mocked coverage of the uniform loader logic (source merging, ancillary joins, drop_pandemic_seasons) lives in
tests/iddata/unit/test_sources.py; the tests here are integration sanity checks that the real sources still load and
that as_of snapshot selection works.
"""

import datetime

import pytest

from iddata.ancillary.population import PopulationData
from iddata.constants import PANDEMIC_SEASONS
from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource
from iddata.sources.smh import SMHDataSource

_DEFAULT_AS_OF = datetime.date.fromisoformat("2023-12-30")
_NSSP_AS_OF = datetime.date.fromisoformat("2025-09-20")


def _smh_test_source(**kwargs) -> SMHDataSource:
    """
    An SMHDataSource trimmed to one model and two trajectories. Loading every model for rounds 4-6 peaks at ~17 GB, more
    than a GitHub-hosted runner has. In round 4 these output_type_ids cover all 52 locations (including US); in rounds
    5+ output_type_ids are per-location, so they return only a few rows.
    """
    return SMHDataSource(model_id=["MOBS_NEU-GLEAM_FLU"], output_type_id=["1", "2"], **kwargs)


@pytest.mark.parametrize("sources, expected_source_values", [
    ([NHSNDataSource()], {"nhsn"}),
    ([ILINetDataSource()], {"ilinet"}),
    ([FluSurvNetDataSource()], {"flusurvnet"}),
    ([NSSPDataSource()], {"nssp"}),
    ([_smh_test_source()], {"smh"}),
    ([NHSNDataSource(), ILINetDataSource(), FluSurvNetDataSource(), NSSPDataSource(), _smh_test_source()],
     {"nhsn", "ilinet", "flusurvnet", "nssp", "smh"}),
])
def test_load_data_sources(sources, expected_source_values):
    loader = DiseaseDataLoader()

    as_of = _NSSP_AS_OF if any(isinstance(s, NSSPDataSource) for s in sources) else _DEFAULT_AS_OF
    df = loader.load(sources=sources, as_of=as_of)
    # SMH source values are "smh-<model_id>"; collapse them to "smh" so every source is compared the same way
    source_values = df["source"].where(~df["source"].str.startswith("smh-"), "smh")
    assert set(source_values.unique()) == expected_source_values

    # drop_pandemic_seasons defaults to True, so no real source should return usable inc for those seasons.
    # ILINet is currently the only source whose data actually spans one: at these as_of dates NHSN and NSSP
    # start at 2022/23, and FluSurvNet skips 2020/21 and 2021/22 entirely. Requiring the rows to be present
    # for ILINet keeps this from silently degrading to a no-op if that upstream coverage ever changes.
    pandemic_mask = df["season"].isin(PANDEMIC_SEASONS)
    if any(isinstance(s, ILINetDataSource) for s in sources):
        assert pandemic_mask.any()
    if pandemic_mask.any():
        assert df.loc[pandemic_mask, "inc"].isna().all()


def test_nssp_columns():
    loader = DiseaseDataLoader()

    nhsn_df = loader.load(sources=[NHSNDataSource()], as_of=_DEFAULT_AS_OF)
    nssp_df = loader.load(sources=[NSSPDataSource()], as_of=_NSSP_AS_OF)
    assert set(nssp_df.columns) == set(nhsn_df.columns)


def test_smh_wk_end_date_is_saturday():
    # rates=False: this test only checks date alignment, so it skips the population load needed for rates
    df = _smh_test_source(rates=False).load(as_of=_DEFAULT_AS_OF)
    assert (df["wk_end_date"].dt.dayofweek == 5).all()


@pytest.mark.parametrize("ancillary, expect_pop", [(None, False), ([PopulationData()], True)])
def test_smh_rates_population_source(ancillary, expect_pop):
    # rates=True converts inc with population whether or not it was requested via ancillary,
    # but pop/log_pop are only returned when requested.
    counts = _smh_test_source(rates=False).load(as_of=_DEFAULT_AS_OF)
    rates = _smh_test_source(rates=True).load(as_of=_DEFAULT_AS_OF, ancillary=ancillary)

    assert ("pop" in rates.columns) == expect_pop
    assert ("log_pop" in rates.columns) == expect_pop
    # every location, including the national row, must get a population to convert with
    assert (rates["agg_level"] == "national").any()
    assert rates["inc"].notna().all()
    # every location has pop > 100k, so converting to rates per 100k must shrink inc
    assert rates["inc"].sum() < counts["inc"].sum()


def test_nssp_locations():
    select_date = "2025-09-06"
    select_locations = ["US", "01", "25", "25"]
    expected_agg_levels = ["national", "state", "state", "hsa"]

    loader = DiseaseDataLoader()
    df = loader.load(sources=[NSSPDataSource()], as_of=_NSSP_AS_OF)
    subset_df = df.loc[(df["wk_end_date"] == select_date) & (df["location"].isin(select_locations))]

    # Get actual aggregation levels as a sorted list to preserve duplicates
    actual_agg_levels = sorted(subset_df["agg_level"].tolist())

    assert actual_agg_levels == sorted(expected_agg_levels)


@pytest.mark.parametrize("pinned", [True, False])
@pytest.mark.parametrize("source_cls, pinned_as_of, wk_end_date_expected", [
    (NHSNDataSource, _DEFAULT_AS_OF, "2023-12-23"),
    (NSSPDataSource, _NSSP_AS_OF, "2025-09-06"),
])
def test_as_of_selects_snapshot(pinned, source_cls, pinned_as_of, wk_end_date_expected):
    """A pinned as_of must resolve to that exact snapshot; as_of=today must resolve to one at least that recent."""
    loader = DiseaseDataLoader()

    as_of = pinned_as_of if pinned else datetime.date.today()
    df = loader.load(sources=[source_cls()], as_of=as_of)

    wk_end_date_actual = str(df["wk_end_date"].max())[:10]
    if pinned:
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual >= wk_end_date_expected

    # pandemic seasons have inc NaN'd out by default, so the earliest season with data is post-pandemic
    assert df.dropna(subset=["inc"])["season"].min() == "2022/23"


@pytest.mark.parametrize("locations", [
    None,
    ["California", "Colorado", "Connecticut"],
])
def test_flusurvnet_locations_filter(locations):
    loader = DiseaseDataLoader()

    df = loader.load(
        sources=[FluSurvNetDataSource(locations=locations)],
        as_of=_DEFAULT_AS_OF,
    )

    if locations is None:
        assert len(df["location"].unique()) > 3
    else:
        assert len(df["location"].unique()) == len(locations)
