"""End-to-end DiseaseDataLoader tests against real data. These require network access to S3 and CDC endpoints.

Fast, mocked coverage of the uniform loader logic (source merging, ancillary joins, drop_pandemic_seasons) lives in
tests/iddata/unit/test_sources.py; the tests here are integration sanity checks that the real sources still load and
that as_of snapshot selection works.
"""

import datetime

import pytest

from iddata.loader import DiseaseDataLoader
from iddata.sources.flusurvnet import FluSurvNetDataSource
from iddata.sources.ilinet import ILINetDataSource
from iddata.sources.nhsn import NHSNDataSource
from iddata.sources.nssp import NSSPDataSource

_DEFAULT_AS_OF = datetime.date.fromisoformat("2023-12-30")
_NSSP_AS_OF = datetime.date.fromisoformat("2025-09-20")


@pytest.mark.parametrize("sources, expected_source_values", [
    ([NHSNDataSource()], {"nhsn"}),
    ([ILINetDataSource()], {"ilinet"}),
    ([FluSurvNetDataSource()], {"flusurvnet"}),
    ([NSSPDataSource()], {"nssp"}),
    ([NHSNDataSource(), ILINetDataSource(), FluSurvNetDataSource(), NSSPDataSource()],
     {"nhsn", "ilinet", "flusurvnet", "nssp"}),
])
def test_load_data_sources(sources, expected_source_values):
    loader = DiseaseDataLoader()

    as_of = _NSSP_AS_OF if any(isinstance(s, NSSPDataSource) for s in sources) else _DEFAULT_AS_OF
    df = loader.load(sources=sources, as_of=as_of)
    assert set(df["source"].unique()) == expected_source_values


def test_nssp_columns():
    loader = DiseaseDataLoader()

    nhsn_df = loader.load(sources=[NHSNDataSource()], as_of=_DEFAULT_AS_OF)
    nssp_df = loader.load(sources=[NSSPDataSource()], as_of=_NSSP_AS_OF)
    assert set(nssp_df.columns) == set(nhsn_df.columns)


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
