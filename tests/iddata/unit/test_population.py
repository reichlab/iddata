"""Unit tests for the PopulationData ancillary source. These use no network access.

Coverage of the real S3/SEER/census.gov loads lives in tests/iddata/integration/test_population.py.
"""

import datetime

import pytest

from iddata.ancillary.population import _all_seasons


@pytest.mark.parametrize("as_of, season_expected", [
    # rollover falls on epiweek 30 -> 31, which lands a day later than usual in 2026
    (datetime.date(2026, 8, 1), "2025/26"),  # still epiweek 30: previous season
    (datetime.date(2026, 8, 2), "2026/27"),  # epiweek 31: new season begins
    (datetime.date(2027, 1, 15), "2026/27"),  # mid-season, after new year
])
def test_all_seasons(as_of, season_expected):
    seasons = _all_seasons(as_of)

    assert seasons[0] == "1997/98"
    assert seasons[-1] == season_expected
    assert len(seasons) == len(set(seasons))  # no duplicates
    assert seasons == sorted(seasons)  # strictly increasing


def test_all_seasons_defaults_to_today(monkeypatch):
    from iddata.ancillary import population

    class _FakeDate(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date(2026, 8, 2)

    monkeypatch.setattr(population, "date", _FakeDate)

    assert _all_seasons() == _all_seasons(datetime.date(2026, 8, 2))
