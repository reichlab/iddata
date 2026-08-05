"""Unit tests for iddata.nowcast.vintage_cache.VintageCache."""

import datetime

import pandas as pd

from iddata.enums import SourceType
from iddata.nowcast.triangle import weekly_as_of_dates
from iddata.nowcast.vintage_cache import VintageCache


class _FakeSource:
    """Returns a distinct frame per as_of, tracking call count for memoization tests."""

    source_name = SourceType.NHSN


    def __init__(self):
        self.load_call_count = 0
        self.calls = []


    def load(self, as_of=None):
        self.load_call_count += 1
        self.calls.append(as_of)
        return pd.DataFrame({
            "location": ["US"], "agg_level": ["national"],
            "wk_end_date": [pd.Timestamp(as_of)], "inc": [float(as_of.toordinal())], "source": ["nhsn"],
        })


class _GapSimulatingSource:
    """
    Mimics NHSNDataSource's real behavior across the confirmed 2024-05-01 to 2024-11-15 data
    gap: `get_versioned_file_path` finds no newer archived file until `gap_end`, so every as_of
    in [gap_start, gap_end) resolves to the exact same stale snapshot (the one from gap_start).
    """

    source_name = SourceType.NHSN


    def __init__(self, gap_start: datetime.date, gap_end: datetime.date):
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.load_call_count = 0


    def load(self, as_of=None):
        self.load_call_count += 1
        resolved_as_of = self.gap_start if self.gap_start <= as_of < self.gap_end else as_of
        return pd.DataFrame({
            "location": ["US"], "agg_level": ["national"],
            "wk_end_date": [pd.Timestamp(resolved_as_of)],
            "inc": [float(resolved_as_of.toordinal())], "source": ["nhsn"],
        })


class TestVintageCacheMemoization:
    def test_get_many_fetches_each_distinct_as_of_once(self):
        src = _FakeSource()
        cache = VintageCache()
        as_of_dates = [datetime.date(2026, 1, 3), datetime.date(2026, 1, 10)]

        cache.get_many(src, as_of_dates)
        assert src.load_call_count == 2

        # requesting the same dates again should use the cache, not re-fetch
        cache.get_many(src, as_of_dates)
        assert src.load_call_count == 2


    def test_get_many_only_fetches_newly_requested_dates(self):
        src = _FakeSource()
        cache = VintageCache()
        cache.get_many(src, [datetime.date(2026, 1, 3)])
        assert src.load_call_count == 1

        cache.get_many(src, [datetime.date(2026, 1, 3), datetime.date(2026, 1, 10)])
        assert src.load_call_count == 2  # only the new date triggers a fetch


    def test_get_many_returns_a_frame_per_requested_date(self):
        src = _FakeSource()
        cache = VintageCache()
        as_of_dates = [datetime.date(2026, 1, 3), datetime.date(2026, 1, 10)]

        result = cache.get_many(src, as_of_dates)

        assert set(result.keys()) == set(as_of_dates)
        assert all(isinstance(df, pd.DataFrame) for df in result.values())


class TestNDistinctVintages:
    def test_counts_genuinely_different_vintages(self):
        src = _FakeSource()
        cache = VintageCache()
        as_of_dates = [datetime.date(2026, 1, 3), datetime.date(2026, 1, 10), datetime.date(2026, 1, 17)]

        cache.get_many(src, as_of_dates)

        assert cache.n_distinct_vintages(as_of_dates) == 3


    def test_nhsn_2024_data_gap_dedupes_to_a_single_distinct_vintage(self):
        # Regression test for the confirmed ~6.5-month NHSN gap (2024-05-01 to 2024-11-15):
        # every weekly as_of requested inside it must resolve to ONE distinct vintage, not be
        # miscounted as many independent snapshots (which would corrupt min_vintages checks and,
        # if used naively, the reporting triangle itself).
        gap_start = datetime.date(2024, 5, 1)
        gap_end = datetime.date(2024, 11, 15)
        src = _GapSimulatingSource(gap_start, gap_end)
        cache = VintageCache()

        as_of_dates = [d for d in weekly_as_of_dates(gap_end, 30) if gap_start <= d < gap_end]
        assert len(as_of_dates) > 1  # sanity check: we're actually exercising multiple requested dates

        cache.get_many(src, as_of_dates)

        assert cache.n_distinct_vintages(as_of_dates) == 1


    def test_dates_outside_the_gap_are_not_deduped(self):
        gap_start = datetime.date(2024, 5, 1)
        gap_end = datetime.date(2024, 11, 15)
        src = _GapSimulatingSource(gap_start, gap_end)
        cache = VintageCache()

        as_of_dates = [gap_start - datetime.timedelta(weeks=1), gap_end, gap_end + datetime.timedelta(weeks=1)]
        cache.get_many(src, as_of_dates)

        assert cache.n_distinct_vintages(as_of_dates) == 3
