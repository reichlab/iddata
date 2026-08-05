"""Fetches and caches historical `DataSource.load(as_of=...)` vintages for nowcasting."""

from __future__ import annotations

import datetime

import pandas as pd

from iddata.sources.base import DataSource


class VintageCache:
    """
    Fetches `source.load(as_of=d)` for a batch of historical `as_of` dates, memoizing by the
    requested date and tracking which requested dates actually resolved to the *same*
    underlying snapshot (e.g. NHSN's 2024-05-01 to 2024-11-15 gap, where every `as_of` in that
    window resolves to the same stale archived file via `get_versioned_file_path`).

    This does not avoid the redundant network fetch itself for distinct-but-stale-duplicate
    `as_of` dates (doing so would require reaching into source-specific internals like
    `get_versioned_file_path`, which `DataSource` doesn't expose) -- but it does avoid
    re-fetching an exact-duplicate `as_of` twice, and critically, `n_distinct_vintages` lets
    callers count *actual* distinct snapshots rather than requested dates, so a data gap
    doesn't get miscounted as many independent vintages (see `min_vintages` in `NowcastConfig`).
    """

    def __init__(self):
        self._by_as_of: dict[datetime.date, pd.DataFrame] = {}
        self._fingerprint_by_as_of: dict[datetime.date, tuple] = {}


    def get_many(self, source: DataSource, as_of_dates: list[datetime.date]) -> dict[datetime.date, pd.DataFrame]:
        """Fetch (and cache) `source.load(as_of=d)` for each `d` in `as_of_dates`."""
        result = {}
        for d in sorted(set(as_of_dates)):
            if d not in self._by_as_of:
                df = source.load(as_of=d)
                self._by_as_of[d] = df
                self._fingerprint_by_as_of[d] = self._fingerprint(df)
            result[d] = self._by_as_of[d]
        return result


    def n_distinct_vintages(self, as_of_dates: list[datetime.date]) -> int:
        """Number of distinct underlying snapshots among the given (already-fetched) as_of dates."""
        fingerprints = {self._fingerprint_by_as_of[d] for d in as_of_dates if d in self._fingerprint_by_as_of}
        return len(fingerprints)


    @staticmethod
    def _fingerprint(df: pd.DataFrame) -> tuple:
        """Cheap identity check: two fetches with the same fingerprint are the same underlying snapshot."""
        return (df["wk_end_date"].max(), len(df), round(float(df["inc"].sum()), 6))
