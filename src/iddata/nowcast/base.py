"""
Core seam for nowcasting: `NowcastConfig`, the `Nowcaster` extension point, and the
`NowcastedDataSource` decorator that lets `DiseaseDataLoader.load()` apply a nowcast correction
transparently, with no changes needed to `DataSource` implementations or to `idmodels`.
"""

from __future__ import annotations

import datetime
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from iddata.enums import SourceType
from iddata.sources.base import DataSource


@dataclass
class NowcastConfig:
    """
    Opt-in configuration for correcting the trailing, still-incomplete observations of a
    DataSource for reporting delay/backfill.

    max_delay_weeks : int | None
        Number of weeks of delay considered "incomplete" and therefore corrected. If None
        (the default), each `Nowcaster` looks up a static, per-`Disease` default (see e.g.
        `iddata.constants.NHSN_MAX_DELAY_WEEKS`) rather than estimating it live -- pass an
        explicit int to override.
    training_window_weeks : int | None
        Number of trailing reference weeks used to fit the delay distribution. Defaults to
        `max(3 * max_delay_weeks, 12)` (see `nhsn.cap_training_window_to_cutover`'s docstring
        for why a bare `3x` multiplier isn't enough on its own once max_delay_weeks is small).
    min_vintages : int
        Minimum number of *distinct* historical vintages (see `VintageCache.n_distinct_vintages`)
        required to fit a delay distribution; governs `on_insufficient_data` below.
    on_insufficient_data : str
        "passthrough" (default): warn and return the uncorrected data if `min_vintages` isn't
        met. "raise": raise a ValueError instead.
    sources : tuple[SourceType, ...]
        Which sources to correct, if a registered `Nowcaster` exists for them.
    pmf_shrinkage_k : float
        Shrinks each (location, agg_level) group's own fitted delay-PMF toward a PMF pooled
        across every group, weighted by that group's own data volume `n` (total case count in
        its fitting window) via `w = n / (n + pmf_shrinkage_k)` -- i.e. `pmf_shrinkage_k` is the
        volume at which a group's own fit and the pooled fit get equal weight. Only used by
        `NHSNNowcaster`; see its module docstring and the project plan's "v2 Attempt" sections.
        A pure per-location fit (`pmf_shrinkage_k=0`) was tried and made backtest error more than
        double vs. leaving the data uncorrected, even for locations whose own fit was otherwise
        accurate -- a small per-location training window makes the chain-ladder ratio estimator
        too noisy on its own. Shrinking toward the pooled fit only helps modestly (~6% reduction
        in the one backtest run so far, after a since-fixed bug in the pooled fit itself was
        corrected -- see `estimate_delay_pooled`'s docstring; an earlier, buggy pooled fit had
        made shrinkage look like it roughly halved the damage, which did not hold up once the
        pooled PMF was computed correctly) and does NOT flip the sign: correction remains worse
        than raw at every `pmf_shrinkage_k` tried (2,000 / 10,000 / 50,000 all land within ~0.1 of
        each other), because the surviving error concentrates specifically on dates near a sharp
        seasonal peak (a different failure mode -- time-varying completion, not location noise --
        that shrinkage doesn't address). Treat the default as a reasonable starting point, not a
        finely-tuned value; the near-flatness across a 25x range of k suggests this parameter
        isn't the lever that matters most here.
    """

    max_delay_weeks: int | None = None
    training_window_weeks: int | None = None
    min_vintages: int = 10
    on_insufficient_data: str = "passthrough"
    sources: tuple[SourceType, ...] = (SourceType.NHSN,)
    pmf_shrinkage_k: float = 10_000.0


    def __post_init__(self):
        if self.max_delay_weeks is not None and self.max_delay_weeks < 1:
            raise ValueError(f"NowcastConfig.max_delay_weeks must be >= 1; got {self.max_delay_weeks}")
        if self.on_insufficient_data not in ("passthrough", "raise"):
            raise ValueError(
                f"NowcastConfig.on_insufficient_data must be 'passthrough' or 'raise'; "
                f"got {self.on_insufficient_data!r}"
            )
        if self.pmf_shrinkage_k < 0:
            raise ValueError(f"NowcastConfig.pmf_shrinkage_k must be >= 0; got {self.pmf_shrinkage_k}")


class Nowcaster(ABC):
    """Corrects the trailing, incomplete observations of a single DataSource's latest snapshot."""

    source_name: SourceType


    def __init__(self, config: NowcastConfig):
        self.config = config


    @abstractmethod
    def correct(self, latest_df: pd.DataFrame, as_of: datetime.date, source: DataSource) -> pd.DataFrame:
        """
        Parameters
        ----------
        latest_df : the standard-schema DataFrame already returned by source.load(as_of=as_of)
        as_of     : the same as_of passed to source.load()
        source    : the underlying (unwrapped) DataSource, used to fetch additional historical
                    vintages needed to fit the delay distribution

        Returns
        -------
        A copy of latest_df, same schema/shape/dtypes, with `inc` values for the trailing,
        still-incomplete weeks per (location, agg_level) group replaced by nowcast estimates.
        All earlier (already-complete) rows are returned unchanged.
        """
        ...


class NowcastedDataSource(DataSource):
    """Decorator: wraps a DataSource so that load() returns a delay-corrected latest snapshot."""

    def __init__(self, source: DataSource, nowcaster: Nowcaster):
        self._source = source
        self._nowcaster = nowcaster


    @property
    def source_name(self) -> SourceType:
        return self._source.source_name


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        df = self._source.load(as_of=as_of)
        return self._nowcaster.correct(df, as_of=as_of, source=self._source)


_NOWCASTER_REGISTRY: dict[SourceType, type[Nowcaster]] = {}


def register_nowcaster(source_name: SourceType, cls: type[Nowcaster]) -> None:
    """Register a Nowcaster subclass to be used for a given SourceType by wrap_sources()."""
    _NOWCASTER_REGISTRY[source_name] = cls


def wrap_sources(sources: list[DataSource], config: NowcastConfig) -> list[DataSource]:
    """
    Wrap only sources whose source_name is in config.sources AND has a registered Nowcaster.

    Warns if config.sources requests a SourceType with no registered Nowcaster at all (e.g.
    SourceType.NSSP today, before a v2 NSSPNowcaster exists) -- otherwise that request would
    silently no-op, which could easily be mistaken for "nowcasting is happening."
    """
    unregistered = [s for s in config.sources if s not in _NOWCASTER_REGISTRY]
    if unregistered:
        warnings.warn(
            f"NowcastConfig.sources requested nowcasting for {[s.value for s in unregistered]}, "
            "but no Nowcaster is registered for those source(s); they will be left uncorrected. "
            f"Registered nowcasters: {sorted(s.value for s in _NOWCASTER_REGISTRY)}"
        )

    result = []
    for src in sources:
        cls = _NOWCASTER_REGISTRY.get(src.source_name)
        if cls is not None and src.source_name in config.sources:
            result.append(NowcastedDataSource(src, cls(config)))
        else:
            result.append(src)
    return result
