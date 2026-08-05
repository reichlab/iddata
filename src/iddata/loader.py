import datetime
import warnings

import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData
from iddata.constants import PANDEMIC_SEASONS
from iddata.enums import SourceType
from iddata.nowcast.base import NowcastConfig, wrap_sources
from iddata.sources.base import DataSource


class DiseaseDataLoader:
    """
    Thin orchestrator: loads data from DataSource objects and optionally merges ancillary data.
    """


    def load(self, sources: list[DataSource], as_of: datetime.date,
             ancillary: list[AncillaryData] | None = None,
             drop_pandemic_seasons: bool = True,
             nowcast: NowcastConfig | None = None) -> pd.DataFrame:
        """
        Load and merge data from the specified sources, plus any ancillary data. Does NOT apply power transforms or
        center/scale normalization.

        Parameters
        ----------
        sources : list[DataSource]
            Instantiated DataSource objects to load from.
        as_of : datetime.date
            Reference date passed to each source's load() method.
        ancillary : list[AncillaryData] | None
            Supplementary data merged into the result by location (left join).
            Typically [PopulationData()] for models that need pop and log_pop.
        drop_pandemic_seasons : bool
            If True (default), set inc to NaN for pandemic seasons across all sources.
        nowcast : NowcastConfig | None
            If provided, wraps eligible sources (see `NowcastConfig.sources`) so their most
            recent, still-incomplete observations are corrected for reporting delay before
            being loaded. Default None preserves existing behavior exactly.
        """
        if nowcast is not None:
            sources = wrap_sources(sources, nowcast)

        if not drop_pandemic_seasons and as_of < datetime.date(2024, 11, 15) and \
                any(src.source_name == SourceType.NHSN for src in sources):
            warnings.warn(
                "NHSN does not contain complete data during pandemic seasons for an as_of date before 2024-11-15."
            )
        if not drop_pandemic_seasons and any(
            src.source_name == SourceType.FLUSURVNET and getattr(src, "burden_adj", False)
            for src in sources
        ):
            warnings.warn(
                "FluSurv-NET burden adjustment estimates do not exist for pandemic seasons; "
                "those seasons will have NaN inc regardless of drop_pandemic_seasons."
            )

        frames = [src.load(as_of=as_of) for src in sources]
        df = pd.concat(frames, axis=0).sort_values(["source", "location", "wk_end_date"])

        if drop_pandemic_seasons:
            df.loc[df["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        if ancillary:
            for anc in ancillary:
                anc_df = anc.load()
                join_keys = ["location", "season"] if "season" in anc_df.columns else ["location"]
                if "agg_level" in anc_df.columns and "agg_level" in df.columns:
                    join_keys.append("agg_level")
                df = df.merge(anc_df, how="left", on=join_keys)

        return df
