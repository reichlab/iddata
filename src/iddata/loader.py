import datetime
import warnings

import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData
from iddata.constants import PANDEMIC_SEASONS
from iddata.enums import SourceType
from iddata.sources.base import DataSource


class DiseaseDataLoader:
    """
    Thin orchestrator: loads data from DataSource objects and optionally merges ancillary data.
    """


    def load(self, sources: list[DataSource], as_of: datetime.date,
             ancillary: list[AncillaryData] | None = None,
             drop_pandemic_seasons: bool = True) -> pd.DataFrame:
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
        """
        if not drop_pandemic_seasons and as_of < datetime.date(2024, 11, 15) and \
                any(src.source_name == SourceType.NHSN for src in sources):
            warnings.warn(
                "NHSN does not contain complete data during pandemic seasons for an as_of date before 2024-11-15."
            )

        frames = [src.load(as_of=as_of) for src in sources]
        df = pd.concat(frames, axis=0).sort_values(["source", "location", "wk_end_date"])

        if drop_pandemic_seasons:
            df.loc[df["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        if ancillary:
            for anc in ancillary:
                anc_df = anc.load()
                join_keys = ["location", "season"] if "season" in anc_df.columns else ["location"]
                df = df.merge(anc_df, how="left", on=join_keys)

            # Census doesn't provide HSA pop; null it out to avoid spurious joins
            if "pop" in df.columns:
                df["pop"] = np.where(df["agg_level"] == "hsa", np.nan, df["pop"])
            if "log_pop" in df.columns:
                df["log_pop"] = np.where(df["agg_level"] == "hsa", np.nan, df["log_pop"])
        return df
