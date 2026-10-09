import datetime
import warnings

import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData, merge_ancillary
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
        Load and merge data from the specified sources, plus any ancillary data. 
        Does NOT apply power transforms or center/scale normalization.

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
        if not drop_pandemic_seasons and any(
            src.source_name == SourceType.FLUSURVNET and getattr(src, "burden_adj", False)
            for src in sources
        ):
            warnings.warn(
                "FluSurv-NET burden adjustment estimates do not exist for pandemic seasons; "
                "those seasons will have NaN inc regardless of drop_pandemic_seasons."
            )

        if not sources:
            raise ValueError("DiseaseDataLoader.load() requires at least one source.")

        non_smh_sources = [src for src in sources if src.source_name != SourceType.SMH]
        smh_source = next((src for src in sources if src.source_name == SourceType.SMH), None)

        # SMH merges ancillary data itself because it rewrites location and season before returning, after which they
        # no longer match the ancillary keys.
        frames = []
        if non_smh_sources:
            df = pd.concat([src.load(as_of=as_of) for src in non_smh_sources], axis=0)
            for anc in ancillary or []:
                df = merge_ancillary(df, anc, as_of)
            frames.append(df)
        if smh_source is not None:
            frames.append(smh_source.load(as_of=as_of, ancillary=ancillary))

        # season sorts in the same order as wk_end_date for surveillance sources, so including it only changes the order
        # of SMH rows, where it keeps each trajectory together.
        df = pd.concat(frames, axis=0).sort_values(["source", "location", "season", "wk_end_date"])

        # SMH rows are not masked here: their season values combine the season with a scenario and trajectory ID (e.g.
        # "2023/24A-12"), so they never match PANDEMIC_SEASONS. SMH rounds 4-6 do not cover any pandemic season.
        if drop_pandemic_seasons:
            df.loc[df["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        return df
