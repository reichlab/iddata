from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class AncillaryData(ABC):
    """
    Base class for supplementary data used by models but never as training targets.

    Unlike DataSource subclasses:
      - AncillaryData has no standard schema; format is implementation-defined.
      - as_of is optional and implementation-defined: most implementations ignore it, but
        one that reflects wall-clock time (e.g. "the current season") should accept it so
        that a query is reproducible from its inputs instead of depending on real-world time.
    """


    @abstractmethod
    def load(self, as_of: date | None = None) -> pd.DataFrame:
        """
        Load and return the ancillary data.

        Parameters
        ----------
        as_of : date | None
            Reference date to load the data as of. Implementations that don't depend on the
            current date may ignore this. Defaults to None, which implementations should
            interpret as "as of today".

        Returns a DataFrame whose columns are implementation-defined. DiseaseDataLoader.load() merges this into the
        surveillance DataFrame by location (left join).
        """
        ...


def merge_ancillary(df: pd.DataFrame, anc: AncillaryData, as_of: date | None) -> pd.DataFrame:
    """
    Left-join `anc`'s data onto `df` by location, plus season and agg_level when both frames have them.
    """
    anc_df = anc.load(as_of=as_of)
    join_keys = ["location", "season"] if "season" in anc_df.columns else ["location"]
    if "agg_level" in anc_df.columns and "agg_level" in df.columns:
        join_keys.append("agg_level")
    return df.merge(anc_df, how="left", on=join_keys)
