import datetime
from abc import ABC, abstractmethod

import pandas as pd

from iddata.enums import SourceType


class DataSource(ABC):
    """
    Abstract base class for a single disease surveillance data source.

    All implementations return a DataFrame with the standard schema:
        location     (str):      FIPS code, HHS region string, or HSA NCI ID
        agg_level    (str):      aggregation level of the row, e.g. "state", "national", "hsa", "hhs region"
        wk_end_date  (datetime): Saturday end-of-week date
        season       (str):      (northern hemisphere) infectious disease season, e.g., "2023/24"
        season_week  (int):      weeks since MMWR week 30 of the prior year (1-based)
        inc          (float):    incidence in source-specific units
        source       (str):      source name (equals source_name.value)
    """


    @property
    @abstractmethod
    def source_name(self) -> SourceType:
        """Returns the SourceType for this data source."""
        ...


    @abstractmethod
    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load data available as of `as_of`. Versioned sources (NHSN, NSSP) raise ValueError if as_of is None.
        Non-versioned sources (ILINet, FluSurvNet) ignore as_of.
        """
        ...
