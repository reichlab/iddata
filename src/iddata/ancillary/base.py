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
