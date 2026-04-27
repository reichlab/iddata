from abc import ABC, abstractmethod

import pandas as pd


class AncillaryData(ABC):
    """
    Base class for supplementary data used by models but never as training targets.

    Unlike DataSource subclasses:
      - AncillaryData has no standard schema; format is implementation-defined.
      - Implementations are not expected to be versioned (no as_of parameter).
    """


    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and return the ancillary data.

        Returns a DataFrame whose columns are implementation-defined. DiseaseDataLoader.load() merges this into the
        surveillance DataFrame by location (left join).
        """
        ...
