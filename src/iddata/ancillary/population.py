import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData


class PopulationData(AncillaryData):
    """
    US Census population by location.

    Returns a DataFrame with columns:
        location  (str):   FIPS code or national identifier
        pop       (float): state or national population
        log_pop   (float): log(pop)
    """

    def load(self) -> pd.DataFrame:
        """Load population data from S3 and return per-location (most recent year)."""
        from iddata.sources.nhsn import _load_us_census
        census = _load_us_census()
        # Use the most recent season's population for each location
        latest = census.sort_values("season").groupby("location").last().reset_index()
        latest = latest[["location", "pop"]]
        latest = latest[~latest["pop"].isna()]
        # Exclude HSA-level rows (no census pop)
        latest["log_pop"] = np.log(latest["pop"])
        return latest
