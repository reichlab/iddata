import datetime
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata import utils
from iddata.ancillary.population import _load_us_census
from iddata.constants import PANDEMIC_SEASONS, S3_DATA_RAW_URL
from iddata.enums import Disease, SourceType
from iddata.s3 import get_versioned_file_path
from iddata.sources.base import DataSource
from iddata.utils import load_fips_mappings


class NHSNDataSource(DataSource):
    source_name = SourceType.NHSN


    def __init__(self, disease: Disease = Disease.FLU, rates: bool = True, drop_pandemic_seasons: bool = True):
        self.disease = disease
        self.rates = rates
        self.drop_pandemic_seasons = drop_pandemic_seasons


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load NHSN hospitalization data. Raises ValueError if as_of is None. Routes to the HHS archive for as_of <
        2024-11-15, or the NHSN source for later dates.
        """
        if as_of is None:
            raise ValueError("NHSNDataSource requires as_of to be specified.")

        if isinstance(as_of, str):
            as_of = datetime.date.fromisoformat(as_of)
        if as_of < datetime.date.fromisoformat("2024-11-15"):
            if not self.drop_pandemic_seasons:
                raise NotImplementedError(
                    "NHSNDataSource does not support drop_pandemic_seasons=False for as_of prior to 2024-11-15.")
            if self.disease != Disease.FLU:
                raise NotImplementedError(
                    f"NHSNDataSource only supports Disease.FLU for as_of prior to 2024-11-15; got {self.disease}.")
            dat = self._load_from_hhs(as_of)
        else:
            dat = self._load_from_nhsn(as_of)

        ew_str = dat.apply(utils.date_to_ew_str, axis=1)
        dat["season"] = utils.convert_epiweek_to_season(ew_str)
        dat["season_week"] = utils.convert_epiweek_to_season_week(ew_str)
        dat = dat.sort_values(by=["season", "season_week"])

        if self.drop_pandemic_seasons:
            dat.loc[dat["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        if self.rates:
            pops = _load_us_census()
            dat = dat.merge(pops[["location", "season", "pop"]], on=["location", "season"], how="left") \
                .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
        dat["source"] = SourceType.NHSN.value
        return dat


    def _load_from_hhs(self, as_of: datetime.date) -> pd.DataFrame:
        """Returns raw DataFrame with location, wk_end_date, inc columns."""
        file_path = get_versioned_file_path(
            "infectious-disease-data/data-raw/influenza-hhs/hhs-????-??-??.csv",
            as_of)
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, file_path))
        dat.rename(columns={"date": "wk_end_date"}, inplace=True)
        return dat[["location", "wk_end_date", "inc"]]


    def _load_from_nhsn(self, as_of: datetime.date) -> pd.DataFrame:
        """Returns raw DataFrame with location, wk_end_date, inc columns."""
        if self.disease not in (Disease.FLU, Disease.COVID):
            raise ValueError("NHSNDataSource supports only Disease.FLU and Disease.COVID.")

        file_path = get_versioned_file_path(
            "infectious-disease-data/data-raw/influenza-nhsn/nhsn-????-??-??.csv",
            as_of)
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, file_path))

        if self.disease == Disease.FLU:
            inc_colname = "Total Influenza Admissions"
        else:
            inc_colname = "Total COVID-19 Admissions"
        dat = dat[["Geographic aggregation", "Week Ending Date", inc_colname]]
        dat.columns = ["abbreviation", "wk_end_date", "inc"]
        dat.loc[dat["abbreviation"] == "USA", "abbreviation"] = "US"

        fips_mappings = load_fips_mappings()
        dat = dat.merge(fips_mappings, on=["abbreviation"], how="left")
        return dat[["location", "wk_end_date", "inc"]]
