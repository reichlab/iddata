import datetime
import warnings
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata import utils
from iddata.constants import PANDEMIC_SEASONS, S3_DATA_RAW_URL
from iddata.enums import AggLevel, Disease, SourceType
from iddata.s3 import get_versioned_file_path
from iddata.sources.base import DataSource
from iddata.utils import load_fips_mappings


class NSSPDataSource(DataSource):
    source_name = SourceType.NSSP


    def __init__(self, disease: Disease = Disease.FLU, drop_pandemic_seasons: bool = True,
                 agg_level: AggLevel = AggLevel.STATE):
        self.disease = disease
        self.drop_pandemic_seasons = drop_pandemic_seasons
        self.agg_level = agg_level


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load NSSP emergency department visit data. Raises ValueError if as_of is None. Only supports as_of >=
        2025-09-17. Returns all available granularities (state + HSA rows).
        """
        if as_of is None:
            raise ValueError("NSSP requires as_of to be specified.")

        if isinstance(as_of, str):
            as_of = datetime.date.fromisoformat(as_of)
        if as_of < datetime.date.fromisoformat("2025-09-17"):
            raise NotImplementedError("NSSP only supports as_of >= 2025-09-17.")

        valid_diseases = (Disease.FLU, Disease.COVID, Disease.RSV)
        if self.disease not in valid_diseases:
            raise ValueError(f"NSSP supports {valid_diseases}; got {self.disease}.")

        file_path = get_versioned_file_path(
            "infectious-disease-data/data-raw/nssp/nssp-????-??-??.csv",
            as_of)
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, file_path))

        if self.disease == Disease.FLU:
            inc_colname = "percent_visits_influenza"
        elif self.disease == Disease.COVID:
            inc_colname = "percent_visits_covid"
        else:
            inc_colname = "percent_visits_rsv"

        # filter, for each hsa_nci_id (excluding states) to include one value per week
        # because some that contain multiple counties are duplicated in the data
        # also, here `fips` is the full 6-digit code
        dat = (
            dat.sort_values(by=["fips", "hsa_nci_id", "week_end"], ascending=[True, True, False])
            .assign(unique_id=lambda x: np.where(x["hsa_nci_id"] == "All", x["fips"], x["hsa_nci_id"]))
            .drop_duplicates(subset=["unique_id", "week_end"], keep="first")
        )

        # keep hsa_nci_id as this is the location code we will be indexing on
        dat = dat[["geography", "hsa_nci_id", "week_end", inc_colname]]
        dat.columns = ["location_name", "hsa_nci_id", "wk_end_date", "inc"]

        # get to location codes / (2-digit) FIPS
        fips_mappings = load_fips_mappings()
        dat = dat.merge(fips_mappings, on=["location_name"], how="left") \
            .rename(columns={"location": "fips_code"})

        dat = utils.add_season_columns(dat)

        if self.drop_pandemic_seasons:
            dat.loc[dat["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["hsa_nci_id"] == "All", "state", "hsa")
        dat["agg_level"] = np.where(dat["fips_code"] == "US", "national", dat["agg_level"])
        dat["location"] = np.where(dat["hsa_nci_id"] == "All", dat["fips_code"], dat["hsa_nci_id"])

        dat["source"] = SourceType.NSSP.value
        dat = self._fill_missing_states(dat)

        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc", "source"]]
        return dat


    def _fill_missing_states(self, dat: pd.DataFrame) -> pd.DataFrame:
        # pull out already-aggregated locations
        df_nssp_states = dat.loc[(dat["agg_level"] == "state") & (~np.isnan(dat["inc"]))]
        nonmissing_states = df_nssp_states["fips_code"].unique()
        if len(nonmissing_states) >= 50: # 49 states (no MO) + PR
            return dat

        # fill in missing data by averaging nssp hsa values for a particular state
        df_nssp_missing_states = (
            dat.loc[(~dat["fips_code"].isin(nonmissing_states)) & (dat["agg_level"] == "hsa")]
            .groupby(["fips_code", "season", "season_week", "wk_end_date", "source"])
            .agg(inc=("inc", "mean"))
            .reset_index()
            .assign(agg_level="state")
        )
        df_nssp_missing_states["location"] = df_nssp_missing_states["fips_code"]
        missing_states = df_nssp_missing_states["fips_code"].unique()
        warnings.warn(f"Interpolating missing values for states {missing_states}")

        result = pd.concat(
            [dat.loc[dat["agg_level"] != "state"], df_nssp_states, df_nssp_missing_states],
            join="inner",
            axis=0)
        return result.drop_duplicates(subset=["location", "fips_code", "wk_end_date", "source"], keep="first")
