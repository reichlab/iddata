import datetime
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata import utils
from iddata.constants import PANDEMIC_SEASONS, S3_DATA_RAW_URL
from iddata.enums import Disease, SourceType
from iddata.s3 import get_versioned_file_path
from iddata.sources.base import DataSource


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
            if self.disease != Disease.FLU:
                raise NotImplementedError(
                    f"NHSN via HHS archive (as_of < 2024-11-15) only supports Disease.FLU; got {self.disease}."
                )
            return self._load_from_hhs(as_of)
        return self._load_from_nhsn(as_of)


    def _load_from_hhs(self, as_of: datetime.date) -> pd.DataFrame:
        file_path = get_versioned_file_path(
            "infectious-disease-data/data-raw/influenza-hhs/hhs-????-??-??.csv",
            as_of)
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, file_path))
        dat.rename(columns={"date": "wk_end_date"}, inplace=True)

        ew_str = dat.apply(utils.date_to_ew_str, axis=1)
        dat["season"] = utils.convert_epiweek_to_season(ew_str)
        dat["season_week"] = utils.convert_epiweek_to_season_week(ew_str)
        dat = dat.sort_values(by=["season", "season_week"])

        if self.rates:
            pops = _load_us_census()
            dat = dat.merge(pops, on=["location", "season"], how="left") \
                .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
        dat["source"] = SourceType.NHSN.value
        return dat


    def _load_from_nhsn(self, as_of: datetime.date) -> pd.DataFrame:
        if self.disease not in (Disease.FLU, Disease.COVID):
            raise ValueError("NHSN supports only Disease.FLU and Disease.COVID.")

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

        fips_mappings = pd.read_csv(urljoin(S3_DATA_RAW_URL, "fips-mappings/fips_mappings.csv"))
        dat = dat.merge(fips_mappings, on=["abbreviation"], how="left")

        ew_str = dat.apply(utils.date_to_ew_str, axis=1)
        dat["season"] = utils.convert_epiweek_to_season(ew_str)
        dat["season_week"] = utils.convert_epiweek_to_season_week(ew_str)
        dat = dat.sort_values(by=["season", "season_week"])

        if self.drop_pandemic_seasons:
            dat.loc[dat["season"].isin(PANDEMIC_SEASONS), "inc"] = np.nan

        if self.rates:
            pops = _load_us_census()
            dat = dat.merge(pops, on=["location", "season"], how="left") \
                .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
        dat["source"] = SourceType.NHSN.value
        return dat


def _load_us_census() -> pd.DataFrame:
    """Load US Census population data (location × season)."""
    from itertools import product

    def _load_one(f):
        dat = pd.read_csv(f, engine="python", dtype={"STATE": str})
        dat = dat.loc[(dat["NAME"] == "United States") | (dat["STATE"] != "00"),
                      (dat.columns == "STATE") | (dat.columns.str.startswith("POPESTIMATE"))]
        dat = dat.melt(id_vars="STATE", var_name="season", value_name="pop")
        dat.rename(columns={"STATE": "location"}, inplace=True)
        dat.loc[dat["location"] == "00", "location"] = "US"
        dat["season"] = dat["season"].str[-4:]
        dat["season"] = dat["season"] + "/" + (dat["season"].str[-2:].astype(int) + 1).astype(str)
        return dat


    files = [urljoin(S3_DATA_RAW_URL, "us-census/nst-est2019-alldata.csv"),
             urljoin(S3_DATA_RAW_URL, "us-census/NST-EST2023-ALLDATA.csv")]
    us_pops = pd.concat([_load_one(f) for f in files], axis=0)

    fips_mappings = pd.read_csv(urljoin(S3_DATA_RAW_URL, "fips-mappings/fips_mappings.csv"))
    hhs_pops = (
        us_pops.query("location != 'US'")
        .merge(
            fips_mappings.query("location != 'US'")
            .assign(hhs_region=lambda x: "Region " + x["hhs_region"].astype(int).astype(str)),
            on="location",
            how="left",
        )
        .groupby(["hhs_region", "season"])["pop"]
        .sum()
        .reset_index()
        .rename(columns={"hhs_region": "location"})
    )
    dat = pd.concat([us_pops, hhs_pops], axis=0)

    all_locations = dat["location"].unique()
    all_seasons = [str(y) + "/" + str(y + 1)[-2:] for y in range(1997, 2026)]
    full_result = pd.DataFrame.from_records(product(all_locations, all_seasons))
    full_result.columns = ["location", "season"]
    dat = (
        full_result.merge(dat, how="left", on=["location", "season"])
        .set_index("location")
        .groupby(["location"])
        .bfill()
        .groupby(["location"])
        .ffill()
        .reset_index()
    )
    return dat
