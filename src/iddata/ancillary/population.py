from itertools import product
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData
from iddata.constants import S3_DATA_RAW_URL


def _load_us_census() -> pd.DataFrame:
    """Load US Census population data (location × season)."""

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


class PopulationData(AncillaryData):
    """
    US Census population by location and season.

    Returns a DataFrame with columns:
        location  (str):   FIPS code or national identifier
        season    (str):   e.g., "2023/24"
        pop       (float): population estimate for that season's year
        log_pop   (float): log(pop)
    """

    def load(self) -> pd.DataFrame:
        """Load population data from S3, returning per-location-season estimates."""
        census = _load_us_census()
        census = census[["location", "season", "pop"]]
        census = census[~census["pop"].isna()]
        census["log_pop"] = np.log(census["pop"])
        return census
