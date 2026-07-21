from itertools import product
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata.ancillary.base import AncillaryData
from iddata.constants import S3_DATA_RAW_URL

_SEER_HSA_CROSSWALK_URL = "https://seer.cancer.gov/seerstat/variables/countyattribs/Health.Service.Areas.xls"
_CENSUS_COUNTY_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv"
_CENSUS_COUNTY_2010_2019_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"

# FIPS codes retired after the crosswalk was last published; maps old code -> list of current replacements
_STALE_FIPS = {
    "02270": ["02158"],          # Wade Hampton Census Area renamed Kusilvak (2015)
    "02261": ["02063", "02066"], # Valdez-Cordova split into Chugach + Copper River (2019)
}


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
    dat["agg_level"] = np.where(dat["location"] == "US", "national",
                       np.where(dat["location"].str.startswith("Region"), "hhs region", "state"))

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


def _load_county_pop_long(url: str) -> pd.DataFrame:
    """Fetch a Census county population file and return (fips, year, pop) in long format."""
    df = pd.read_csv(url, encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    df = df[df["COUNTY"] != "000"].copy()
    df["fips"] = df["STATE"].str.zfill(2) + df["COUNTY"].str.zfill(3)
    pop_cols = [c for c in df.columns if c.startswith("POPESTIMATE")]
    long = df[["fips"] + pop_cols].melt(id_vars="fips", var_name="year_col", value_name="pop")
    long["year"] = long["year_col"].str.replace("POPESTIMATE", "").astype(int)
    return long[["fips", "year", "pop"]]


def _load_hsa_populations() -> pd.DataFrame:
    """Load HSA-level population by aggregating Census county data via the NCI SEER crosswalk."""
    # --- Crosswalk ---
    raw = pd.read_excel(_SEER_HSA_CROSSWALK_URL, sheet_name="HSA (NCI Modified)", header=None, engine="xlrd")
    crosswalk = (
        raw.iloc[1:]
        .rename(columns={0: "location", 1: "hsa_name", 2: "state_county", 3: "fips"})
        .assign(
            location=lambda x: x["location"].astype(str).str.strip(),
            fips=lambda x: x["fips"].astype(str).str.zfill(5),
        )[["location", "fips"]]
        .drop_duplicates()
    )
    # Drop Census placeholder codes (county portion >= 900, e.g. 02900, 02999, 09999, 15900-15999)
    crosswalk = crosswalk[crosswalk["fips"].str[2:].astype(int) < 900]
    # Replace stale FIPS codes with their current equivalents
    replacement_rows = []
    stale_mask = crosswalk["fips"].isin(_STALE_FIPS)
    for _, row in crosswalk[stale_mask].iterrows():
        for new_fips in _STALE_FIPS[row["fips"]]:
            replacement_rows.append({"location": row["location"], "fips": new_fips})
    crosswalk = pd.concat(
        [crosswalk[~stale_mask], pd.DataFrame(replacement_rows)], ignore_index=True
    )

    # --- Census county populations ---
    # Use the 2020-2024 file for all states except CT: CT reorganized counties in 2022 and the
    # new planning-region FIPS (09110-09190) don't match the old county codes in the crosswalk.
    pop_recent = _load_county_pop_long(_CENSUS_COUNTY_URL)
    pop_non_ct = pop_recent[~pop_recent["fips"].str.startswith("09")]
    # Use the 2010-2019 file for CT only: old county FIPS (09001-09015) match the crosswalk exactly.
    pop_ct = _load_county_pop_long(_CENSUS_COUNTY_2010_2019_URL)
    pop_ct = pop_ct[pop_ct["fips"].str.startswith("09")]

    county_pop = pd.concat([pop_non_ct, pop_ct], axis=0, ignore_index=True)

    # --- Aggregate to HSA via crosswalk ---
    merged = crosswalk.merge(county_pop, on="fips", how="left")
    hsa_long = merged.groupby(["location", "year"])["pop"].sum(min_count=1).reset_index()
    hsa_long = hsa_long.dropna(subset=["year"])
    hsa_long["year"] = hsa_long["year"].astype(int)

    # --- HSA 996 (Alaska) and 997 (Hawaii): crosswalk uses fake FIPS for these whole-state HSAs;
    #     use state totals from the 2020-2024 Census file instead. ---
    ak_hi = pop_recent[pop_recent["fips"].str[:2].isin(["02", "15"])].copy()
    ak_hi["location"] = ak_hi["fips"].str[:2].map({"02": "996", "15": "997"})
    ak_hi_long = ak_hi.groupby(["location", "year"])["pop"].sum(min_count=1).reset_index()
    hsa_long = pd.concat([hsa_long, ak_hi_long], axis=0, ignore_index=True)

    # --- Convert year to season string (e.g. 2023 -> "2023/24") ---
    hsa_long["season"] = hsa_long["year"].astype(str) + "/" + hsa_long["year"].apply(lambda y: str(y + 1)[-2:])
    hsa_long = hsa_long[["location", "season", "pop"]]

    # --- Extend to all seasons via forward/backward fill (matching the approach in _load_us_census) ---
    all_seasons = [str(y) + "/" + str(y + 1)[-2:] for y in range(1997, 2026)]
    full_result = pd.DataFrame.from_records(
        product(hsa_long["location"].unique(), all_seasons), columns=["location", "season"]
    )
    result = (
        full_result.merge(hsa_long, how="left", on=["location", "season"])
        .set_index("location")
        .groupby("location")
        .bfill()
        .groupby("location")
        .ffill()
        .reset_index()
    )
    result["agg_level"] = "hsa"
    return result


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
        hsa = _load_hsa_populations()
        dat = pd.concat(
            [census[["location", "season", "pop", "agg_level"]], hsa[["location", "season", "pop", "agg_level"]]],
            axis=0,
        )
        dat = dat[~dat["pop"].isna()]
        dat["log_pop"] = np.log(dat["pop"])
        return dat
