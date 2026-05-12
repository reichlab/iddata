import datetime
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pymmwr

from iddata import utils
from iddata.ancillary.population import _load_us_census
from iddata.constants import S3_DATA_RAW_URL
from iddata.enums import AggLevel, SourceType
from iddata.sources.base import DataSource
from iddata.utils import load_fips_mappings


class FluSurvNetDataSource(DataSource):
    source_name = SourceType.FLUSURVNET

    _DEFAULT_LOCATIONS = ["California", "Colorado", "Connecticut", "Entire Network", "Georgia", "Maryland", "Michigan",
                          "Minnesota", "New Mexico", "New York - Albany", "New York - Rochester", "Ohio", "Oregon",
                          "Tennessee", "Utah"]


    def __init__(self, burden_adj: bool = True, agg_level: AggLevel = AggLevel.STATE,
                 locations: list[str] | None = None):
        self.burden_adj = burden_adj
        self.agg_level = agg_level
        self.locations = locations if locations is not None else self._DEFAULT_LOCATIONS


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load FluSurv-NET data. as_of is accepted but ignored (no versioned snapshots). Returns data aggregated to
        state-level FIPS codes and national.
        """
        seasons = ["20" + str(yy) + "/" + str(yy + 1) for yy in range(10, 23)]
        dat = self._load_base(seasons=seasons, locations=self.locations)

        if self.burden_adj:
            hosp_burden_adj = self._calc_hosp_burden_adj()
            dat = pd.merge(dat, hosp_burden_adj, on="season")
            dat["inc"] = dat["inc"] * dat["adj_factor"]

        # fill missing dates per location
        gd = dat.groupby("location")
        dat = pd.concat([self._fill_missing_dates(df) for _, df in gd], axis=0)
        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc", "source"]]

        # Apply FluSurvNet-specific scaling
        dat["inc"] = (dat["inc"] + np.exp(-3)) / 2.5

        # Aggregate to FIPS codes
        dat = self._aggregate_to_fips(dat)

        dat["source"] = SourceType.FLUSURVNET.value
        return dat


    def _load_base(self, seasons=None, locations=None, age_labels=None) -> pd.DataFrame:
        if age_labels is None:
            age_labels = ["Overall"]
        dat_old = pd.read_csv(urljoin(S3_DATA_RAW_URL, "influenza-flusurv/flusurv-rates/old-flusurv-rates.csv"),
                              encoding="ISO-8859-1", engine="python")
        dat_old.columns = dat_old.columns.str.lower()
        dat_old["season"] = dat_old["sea_label"].str.replace("-", "/")
        dat_old["inc"] = dat_old["weeklyrate"]
        dat_old["location"] = dat_old["region"]
        dat_old["agg_level"] = np.where(dat_old["location"] == "Entire Network", "national", "site")
        dat_old = dat_old[dat_old["age_label"].isin(age_labels)]
        dat_old["wk_end_date"] = pd.to_datetime(dat_old["wk_end"])
        dat_old = dat_old[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]

        dat_new = pd.read_csv(urljoin(S3_DATA_RAW_URL, "influenza-flusurv/flusurv-rates/flusurv-rates-2022-23.csv"),
                              encoding="ISO-8859-1", engine="python")
        dat_new.columns = dat_new.columns.str.lower()
        dat_new = dat_new.loc[
            (dat_new["age category"] == "Overall")
            & (dat_new["sex category"] == "Overall")
            & (dat_new["race category"] == "Overall")
            ]
        dat_new = dat_new.loc[~((dat_new["catchment"] == "Entire Network") & (dat_new["network"] != "FluSurv-NET"))]
        dat_new["location"] = dat_new["catchment"]
        dat_new["agg_level"] = np.where(dat_new["location"] == "Entire Network", "national", "site")
        dat_new["season"] = dat_new["year"].str.replace("-", "/")
        epiweek = dat_new["mmwr-year"].astype(str) + dat_new["mmwr-week"].astype(str)
        dat_new["season_week"] = utils.convert_epiweek_to_season_week(epiweek)
        dat_new["wk_end_date"] = dat_new.apply(
            lambda x: pymmwr.epiweek_to_date(
                pymmwr.Epiweek(year=x["mmwr-year"], week=x["mmwr-week"], day=7)
            ).strftime("%Y-%m-%d"),
            axis=1,
        )
        dat_new["wk_end_date"] = pd.to_datetime(dat_new["wk_end_date"])
        dat_new["inc"] = dat_new["weekly rate "]
        dat_new = dat_new[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]

        dat = pd.concat([dat_old, dat_new], axis=0)
        if locations is not None:
            dat = dat[dat["location"].isin(locations)]
        if seasons is not None:
            dat = dat[dat["season"].isin(seasons)]
        dat["source"] = SourceType.FLUSURVNET.value
        return dat


    def _calc_hosp_burden_adj(self) -> pd.DataFrame:
        dat = self._load_base(seasons=["20" + str(yy) + "/" + str(yy + 1) for yy in range(10, 23)],
                              locations=["Entire Network"])
        burden_adj = dat[dat["location"] == "Entire Network"].groupby("season")["inc"].sum().reset_index()
        burden_adj.columns = ["season", "cum_rate"]

        us_census = _load_us_census().query("location == 'US'").drop("location", axis=1)
        burden_adj = pd.merge(burden_adj, us_census, on="season")

        burden_estimates = pd.read_csv(urljoin(S3_DATA_RAW_URL, "burden-estimates/burden-estimates.csv"),
                                       engine="python")
        burden_estimates.columns = ["season", "hosp_burden"]
        burden_estimates["season"] = burden_estimates["season"].str[:4] + "/" + burden_estimates["season"].str[7:9]
        burden_adj = pd.merge(burden_adj, burden_estimates, on="season")
        burden_adj["reported_burden_est"] = burden_adj["cum_rate"] * burden_adj["pop"] / 100000
        burden_adj["adj_factor"] = burden_adj["hosp_burden"] / burden_adj["reported_burden_est"]
        return burden_adj


    def _fill_missing_dates(self, location_df: pd.DataFrame) -> pd.DataFrame:
        df = location_df.set_index("wk_end_date").asfreq("W-SAT").reset_index()
        fill_cols = ["agg_level", "location", "season", "source"]
        fill_cols = [c for c in fill_cols if c in df.columns]
        df[fill_cols] = df[fill_cols].ffill()
        return df


    def _aggregate_to_fips(self, dat: pd.DataFrame) -> pd.DataFrame:
        fips_mappings = load_fips_mappings()
        df_by_state = (
            dat.loc[dat["location"] != "Entire Network"]
            .assign(
                state=lambda x: np.where(
                    x["location"].isin(["New York - Albany", "New York - Rochester"]),
                    "New York",
                    x["location"],
                )
            )
            .merge(fips_mappings.rename(columns={"location": "fips"}), left_on="state", right_on="location_name")
            .groupby(["fips", "season", "season_week", "wk_end_date", "source"])
            .agg(inc=("inc", "mean"))
            .reset_index()
            .rename(columns={"fips": "location"})
            .assign(agg_level="state")
        )

        df_national = dat.loc[dat["location"] == "Entire Network"].copy()
        df_national["location"] = "US"
        return pd.concat([df_national, df_by_state], axis=0)
