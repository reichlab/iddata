import datetime
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata.constants import S3_DATA_RAW_URL
from iddata.enums import AggLevel, SourceType
from iddata.sources.base import DataSource
from iddata.utils import load_fips_mappings


class ILINetDataSource(DataSource):
    source_name = SourceType.ILINET


    def __init__(self, scale_to_positive: bool = True, agg_level: AggLevel = AggLevel.STATE):
        self.scale_to_positive = scale_to_positive
        self.agg_level = agg_level


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load ILINet data. as_of is accepted but ignored (no versioned snapshots).
        Returns all granularities (state, national, region) with FIPS location codes.
        """
        if as_of is not None:
                raise NotImplementedError(
                    f"ILINet does not support versioned data; static data will be loaded")
        
        files = [urljoin(S3_DATA_RAW_URL, "influenza-ilinet/ilinet.csv"),
                 urljoin(S3_DATA_RAW_URL, "influenza-ilinet/ilinet_hhs.csv"),
                 urljoin(S3_DATA_RAW_URL, "influenza-ilinet/ilinet_state.csv")]
        dat = pd.concat([pd.read_csv(f, encoding="ISO-8859-1", engine="python") for f in files], axis=0)
        # ILINet data is reported as a rate; use unweighted ILI for states, weighted for others
        # Functionality for using raw `ilitotal` counts (never used in practice) has been removed
        dat["inc"] = np.where(dat["region_type"] == "States", dat["unweighted_ili"], dat["weighted_ili"])
        dat["wk_end_date"] = pd.to_datetime(dat["week_start"]) + pd.Timedelta(6, "days")
        dat = dat[["region_type", "region", "year", "week", "season", "season_week", "wk_end_date", "inc"]]
        dat.rename(columns={"region_type": "agg_level", "region": "location"}, inplace=True)
        dat["agg_level"] = np.where(dat["agg_level"] == "National", "national", dat["agg_level"].str[:-1].str.lower())
        dat = dat.sort_values(by=["season", "season_week"])

        # drop out-of-season weeks for early seasons
        early_seasons = [str(yyyy) + "/" + str(yyyy + 1)[2:] for yyyy in range(1997, 2002)]
        first_report_season = ["2002/03"]
        dat = dat[
            (dat.season.isin(early_seasons) & dat.season_week.isin(range(10, 43)))
            | (dat.season.isin(first_report_season) & dat.season_week.isin(range(10, 53)))
            | (~dat.season.isin(early_seasons + first_report_season))
            ]
        # region 10 data prior to 2010/11 is bad, drop it
        dat = dat[~((dat["location"] == "Region 10") & (dat["season"] < "2010/11"))]

        if self.scale_to_positive:
            nrevss = self._load_who_nrevss_positive()
            dat = pd.merge(left=dat, right=nrevss, how="left", on=["agg_level", "location", "season", "season_week"])
            dat["inc"] = dat["inc"] * dat["percent_positive"] / 100.0
            dat.drop("percent_positive", axis=1, inplace=True)

        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]

        dat["source"] = SourceType.ILINET.value

        dat = self._aggregate_to_fips(dat)
        return dat


    def _load_who_nrevss_positive(self) -> pd.DataFrame:
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, "influenza-who-nrevss/who-nrevss.csv"),
                          encoding="ISO-8859-1", engine="python")
        dat = dat[["region_type", "region", "year", "week", "season", "season_week", "percent_positive"]]
        dat.rename(columns={"region_type": "agg_level", "region": "location"}, inplace=True)
        dat["agg_level"] = np.where(dat["agg_level"] == "National", "national", dat["agg_level"].str[:-1].str.lower())
        return dat


    def _aggregate_to_fips(self, dat: pd.DataFrame) -> pd.DataFrame:
        # aggregate ilinet sites in New York to state level, mainly to facilitate adding populations
        fips_mappings = load_fips_mappings()
        ilinet_nonstates = ["National", "Region 1", "Region 2", "Region 3", "Region 4", "Region 5", "Region 6",
                            "Region 7", "Region 8", "Region 9", "Region 10"]

        df_by_state = (
            dat.loc[(~dat["location"].isin(ilinet_nonstates)) & (dat["location"] != "78")]
            .assign(
                state=lambda x: np.where(
                    x["location"].isin(["New York", "New York City"]), "New York", x["location"]
                )
            )
            .assign(
                state=lambda x: np.where(
                    x["state"] == "Commonwealth of the Northern Mariana Islands",
                    "Northern Mariana Islands",
                    x["state"],
                )
            )
            .merge(fips_mappings.rename(columns={"location": "fips"}), left_on="state", right_on="location_name")
            .groupby(["state", "fips", "season", "season_week", "wk_end_date", "source"])
            .agg(inc=("inc", "mean"))
            .reset_index()
            .drop(columns=["state"])
            .rename(columns={"fips": "location"})
            .assign(agg_level="state")
        )

        df_nonstates = dat.loc[dat["location"].isin(ilinet_nonstates)].copy()
        df_nonstates["location"] = np.where(df_nonstates["location"] == "National", "US", df_nonstates["location"])

        return pd.concat([df_nonstates, df_by_state], axis=0)
