import datetime
import warnings
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from iddata import utils
from iddata.ancillary.base import AncillaryData
from iddata.constants import SMH_DATA_PARQUET_URL
from iddata.enums import AggLevel, Disease, SourceType
from iddata.sources.base import DataSource


class SMHDataSource(DataSource):
    source_name = SourceType.SMH

    def __init__(
        self, disease: Disease = Disease.FLU, agg_level: AggLevel = AggLevel.STATE
    ):
        self.disease = disease
        self.agg_level = agg_level

    def load(
        self,
        as_of: datetime.date | None = None,
        ancillary: list[AncillaryData] | None = None,
    ) -> pd.DataFrame:
        """
        Load SMH weekly hospitalization trajectory predictions. Raises ValueError if as_of is None. Only supports as_of >=
        2025-09-17.
        """
        if as_of is not None: # will be replaced later
            warnings.warn("SMH does not yet support versioned data; static data for round 5 will be loaded")

        valid_diseases = (Disease.FLU)
        if self.disease not in valid_diseases:
            raise ValueError(f"SMH supports {valid_diseases}; got {self.disease}.")

        # FLU vs COVID scenario modeling hub
        if self.disease == Disease.FLU:
            disease_name = "flu"
        # elif self.disease == Disease.COVID:
        #     disease_name = "covid"

        parquet_path = f"{disease_name}_scenario-round5_gz.parquet"
        dat = pd.read_parquet(urljoin(SMH_DATA_PARQUET_URL, parquet_path), engine="pyarrow")

        # get to location codes/FIPS
        origin_horizon = dat[["origin_date", "horizon"]].drop_duplicates()
        origin_horizon["target_end_date"] = pd.to_datetime(origin_horizon["origin_date"]) + pd.to_timedelta(7 * origin_horizon["horizon"], unit="D")
        dat = dat.merge(origin_horizon, how="left", on=["origin_date", "horizon"])
        dat["wk_end_date"] = (
            pd.to_datetime(dat["target_end_date"]) + pd.offsets.Week(weekday=5, n=0)
        ).dt.strftime("%Y-%m-%d")
        dat = dat[
            [
                "model_id",
                "scenario_id",
                "location",
                "wk_end_date",
                "output_type_id",
                "value",
            ]
        ].rename(columns={"value": "inc"})

        dat = utils.add_season_columns(dat)

        # merge with populations
        if ancillary:
            for anc in ancillary:
                anc_df = anc.load()
                join_keys = ["location", "season"] if "season" in anc_df.columns else ["location"]
                dat = dat.merge(anc_df, how="left", on=join_keys)

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
        dat["location"] = "syn-" + dat["location"]
        dat["season"] = (
            dat["season"] + dat["scenario_id"].str[0] + "-" + dat["output_type_id"]
        )
        dat["source"] = SourceType.SMH.value + "-" + dat["model_id"]

        cols = ["agg_level", "location", "season", "season_week", "wk_end_date", "inc", "source"]
        if "pop" in dat.columns:
            cols += ["pop", "log_pop"]
        dat = dat[cols]
        return dat
