import datetime
import os
from collections.abc import Callable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from epidatpy import EpiDataContext

from iddata import utils
from iddata.ancillary.population import _load_us_census
from iddata.constants import S3_DATA_RAW_URL
from iddata.enums import DataBackend, Disease, SourceType
from iddata.s3 import get_versioned_file_path
from iddata.sources.base import DataSource
from iddata.utils import load_fips_mappings

_HHS_TO_NHSN_CUTOVER = datetime.date(2024, 11, 15)  # first influenza-nhsn/ S3 snapshot
_EPIDATA_FIRST_VERSION = datetime.date(2024, 11, 19)  # first V5 report_time for source=nhsn

_EPIDATA_SIGNALS = {
    Disease.FLU: "confirmed_admissions_flu_ew",
    Disease.COVID: "confirmed_admissions_covid_ew",
}


class NHSNDataSource(DataSource):
    """
    NHSN hospital admissions. Two backends are available:

    - `DataBackend.S3` (default): weekly snapshots in the `infectious-disease-data` S3 bucket.
    - `DataBackend.EPIDATA`: CMU Delphi's Epidata V5 API. Requires the DELPHI_EPIDATA_KEY environment variable. Delphi
      has no NHSN versions before 2024-11-19, so as_of dates before then fall back to S3.
    """
    source_name = SourceType.NHSN


    def __init__(self, disease: Disease = Disease.FLU, rates: bool = True,
                 backend: DataBackend = DataBackend.S3, epidata_ctx: EpiDataContext | None = None):
        """
        :param backend: where to load data from (see class docstring). Strings such as "epidata" are accepted.
        :param epidata_ctx: optional epidatpy context for the EPIDATA backend, e.g. `EpiDataContext(use_cache=True)`.
            If None, a default context is created the first time Delphi is queried.
        """
        self.disease = disease
        self.rates = rates
        self.backend = DataBackend(backend)
        self.epidata_ctx = epidata_ctx
        if self.backend == DataBackend.EPIDATA and disease not in _EPIDATA_SIGNALS:
            raise ValueError(f"NHSN epidata backend supports only Disease.FLU and Disease.COVID; got {disease}.")


    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        """
        Load NHSN hospitalization data. Raises ValueError if as_of is None. With the S3 backend, routes to the HHS
        archive for as_of < 2024-11-15, or the NHSN snapshots for later dates. With the EPIDATA backend, as_of >=
        2024-11-19 is loaded from Delphi Epidata, and earlier dates fall back to the S3 routing.
        """
        if as_of is None:
            raise ValueError("NHSN requires as_of to be specified.")

        if isinstance(as_of, str):
            as_of = datetime.date.fromisoformat(as_of)
        loader_fn = self._loader_for(as_of)
        if loader_fn == self._load_from_hhs and self.disease != Disease.FLU:
            raise NotImplementedError(
                f"NHSN only supports disease='flu' for an as_of date prior to 2024-11-15; got {self.disease}.")
        dat = loader_fn(as_of)

        dat = utils.add_season_columns(dat)

        if self.rates:
            pops = _load_us_census(as_of)
            dat = dat.merge(pops[["location", "season", "pop"]], on=["location", "season"], how="left") \
                .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

        dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
        dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
        dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
        dat["source"] = SourceType.NHSN.value
        return dat


    def _loader_for(self, as_of: datetime.date) -> Callable[[datetime.date], pd.DataFrame]:
        """Pick the raw loader for as_of. With backend=EPIDATA, dates before _EPIDATA_FIRST_VERSION fall back to S3."""
        if self.backend == DataBackend.EPIDATA and as_of >= _EPIDATA_FIRST_VERSION:
            return self._load_from_epidata
        if as_of < _HHS_TO_NHSN_CUTOVER:
            return self._load_from_hhs
        return self._load_from_nhsn


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
            raise ValueError("NHSN supports only Disease.FLU and Disease.COVID.")

        file_path = get_versioned_file_path(
            "infectious-disease-data/data-raw/influenza-nhsn/nhsn-????-??-??.csv",
            as_of)
        dat = pd.read_csv(urljoin(S3_DATA_RAW_URL, file_path))

        if self.disease == Disease.FLU:
            inc_colname = "Total Influenza Admissions"
        elif self.disease == Disease.COVID:
            inc_colname = "Total COVID-19 Admissions"
        # snapshots from 2025-09-10 on include HHS region rows ("Region 1" ... "Region 10"), which have no FIPS code
        dat = dat[~dat["Geographic aggregation"].str.startswith("Region")]
        # get to location codes/FIPS
        dat = (
            dat[["Geographic aggregation", "Week Ending Date", inc_colname]]
            .rename(columns={"Geographic aggregation": "abbreviation", "Week Ending Date": "wk_end_date", inc_colname: "inc"})
            .replace({"abbreviation": {"USA": "US"}})
            .merge(load_fips_mappings(), on="abbreviation", how="left")
        )
        return dat[["location", "wk_end_date", "inc"]]


    def _get_epidata_ctx(self) -> EpiDataContext:
        if self.epidata_ctx is None:
            if not os.environ.get("DELPHI_EPIDATA_KEY"):
                raise RuntimeError(
                    "NHSNDataSource(backend='epidata') requires the DELPHI_EPIDATA_KEY environment variable. "
                    "Request a free key at https://api.delphi.cmu.edu/epidata/admin/registration_form")
            self.epidata_ctx = EpiDataContext()  # epidatpy reads DELPHI_EPIDATA_KEY itself
        return self.epidata_ctx


    def _load_from_epidata(self, as_of: datetime.date) -> pd.DataFrame:
        """Returns raw DataFrame with location, wk_end_date, inc columns."""
        dat = self._get_epidata_ctx().epidata_snapshot(
            source="nhsn", signals=_EPIDATA_SIGNALS[self.disease], geo_type=["nation", "state"],
            snapshot_date=as_of, fill_method="source").df()
        if dat.empty:
            raise ValueError(f"Delphi Epidata returned no NHSN rows for snapshot_date={as_of}.")
        # reference_time is a tz-naive Saturday. format it as YYYY-MM-DD like the S3 CSVs so that load()'s
        # pd.to_datetime gives both backends the same datetime resolution. the nullable Float64 value becomes float64
        dat = (
            dat.assign(abbreviation=dat["geo_value"].str.upper(),
                       wk_end_date=dat["reference_time"].dt.strftime("%Y-%m-%d"),
                       inc=dat["value"].astype(float))
            .merge(load_fips_mappings(), on="abbreviation", how="left")
        )
        return dat[["location", "wk_end_date", "inc"]]
