import datetime
from itertools import product
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pymmwr
import s3fs

from iddata import utils


class DiseaseDataLoader():
  def __init__(self) -> None:
    self.data_raw = "https://infectious-disease-data.s3.amazonaws.com/data-raw/"

  def _construct_data_raw_url(self, relative_path):
    return urljoin(self.data_raw, relative_path)

  def load_fips_mappings(self):
    return pd.read_csv(self._construct_data_raw_url("fips-mappings/fips_mappings.csv"))


  def load_flusurv_rates_2022_23(self):
    dat = pd.read_csv(self._construct_data_raw_url("influenza-flusurv/flusurv-rates/flusurv-rates-2022-23.csv"),
                      encoding="ISO-8859-1",
                      engine="python")
    dat.columns = dat.columns.str.lower()
    
    dat = dat.loc[(dat["age category"] == "Overall") &
                  (dat["sex category"] == "Overall") &
                  (dat["race category"] == "Overall")]
    
    dat = dat.loc[~((dat.catchment == "Entire Network") &
                    (dat.network != "FluSurv-NET"))]

    dat["location"] = dat["catchment"]
    dat["agg_level"] = np.where(dat["location"] == "Entire Network", "national", "site")
    dat["season"] = dat["year"].str.replace("-", "/")
    epiweek = dat["mmwr-year"].astype(str) + dat["mmwr-week"].astype(str)
    dat["season_week"] = utils.convert_epiweek_to_season_week(epiweek)
    dat["wk_end_date"] = dat.apply(
      lambda x: pymmwr.epiweek_to_date(pymmwr.Epiweek(year=x["mmwr-year"],
                                                      week=x["mmwr-week"],
                                                      day=7))
                                      .strftime("%Y-%m-%d"),
        axis=1)
    dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
    dat["inc"] = dat["weekly rate "]
    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
    
    return dat


  def load_flusurv_rates_base(self, 
                              seasons=None,
                              locations=["California", "Colorado", "Connecticut", "Entire Network",
                                        "Georgia", "Maryland", "Michigan", "Minnesota", "New Mexico",
                                        "New York - Albany", "New York - Rochester", "Ohio", "Oregon",
                                        "Tennessee", "Utah"],
                              age_labels=["0-4 yr", "5-17 yr", "18-49 yr", "50-64 yr", "65+ yr", "Overall"]
                              ):
    # read flusurv data and do some minimal preprocessing
    dat = pd.read_csv(self._construct_data_raw_url("influenza-flusurv/flusurv-rates/old-flusurv-rates.csv"),
                      encoding="ISO-8859-1",
                      engine="python")
    dat.columns = dat.columns.str.lower()
    dat["season"] = dat.sea_label.str.replace("-", "/")
    dat["inc"] = dat.weeklyrate
    dat["location"] = dat["region"]
    dat["agg_level"] = np.where(dat["location"] == "Entire Network", "national", "site")
    dat = dat[dat.age_label.isin(age_labels)]
    
    dat = dat.sort_values(by=["wk_end"])
    
    dat["wk_end_date"] = pd.to_datetime(dat["wk_end"])
    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
    
    # add in data from 2022/23 season
    dat = pd.concat(
      [dat, self.load_flusurv_rates_2022_23()],
      axis = 0
    )
    
    dat = dat[dat.location.isin(locations)]
    if seasons is not None:
      dat = dat[dat.season.isin(seasons)]
    
    dat["source"] = "flusurvnet"
    
    return dat


  def load_one_us_census_file(self, f):
    dat = pd.read_csv(f, engine="python", dtype={"STATE": str})
    dat = dat.loc[(dat["NAME"] == "United States") | (dat["STATE"] != "00"),
                  (dat.columns == "STATE") | (dat.columns.str.startswith("POPESTIMATE"))]
    dat = dat.melt(id_vars = "STATE", var_name="season", value_name="pop")
    dat.rename(columns={"STATE": "location"}, inplace=True)
    dat.loc[dat["location"] == "00", "location"] = "US"
    dat["season"] = dat["season"].str[-4:]
    dat["season"] = dat["season"] + "/" + (dat["season"].str[-2:].astype(int) + 1).astype(str)
    
    return dat


  def load_us_census(self, fillna = True):
    files = [
      self._construct_data_raw_url("us-census/nst-est2019-alldata.csv"),
      self._construct_data_raw_url("us-census/NST-EST2023-ALLDATA.csv")]
    us_pops = pd.concat([self.load_one_us_census_file(f) for f in files], axis=0)
    
    fips_mappings = pd.read_csv(self._construct_data_raw_url("fips-mappings/fips_mappings.csv"))
    
    hhs_pops = us_pops.query("location != 'US'") \
      .merge(
          fips_mappings.query("location != 'US'") \
              .assign(hhs_region=lambda x: "Region " + x["hhs_region"].astype(int).astype(str)),
          on="location",
          how = "left"
      ) \
      .groupby(["hhs_region", "season"]) \
      ["pop"] \
      .sum() \
      .reset_index() \
      .rename(columns={"hhs_region": "location"})
    
    dat = pd.concat([us_pops, hhs_pops], axis=0)
    
    if fillna:
      all_locations = dat["location"].unique()
      all_seasons = [str(y) + "/" + str(y+1)[-2:] for y in range(1997, 2025)]
      full_result = pd.DataFrame.from_records(product(all_locations, all_seasons))
      full_result.columns = ["location", "season"]
      dat = full_result.merge(dat, how="left", on=["location", "season"]) \
        .set_index("location") \
        .groupby(["location"]) \
        .bfill() \
        .groupby(["location"]) \
        .ffill() \
        .reset_index()
    
    return dat


  def load_hosp_burden(self):
    burden_estimates = pd.read_csv(
      self._construct_data_raw_url("burden-estimates/burden-estimates.csv"),
      engine="python")

    burden_estimates.columns = ["season", "hosp_burden"]

    burden_estimates["season"] = burden_estimates["season"].str[:4] + "/" + burden_estimates["season"].str[7:9]

    return burden_estimates


  def calc_hosp_burden_adj(self):
    dat = self.load_flusurv_rates_base(
      seasons = ["20" + str(yy) + "/" + str(yy+1) for yy in range(10, 23)],
      locations= ["Entire Network"],
      age_labels = ["Overall"]
    )

    burden_adj = dat[dat.location == "Entire Network"] \
      .groupby("season")["inc"] \
      .sum()
    burden_adj = burden_adj.reset_index()
    burden_adj.columns = ["season", "cum_rate"]

    us_census = self.load_us_census().query("location == 'US'").drop("location", axis=1)
    burden_adj = pd.merge(burden_adj, us_census, on="season")

    burden_estimates = self.load_hosp_burden()
    burden_adj = pd.merge(burden_adj, burden_estimates, on="season")

    burden_adj["reported_burden_est"] = burden_adj["cum_rate"] * burden_adj["pop"] / 100000
    burden_adj["adj_factor"] = burden_adj["hosp_burden"] / burden_adj["reported_burden_est"]

    return burden_adj


  def fill_missing_flusurv_dates_one_location(self, location_df):
    df = location_df.set_index("wk_end_date") \
      .asfreq("W-sat") \
      .reset_index()
    fill_cols = ["agg_level", "location", "season", "pop", "source"]
    fill_cols = [c for c in fill_cols if c in df.columns]
    df[fill_cols] = df[fill_cols].fillna(axis=0, method="ffill")
    return df


  def load_flusurv_rates(self,
                         burden_adj=True,
                         locations=["California", "Colorado", "Connecticut", "Entire Network",
                                    "Georgia", "Maryland", "Michigan", "Minnesota", "New Mexico",
                                    "New York - Albany", "New York - Rochester", "Ohio", "Oregon",
                                    "Tennessee", "Utah"]
                        ):
    # read flusurv data and do some minimal preprocessing
    dat = self.load_flusurv_rates_base(
      seasons = ["20" + str(yy) + "/" + str(yy+1) for yy in range(10, 23)],
      locations = locations,
      age_labels = ["Overall"]
    )
    
    # if requested, make adjustments for overall season burden
    if burden_adj:
      hosp_burden_adj = self.calc_hosp_burden_adj()
      dat = pd.merge(dat, hosp_burden_adj, on="season")
      dat["inc"] = dat["inc"] * dat["adj_factor"]
    
    # fill in missing dates
    gd = dat.groupby("location")
    
    dat = pd.concat(
      [self.fill_missing_flusurv_dates_one_location(df) for _, df in gd],
      axis = 0)
    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc", "source"]]
    
    return dat


  def load_who_nrevss_positive(self):
    dat = pd.read_csv(self._construct_data_raw_url("influenza-who-nrevss/who-nrevss.csv"),
                      encoding="ISO-8859-1",
                      engine="python")
    dat = dat[["region_type", "region", "year", "week", "season", "season_week", "percent_positive"]]
    
    dat.rename(columns={"region_type": "agg_level", "region": "location"},
              inplace=True)
    dat["agg_level"] = np.where(dat["agg_level"] == "National",
                                "national",
                                dat["agg_level"].str[:-1].str.lower())
    return dat


  def load_ilinet(self,
                  response_type="rate",
                  scale_to_positive=True,
                  drop_pandemic_seasons=True,
                  burden_adj=False):
    # read ilinet data and do some minimal preprocessing
    files = [self._construct_data_raw_url("influenza-ilinet/ilinet.csv"),
             self._construct_data_raw_url("influenza-ilinet/ilinet_hhs.csv"),
             self._construct_data_raw_url("influenza-ilinet/ilinet_state.csv")]
    dat = pd.concat(
      [ pd.read_csv(f, encoding="ISO-8859-1", engine="python") for f in files ],
      axis = 0)
    
    if response_type == "rate":
      dat["inc"] = np.where(dat["region_type"] == "States",
                            dat["unweighted_ili"],
                            dat["weighted_ili"])
    else:
      dat["inc"] = dat.ilitotal

    dat["wk_end_date"] = pd.to_datetime(dat["week_start"]) + pd.Timedelta(6, "days")
    dat = dat[["region_type", "region", "year", "week", "season", "season_week", "wk_end_date", "inc"]]
    
    dat.rename(columns={"region_type": "agg_level", "region": "location"},
              inplace=True)
    dat["agg_level"] = np.where(dat["agg_level"] == "National",
                                "national",
                                dat["agg_level"].str[:-1].str.lower())
    dat = dat.sort_values(by=["season", "season_week"])
    
    # for early seasons, drop out-of-season weeks with no reporting
    early_seasons = [str(yyyy) + "/" + str(yyyy + 1)[2:] for yyyy in range(1997, 2002)]
    early_in_season_weeks = [w for w in range(10, 43)]
    first_report_season = ["2002/03"]
    first_report_in_season_weeks = [w for w in range(10, 53)]
    dat = dat[
      (dat.season.isin(early_seasons) & dat.season_week.isin(early_in_season_weeks)) |
      (dat.season.isin(first_report_season) & dat.season_week.isin(first_report_in_season_weeks)) |
      (~dat.season.isin(early_seasons + first_report_season))]
    
    # region 10 data prior to 2010/11 is bad, drop it
    dat = dat[
      ~((dat["location"] == "Region 10") & (dat["season"] < "2010/11"))
    ]
    
    if scale_to_positive:
      dat = pd.merge(
        left=dat,
        right=self.load_who_nrevss_positive(),
        how="left",
        on=["agg_level", "location", "season", "season_week"])
      dat["inc"] = dat["inc"] * dat["percent_positive"] / 100.0
      dat.drop("percent_positive", axis=1)

    if drop_pandemic_seasons:
      dat.loc[dat["season"].isin(["2008/09", "2009/10", "2020/21", "2021/22"]), "inc"] = np.nan

    # if requested, make adjustments for overall season burden
    # if burden_adj:
    #   hosp_burden_adj = calc_hosp_burden_adj()
    #   dat = pd.merge(dat, hosp_burden_adj, on='season')
    #   dat['inc'] = dat['inc'] * dat['adj_factor']

    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
    dat["source"] = "ilinet"
    return dat


  def load_nhsn(self, disease="flu", rates=True, drop_pandemic_seasons=True, as_of=None):
    if as_of is None:
      as_of = datetime.date.today()
    
    if isinstance(as_of, str):
      as_of = datetime.date.fromisoformat(as_of)

    if as_of < datetime.date.fromisoformat("2024-11-15"):
      if not drop_pandemic_seasons:
        raise NotImplementedError("Functionality for loading all seasons of NHSN data with specified as_of date is not implemented.")
      
      if disease != "flu":
        raise NotImplementedError(f"When loading NHSN data with an as_of date prior to 2024-11-15, only disease='flu' is supported; got {str(disease)}.")
      return self.load_nhsn_from_hhs(rates=rates, as_of=as_of)
    else:
      return self.load_nhsn_from_nhsn(
        disease=disease,
        rates=rates,
        as_of=as_of,
        drop_pandemic_seasons=drop_pandemic_seasons
      )


  def load_nhsn_from_hhs(self, rates=True, as_of=None):
    # find the largest stored file dated on or before the as_of date
    as_of_file_path = f"influenza-hhs/hhs-{str(as_of)}.csv"
    glob_results = s3fs.S3FileSystem(anon=True) \
        .glob("infectious-disease-data/data-raw/influenza-hhs/hhs-????-??-??.csv")
    all_file_paths = sorted([f[len("infectious-disease-data/data-raw/"):] for f in glob_results])
    all_file_paths = [f for f in all_file_paths if f <= as_of_file_path]
    file_path = all_file_paths[-1]
    
    dat = pd.read_csv(self._construct_data_raw_url(file_path))
    dat.rename(columns={"date": "wk_end_date"}, inplace=True)

    ew_str = dat.apply(utils.date_to_ew_str, axis=1)
    dat["season"] = utils.convert_epiweek_to_season(ew_str)
    dat["season_week"] = utils.convert_epiweek_to_season_week(ew_str)
    dat = dat.sort_values(by=["season", "season_week"])
    
    if rates:
      pops = self.load_us_census()
      dat = dat.merge(pops, on = ["location", "season"], how="left") \
        .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

    dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
    
    dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
    dat["source"] = "nhsn"
    return dat


  def load_nhsn_from_nhsn(self, disease="flu", rates=True, as_of=None, drop_pandemic_seasons=True):
    valid_diseases = ["flu", "covid"]
    if disease not in valid_diseases:
        raise ValueError("For NHSN data, the only supported diseases are 'flu' and 'covid'.")
    
    # find the largest stored file dated on or before the as_of date
    as_of_file_path = f"influenza-nhsn/nhsn-{str(as_of)}.csv"
    glob_results = s3fs.S3FileSystem(anon=True) \
        .glob("infectious-disease-data/data-raw/influenza-nhsn/nhsn-????-??-??.csv")
    all_file_paths = sorted([f[len("infectious-disease-data/data-raw/"):] for f in glob_results])
    all_file_paths = [f for f in all_file_paths if f <= as_of_file_path]
    file_path = all_file_paths[-1]
    
    dat = pd.read_csv(self._construct_data_raw_url(file_path))
    if disease == "flu":
        inc_colname = "Total Influenza Admissions"
    elif disease == "covid":
        inc_colname = "Total COVID-19 Admissions"
    dat = dat[["Geographic aggregation", "Week Ending Date"] + [inc_colname]]
    dat.columns = ["abbreviation", "wk_end_date", "inc"]
    
    # rename USA to US
    dat.loc[dat.abbreviation == "USA", "abbreviation"] = "US"
    
    # get to location codes/FIPS
    fips_mappings = self.load_fips_mappings()
    dat = dat.merge(fips_mappings, on=["abbreviation"], how="left")
    
    ew_str = dat.apply(utils.date_to_ew_str, axis=1)
    dat["season"] = utils.convert_epiweek_to_season(ew_str)
    dat["season_week"] = utils.convert_epiweek_to_season_week(ew_str)
    dat = dat.sort_values(by=["season", "season_week"])

    if drop_pandemic_seasons:
      dat.loc[dat["season"].isin(["2020/21", "2021/22"]), "inc"] = np.nan
    
    if rates:
      pops = self.load_us_census()
      dat = dat.merge(pops, on = ["location", "season"], how="left") \
        .assign(inc=lambda x: x["inc"] / x["pop"] * 100000)

    dat["wk_end_date"] = pd.to_datetime(dat["wk_end_date"])
    
    dat["agg_level"] = np.where(dat["location"] == "US", "national", "state")
    dat = dat[["agg_level", "location", "season", "season_week", "wk_end_date", "inc"]]
    dat["source"] = "nhsn"
    return dat

  def load_agg_transform_ilinet(self, fips_mappings, **ilinet_kwargs):
    df_ilinet_full = self.load_ilinet(**ilinet_kwargs)
    # df_ilinet_full.loc[df_ilinet_full['inc'] < np.exp(-7), 'inc'] = np.exp(-7)
    df_ilinet_full["inc"] = (df_ilinet_full["inc"] + np.exp(-7)) * 4
    
    # aggregate ilinet sites in New York to state level,
    # mainly to facilitate adding populations
    ilinet_nonstates = ["National", "Region 1", "Region 2", "Region 3",
                        "Region 4", "Region 5", "Region 6", "Region 7",
                        "Region 8", "Region 9", "Region 10"]
    df_ilinet_by_state = df_ilinet_full \
      .loc[(~df_ilinet_full["location"].isin(ilinet_nonstates)) &
          (df_ilinet_full["location"] != "78")] \
      .assign(state = lambda x: np.where(x["location"].isin(["New York", "New York City"]),
                                        "New York",
                                        x["location"])) \
      .assign(state = lambda x: np.where(x["state"] == "Commonwealth of the Northern Mariana Islands",
                                        "Northern Mariana Islands",
                                        x["state"])) \
      .merge(
        fips_mappings.rename(columns={"location": "fips"}),
        left_on="state",
        right_on="location_name") \
      .groupby(["state", "fips", "season", "season_week", "wk_end_date", "source"]) \
      .apply(lambda x: pd.DataFrame({"inc": [np.mean(x["inc"])]})) \
      .reset_index() \
      .drop(columns = ["state", "level_6"]) \
      .rename(columns = {"fips": "location"}) \
      .assign(agg_level = "state")
    
    df_ilinet_nonstates = df_ilinet_full.loc[df_ilinet_full["location"].isin(ilinet_nonstates)].copy()
    df_ilinet_nonstates["location"] = np.where(df_ilinet_nonstates["location"] == "National",
                                              "US",
                                              df_ilinet_nonstates["location"])
    df_ilinet = pd.concat(
      [df_ilinet_nonstates, df_ilinet_by_state],
      axis = 0)
    
    return df_ilinet


  def load_agg_transform_flusurv(self, fips_mappings, **flusurvnet_kwargs):
    df_flusurv_by_site = self.load_flusurv_rates(**flusurvnet_kwargs)
    # df_flusurv_by_site.loc[df_flusurv_by_site['inc'] < np.exp(-3), 'inc'] = np.exp(-3)
    df_flusurv_by_site["inc"] = (df_flusurv_by_site["inc"] + np.exp(-3)) / 2.5
    
    # aggregate flusurv sites in New York to state level,
    # mainly to facilitate adding populations
    df_flusurv_by_state = df_flusurv_by_site \
      .loc[df_flusurv_by_site["location"] != "Entire Network"] \
      .assign(state = lambda x: np.where(x["location"].isin(["New York - Albany", "New York - Rochester"]),
                                        "New York",
                                        x["location"])) \
      .merge(
        fips_mappings.rename(columns={"location": "fips"}),
        left_on="state",
        right_on="location_name") \
      .groupby(["fips", "season", "season_week", "wk_end_date", "source"]) \
      .apply(lambda x: pd.DataFrame({"inc": [np.mean(x["inc"])]})) \
      .reset_index() \
      .drop(columns = ["level_5"]) \
      .rename(columns = {"fips": "location"}) \
      .assign(agg_level = "state")
    
    df_flusurv_us = df_flusurv_by_site.loc[df_flusurv_by_site["location"] == "Entire Network"].copy()
    df_flusurv_us["location"] = "US"
    df_flusurv = pd.concat(
      [df_flusurv_us, df_flusurv_by_state],
      axis = 0)
    
    return df_flusurv


  def load_data(self, sources=None, flusurvnet_kwargs=None, nhsn_kwargs=None, ilinet_kwargs=None,
                power_transform="4rt"):
    """
    Load influenza data and transform to a scale suitable for input to models.

    Parameters
    ----------
    sources: None or list of sources
        data sources to collect. Defaults to ['flusurvnet', 'nhsn', 'ilinet'].
        If provided as a list, must be a subset of the defaults.
    flusurvnet_kwargs: dictionary of keyword arguments to pass on to `load_flusurv_rates`
    nhsn_kwargs: dictionary of keyword arguments to pass on to `load_nhsn`
    ilinet_kwargs: dictionary of keyword arguments to pass on to `load_ilinet`
    power_transform: string specifying power transform to use: '4rt' or `None`

    Returns
    -------
    Pandas DataFrame
    """
    if sources is None:
        sources = ["flusurvnet", "nhsn", "ilinet"]
    
    if flusurvnet_kwargs is None:
        flusurvnet_kwargs = {}
    
    if nhsn_kwargs is None:
        nhsn_kwargs = {}
    
    if ilinet_kwargs is None:
        ilinet_kwargs = {}
    
    if power_transform not in ["4rt", None]:
        raise ValueError('Only None and "4rt" are supported for the power_transform argument.')
    
    us_census = self.load_us_census()
    fips_mappings = pd.read_csv(self._construct_data_raw_url("fips-mappings/fips_mappings.csv"))
    
    if "nhsn" in sources:
        df_nhsn = self.load_nhsn(**nhsn_kwargs)
        df_nhsn["inc"] = df_nhsn["inc"] + 0.75**4
    else:
        df_nhsn = None
    
    if "ilinet" in sources:
        df_ilinet = self.load_agg_transform_ilinet(fips_mappings=fips_mappings, **ilinet_kwargs)
    else:
        df_ilinet = None
    
    if "flusurvnet" in sources:
        df_flusurv = self.load_agg_transform_flusurv(fips_mappings=fips_mappings, **flusurvnet_kwargs)
    else:
        df_flusurv = None
    
    df = pd.concat(
        [df_nhsn, df_ilinet, df_flusurv],
        axis=0).sort_values(["source", "location", "wk_end_date"])
    
    # log population
    df = df.merge(us_census, how="left", on=["location", "season"])
    df["log_pop"] = np.log(df["pop"])
    
    # process response variable:
    # - fourth root transform to stabilize variability
    # - divide by location- and source- specific 95th percentile
    # - center relative to location- and source- specific mean
    #   (note non-standard order of center/scale)
    if power_transform is None:
        df["inc_trans"] = df["inc"] + 0.01
    elif power_transform == "4rt":
        df["inc_trans"] = (df["inc"] + 0.01)**0.25
    
    df["inc_trans_scale_factor"] = df \
        .assign(
            inc_trans_in_season = lambda x: np.where((x["season_week"] < 10) | (x["season_week"] > 45),
                                                     np.nan,
                                                     x["inc_trans"])) \
        .groupby(["source", "location"])["inc_trans_in_season"] \
        .transform(lambda x: x.quantile(0.95))
    
    df["inc_trans_cs"] = df["inc_trans"] / (df["inc_trans_scale_factor"] + 0.01)
    df["inc_trans_center_factor"] = df \
        .assign(
            inc_trans_cs_in_season = lambda x: np.where((x["season_week"] < 10) | (x["season_week"] > 45),
                                                        np.nan,
                                                        x["inc_trans_cs"])) \
        .groupby(["source", "location"])["inc_trans_cs_in_season"] \
        .transform(lambda x: x.mean())
    df["inc_trans_cs"] = df["inc_trans_cs"] - df["inc_trans_center_factor"]
    
    return(df)