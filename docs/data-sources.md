# Data sources

iddata does not bundle any data. Every `DataSource` and `AncillaryData` class reads raw files at load time, mostly from a
public S3 bucket maintained by the Reich Lab. This document describes that bucket, how its contents get there, and what
each data source reads.

## The `infectious-disease-data` S3 bucket

- **Bucket:** `s3://infectious-disease-data` (region `us-east-1`)
- **Prefix used by iddata:** `data-raw/`
- **HTTPS base URL:** `https://infectious-disease-data.s3.amazonaws.com/data-raw/` (`S3_DATA_RAW_URL` in
  [`src/iddata/constants.py`](../src/iddata/constants.py))

The bucket is publicly readable, so no AWS credentials are needed to use iddata. Files are read over HTTPS with
`pandas.read_csv`, and snapshot listings use an anonymous `s3fs` client.

### Layout

| Source | Prefix under `data-raw/` | Contents | Used by | How it is updated |
|---|---|---|---|---|
| NHSN | `influenza-nhsn/nhsn-YYYY-MM-DD.csv` | Dated snapshots of NHSN hospital admissions (flu and COVID-19) | `NHSNDataSource` (`as_of >= 2024-11-15`; with the Epidata backend, only `as_of` 2024-11-15 to 2024-11-18) | [`snapshot-nhsn-data.yml`](../.github/workflows/snapshot-nhsn-data.yml), weekly |
| NHSN | `influenza-hhs/hhs-YYYY-MM-DD.csv` | Dated snapshots of the older HHS Protect flu hospital admissions data | `NHSNDataSource` (`as_of < 2024-11-15`) | Historical archive; no workflow in this repo writes to it |
| NSSP | `nssp/nssp-YYYY-MM-DD.csv` | Dated snapshots of NSSP ED visit percentages (flu, COVID-19, RSV) | `NSSPDataSource` | [`snapshot-nssp-data.yml`](../.github/workflows/snapshot-nssp-data.yml), twice weekly |
| ILINet | `influenza-ilinet/` | `ilinet.csv`, `ilinet_hhs.csv`, `ilinet_state.csv` | `ILINetDataSource` | Static, uploaded by hand |
| ILINet | `influenza-who-nrevss/who-nrevss.csv` | WHO/NREVSS percent positive | `ILINetDataSource(scale_to_positive=True)` | Static, uploaded by hand |
| FluSurv-NET | `influenza-flusurv/flusurv-rates/` | `old-flusurv-rates.csv`, `flusurv-rates-2022-23.csv` | `FluSurvNetDataSource` | Static, uploaded by hand |
| FluSurv-NET | `burden-estimates/burden-estimates.csv` | CDC seasonal flu hospitalization burden estimates | `FluSurvNetDataSource(burden_adj=True)` | Static, uploaded by hand |
| Other | `us-census/` | `nst-est2019-alldata.csv`, `NST-EST2023-ALLDATA.csv` (state and national population estimates) | `PopulationData`, NHSN rates, FluSurv-NET burden adjustment | Static, uploaded by hand |
| Other | `fips-mappings/fips_mappings.csv` | Location name / abbreviation / FIPS / HHS region crosswalk | Most sources (`utils.load_fips_mappings`) | Static, uploaded by hand |

## Versioned snapshots and `as_of`

NHSN and NSSP data are revised after first release, so iddata keeps a dated copy of each release and serves whichever
one was available on the requested date. This lets models be backtested on the data they would have seen at the time.

This section describes the S3 snapshots, which are the default for NHSN and the only option for NSSP. NHSN can instead
be versioned by Delphi Epidata. See [NHSN from Delphi Epidata](#nhsn-from-delphi-epidata).

[`get_versioned_file_path`](../src/iddata/s3.py) implements the lookup:

1. List every file matching a glob such as `influenza-nhsn/nhsn-????-??-??.csv`.
2. Keep the files whose date is on or before `as_of`, and return the most recent one.
3. Raise `FileNotFoundError` if none qualify.

Consequences worth knowing:

- The date in a file name is the (UTC) date the snapshot was taken, not the last week of data it contains.
- If a scheduled snapshot fails (for example, because the upstream data was stale), no file is written for that day, and
  lookups for that date fall back to the previous snapshot.
- `as_of` is required for `NHSNDataSource` and `NSSPDataSource`. `NSSPDataSource` only supports `as_of >= 2025-09-17`,
  the first NSSP snapshot.
- ILINet, FluSurv-NET, and the ancillary files are not versioned. Those sources warn and ignore `as_of`.
- SMH projections are not versioned either, but `SMHDataSource` uses `as_of` to decide which rounds to include. See
  [Scenario Modeling Hub projections](#scenario-modeling-hub-projections).

## Snapshot workflows

Both workflows run on GitHub Actions and can also be triggered manually (`workflow_dispatch`).

| Workflow | Schedule (UTC) | Upstream source | Writes to |
|---|---|---|---|
| [`snapshot-nhsn-data.yml`](../.github/workflows/snapshot-nhsn-data.yml) | Wednesdays 17:45 | CDC data portal, dataset [`mpgq-jmmr`](https://data.cdc.gov/resource/mpgq-jmmr.csv) | `influenza-nhsn/nhsn-YYYY-MM-DD.csv` |
| [`snapshot-nssp-data.yml`](../.github/workflows/snapshot-nssp-data.yml) | Wednesdays and Fridays 17:45 | CDC data portal, dataset [`rdmq-nq56`](https://data.cdc.gov/resource/rdmq-nq56.csv) | `nssp/nssp-YYYY-MM-DD.csv` |

Each run:

1. Downloads the dataset with `RSocrata::read.socrata` in R.
2. Keeps only the columns iddata uses and cleans them up: it trims dates to `YYYY-MM-DD`, and for NSSP it zero-pads FIPS
   codes to five digits.
3. Fails without uploading if the latest week in the data is older than the most recent completed week (the previous
   Saturday).
4. Assumes the `iddata-github-action` IAM role through GitHub OIDC and copies the file to S3 with `aws s3 cp`.

`snapshot-nhsn-data.yml` keeps running even though NHSN can also be loaded from Delphi Epidata. S3 is still the default
NHSN backend, and the S3 archive is the only source of versions from before 2024-11-19.

Starting with `nhsn-2025-09-10.csv`, NHSN snapshots also include HHS region rows (`Region 1` to `Region 10`).
`NHSNDataSource` drops these rows. Earlier versions kept them with a missing `location` and labeled them as states.

> **NSSP history:** the Wednesday snapshot captures CDC's preliminary NSSP release, and the Friday snapshot captures the
> official one. Preliminary data used to be hosted as `latest.parquet` in the
> [CDCgov/covid19-forecast-hub](https://github.com/CDCgov/covid19-forecast-hub) repository, and earlier versions of the
> workflow read Wednesday snapshots from there. CDC now publishes the preliminary data on the data portal, so both
> snapshots come from the portal. The GitHub-hosted preliminary files had a slightly different form, so Wednesday
> snapshots taken before this change may not match later ones exactly. The differences below come from comparing the
> GitHub file last updated 2026-09-23 with the portal release of 2026-09-30.
>
> - **Columns:** every column the workflow keeps is in both files. The GitHub file also had ARI columns
>   (`percent_visits_ari`, `ed_trends_ari`) and `*_threshold_classification` columns. The portal instead has
>   `percent_visits_combined`, the `percent_visits_smoothed*` columns, and `buildnumber`.
> - **Formatting:** the portal stores `fips` as an integer without leading zeros and `week_end` as a timestamp. The
>   workflow zero-pads and trims these, so snapshots end up in the same format as before.
> - **Rows:** both cover the same geographies and HSAs. The portal has a row for every location in every week, including
>   weeks with no data, while the GitHub file mostly left those rows out. As a result, portal snapshots contain more
>   rows where every value is missing.
> - **Values:** on weeks in both files, over 99% of values match exactly. The differences were mostly in the most recent
>   weeks, which is consistent with normal data revision.
> - **Iowa:** the portal release has no Iowa values, at state or HSA level, for weeks from 2026-07-04 on. The GitHub
>   file still had them. This may be a CDC-side data issue rather than a difference between the sources. Either way,
>   `NSSPDataSource` returns missing values for Iowa in those weeks. Its state gap-filling only fills states with no
>   data in any week, and Iowa has earlier data.

## Data read from outside the bucket

### NHSN from Delphi Epidata

`NHSNDataSource` can load NHSN from CMU Delphi's [Epidata V5 API](https://cmu-delphi.github.io/delphi-epidata/api/v5_signals.html)
(`https://delphi.cmu.edu/epidata/v5/`) via [`epidatpy`](https://pypi.org/project/epidatpy/) instead of from S3. This
is opt-in:

```python
from iddata.enums import DataBackend
from iddata.sources.nhsn import NHSNDataSource

NHSNDataSource(backend=DataBackend.EPIDATA)  # or backend="epidata"
```

- **API key:** the Epidata backend requires the `DELPHI_EPIDATA_KEY` environment variable. Request a free key using
  the [registration form](https://api.delphi.cmu.edu/epidata/admin/registration_form). iddata raises a `RuntimeError`
  if the key is missing when Delphi is first queried. Delphi itself accepts requests without a key (with lower rate
  limits) and doesn't reject an invalid key, so iddata can only check that the variable is set. If you pass your own
  `epidata_ctx`, iddata skips this check.
- **Signals:** source `nhsn`, signals `confirmed_admissions_flu_ew` (flu) and `confirmed_admissions_covid_ew`
  (COVID-19), geo types `nation` and `state`, `fill_method="source"`. Only `Disease.FLU` and `Disease.COVID` are
  supported.
- **Earliest version and S3 fallback:** Delphi's NHSN versions start on 2024-11-19. For an earlier date, Delphi returns
  an empty table rather than an error. So with the Epidata backend, an `as_of` before 2024-11-19 falls back to the S3
  path without a warning: the HHS archive before 2024-11-15, and NHSN snapshots for 2024-11-15 to 2024-11-18. No API
  key is needed for these dates. This means the Epidata backend gives no as-published history before 2024-11-19. For
  later `as_of` dates, both backends return every week back to 2020-08-08 as it stood on that date.
- **Agreement with S3:** for a Wednesday `as_of`, the two backends return identical data (checked on 2025-01-08 and
  2026-10-14). Delphi stamps the Wednesday release 12:00 UTC, before the 17:45 UTC S3 snapshot, so a Wednesday `as_of`
  includes that day's release with either backend. Row order within a week can differ.
- **Non-Wednesday `as_of`:** the backends define "as of" at different granularities. S3 returns the most recent
  snapshot on or before `as_of`, which is usually a Wednesday. Epidata returns each value as last published on or before
  `as_of`, including Thursday, Friday and holiday-shifted releases that S3 never captured. For example, with
  `as_of=2025-01-04` the backends differ in 37 state-weeks and 3 national weeks, all from releases on 2025-01-02 and
  2025-01-03. This is expected, and iddata doesn't try to reconcile it.
- **Performance:** one `load()` makes two requests (nation and state) and takes about 1 to 4 seconds. For backtests
  that load the same `as_of` more than once, pass `epidata_ctx=EpiDataContext(use_cache=True)` to cache responses on
  disk. In one test, a repeated request dropped from about 0.9 to 0.2 seconds.

### Population and HSA files

`PopulationData` also builds health service area (HSA) populations from files it downloads directly, not from S3:

- US Census county population estimates for
  [2020–2024](https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv)
  and [2010–2019](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv)
- The NCI SEER [county-to-HSA crosswalk](https://seer.cancer.gov/seerstat/variables/countyattribs/Health.Service.Areas.xls)

If those URLs change or go offline, `PopulationData` will fail even though the bucket is fine.

### Scenario Modeling Hub projections

`SMHDataSource` reads Flu Scenario Modeling Hub weekly hospitalization (`inc hosp`) trajectories from parquet files
under `SMH_DATA_PARQUET_URL` in [`src/iddata/constants.py`](../src/iddata/constants.py). That URL points to
`eda/data/` in the [`lshandross/gbqr-extend`](https://github.com/lshandross/gbqr-extend) GitHub repository, with one
file per round named `flu_scenario-round<N>_gz.parquet`. These files are static and are not snapshotted by any workflow
in this repo.

`as_of` decides which rounds are loaded, based on which rounds had been released by that date:

| `as_of` | Rounds loaded |
|---|---|
| before 2022-08-14 | none; raises `NotImplementedError` |
| 2022-08-14 to 2024-08-10 | 4 |
| 2024-08-11 to 2025-08-09 | 4, 5 |
| 2025-08-10 and later | 4, 5, 6 |

Each round's projections are complete when released and are not revised, so loading the full file for each included
round is correct.

The returned data differs from the surveillance sources in a few ways:

- `location` is prefixed with `syn-` (for example, `syn-25`) to mark it as synthetic rather than observed data.
- `season` combines the season, the first letter of the scenario ID, and the trajectory's `output_type_id` (for
  example, `2023/24A-12`), so that each trajectory is treated as its own season.
- `source` is `smh-<model_id>`.
- A `round` column is included. `output_type_id` means something different in round 4 (one ID per trajectory shared
  across locations) than in rounds 5 and later (one ID per location), so filtering by `output_type_id` should be done
  within each round and location.
- With `rates=True` (the default), `inc` is a rate per 100,000 people. If you pass population data through `ancillary`,
  that data is used, and `pop`/`log_pop` are included in the output. Otherwise `SMHDataSource` loads `PopulationData`
  only for the conversion and leaves those columns out.

> **Memory use:** loading every model for every round needs a lot of memory. Measured with the default `rates=True`
> and no filters:
>
> | `as_of` | Rounds | Rows | Peak memory |
> |---|---|---|---|
> | 2023-12-30 | 4 | ~6.6 million | ~2.5 GB |
> | 2025-09-20 | 4, 5, 6 | ~58 million | ~17 GB |
>
> That is more than many laptops and GitHub-hosted runners have. Pass `model_id` and/or `output_type_id` to
> `SMHDataSource` to keep only the rows you need. The filters are applied as the parquet files are read, so rows that
> don't match never reach the DataFrame. The integration tests use `model_id=["MOBS_NEU-GLEAM_FLU"]` with
> `output_type_id=["1", "2"]` for this reason.

## Adding a new data source

1. Get the raw data into the bucket under a new `data-raw/<source>/` prefix. If the data is revised over time, add a
   snapshot workflow modeled on the existing ones and name files `<source>-YYYY-MM-DD.csv` so that
   `get_versioned_file_path` can find them.
2. Add a `DataSource` subclass in `src/iddata/sources/` and a `SourceType` member in `src/iddata/enums.py`.
3. Add a row for the source to the tables in this document.
