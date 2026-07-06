# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DiseaseDataLoader.load()` now accepts `drop_pandemic_seasons: bool = True`; when `True`, sets `inc` to `NaN` for pandemic seasons (2008/09, 2009/10, 2020/21, 2021/22) uniformly across all sources after concatenation
- Warning in `DiseaseDataLoader.load()` when `drop_pandemic_seasons=False` and an NHSN source is used with `as_of < 2024-11-15` (HHS archive data is incomplete for pandemic seasons)
- Warning in `DiseaseDataLoader.load()` when `drop_pandemic_seasons=False` and a `FluSurvNetDataSource(burden_adj=True)` is present (CDC burden estimates do not exist for pandemic seasons, so those rows will have `NaN` inc regardless)
- `NSSPDataSource` data source for NSSP emergency department visit data

### Changed
- **Breaking**: `drop_pandemic_seasons` removed from `NHSNDataSource` and `ILINetDataSource` constructors; pandemic season handling is now a loader-level concern applied uniformly to all sources
- `FluSurvNetDataSource.load()`: inner join with burden adjustment factors replaced by left join, so pandemic seasons produce `NaN` inc rather than being silently absent from the output when `burden_adj=True`

## [2.0.0]

### Added
- `DataSource` abstract base class in `sources/base.py`; concrete subclasses `NHSNDataSource`, `NSSPDataSource`, `ILINetDataSource`, `FluSurvNetDataSource` each encapsulate source-specific loading logic
- `AncillaryData` abstract base class for supplementary data (e.g. `PopulationData`); separates covariates from surveillance targets
- Canonical enums `Disease`, `SourceType`, `AggLevel` in `enums.py`; imported by idmodels
- `s3.py` module with versioned S3 file lookup logic extracted from the loader
- `constants.py` module for shared constants

### Changed
- **Breaking**: `DiseaseDataLoader` replaced with a thin orchestrator; all source-specific loading logic moved into `DataSource` subclasses
- **Breaking**: source-specific numeric pre-transforms removed from `DataSource.load()` — NHSN `+0.75**4`, ILINet `(+exp(-7))*4`, FluSurvNet `(+exp(-3))/2.5` are no longer applied in iddata; `DataSource.load()` now returns data in original measurement units; these transforms are applied and inverted in idmodels
- Power transform and center/scale normalization removed from loader (now handled entirely in idmodels)

## [0.1.0]

### Added
- Initial package setup and structure
- Data loading for NHSN, ILINet, FluSurv-NET, and NSSP sources
- S3-based data access for versioned snapshots

[Unreleased]: https://github.com/reichlab/iddata/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/reichlab/iddata/compare/v0.1.0...v2.0.0
[0.1.0]: https://github.com/reichlab/iddata/releases/tag/v0.1.0
