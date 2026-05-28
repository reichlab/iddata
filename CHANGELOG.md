# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
