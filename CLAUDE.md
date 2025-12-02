# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

iddata is a Python package for accessing infectious disease data from various sources (FluSurv-NET, NHSN, ILINet, NSSP, WHO-NREVSS). Data is fetched from S3 (`infectious-disease-data.s3.amazonaws.com/data-raw/`).

## Development Commands

```bash
# Install dependencies and package in editable mode
python -m pip install -r requirements/requirements-dev.txt && python -m pip install -e .

# Run tests
python -m pytest

# Run a single test
python -m pytest tests/iddata/unit/test_load_data.py::test_load_data_sources

# Lint and format (via pre-commit)
pre-commit run --all-files

# Update lockfiles after changing pyproject.toml dependencies
uv pip compile pyproject.toml -o requirements/requirements.txt && uv pip compile pyproject.toml --extra dev -o requirements/requirements-dev.txt
```

## Architecture

The package has a single main class `DiseaseDataLoader` in `src/iddata/loader.py` that provides methods to load data from different sources:

- `load_data()` - Main entry point that combines multiple sources with transformations
- `load_nhsn()` - Hospital admissions data (flu/covid) from NHSN or legacy HHS
- `load_ilinet()` - ILI surveillance data (national, HHS regions, states)
- `load_flusurv_rates()` - FluSurv-NET hospitalization rates by site
- `load_nssp()` - NSSP emergency department visit data (flu/covid/rsv)

Supporting utilities in `src/iddata/utils.py` handle epiweek/season conversions using `pymmwr`.

## Data Conventions

- Seasons are formatted as "YYYY/YY" (e.g., "2022/23")
- Season weeks start at week 31 (late July) and run through week 30 of the following year
- Pandemic seasons (2008/09, 2009/10, 2020/21, 2021/22) are often dropped via `drop_pandemic_seasons` parameter
- `as_of` parameter allows loading historical data snapshots

## Code Style

- Ruff for linting/formatting with 120 char line length and double quotes
- Pre-commit hooks run ruff, mypy, codespell, and security checks
- Python 3.12 (see .python-version)
