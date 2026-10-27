"""NHSNDataSource(backend=EPIDATA) against Delphi's Epidata V5 API, compared with the S3 backend.

Skipped unless DELPHI_EPIDATA_KEY is set. Mocked coverage of the epidata path lives in tests/iddata/unit/test_sources.py.
"""

import datetime
import os

import pandas as pd
import pytest

from iddata.enums import DataBackend
from iddata.sources.nhsn import NHSNDataSource

pytestmark = pytest.mark.skipif(not os.environ.get("DELPHI_EPIDATA_KEY"), reason="DELPHI_EPIDATA_KEY is not set")


def _load_both(as_of: datetime.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (epidata, s3) frames for as_of, sorted so that row order doesn't matter."""
    dats = [NHSNDataSource(backend=backend, rates=False).load(as_of=as_of)
            for backend in (DataBackend.EPIDATA, DataBackend.S3)]
    return tuple(dat.sort_values(["location", "wk_end_date"]).reset_index(drop=True) for dat in dats)


def test_epidata_matches_s3_on_wednesday():
    # Delphi's Wednesday release is stamped 12:00 UTC, before the 17:45 UTC S3 snapshot, so the two should agree
    epidata_dat, s3_dat = _load_both(datetime.date(2025, 1, 8))
    assert len(epidata_dat) > 0
    pd.testing.assert_frame_equal(epidata_dat, s3_dat)


def test_epidata_schema_matches_s3():
    epidata_dat, s3_dat = _load_both(datetime.date(2025, 1, 8))
    assert list(epidata_dat.columns) == list(s3_dat.columns)
    assert epidata_dat.dtypes.to_dict() == s3_dat.dtypes.to_dict()
    assert epidata_dat["inc"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(epidata_dat["wk_end_date"])


def test_epidata_falls_back_to_s3_before_first_version():
    epidata_dat, s3_dat = _load_both(datetime.date(2024, 3, 2))
    pd.testing.assert_frame_equal(epidata_dat, s3_dat)
