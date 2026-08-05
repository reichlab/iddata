"""
Real-network smoke test for NHSNNowcaster: hits real S3 NHSN vintages (anonymous), following the
same real-network convention as the other tests in tests/iddata/unit/test_load_data.py.

This only checks that correction runs end-to-end without crashing and returns a same-schema
DataFrame. It intentionally does NOT assert that correction reduces error against later-revised
truth data -- that's a maintainer-run diagnostic (`scripts/backtest_nhsn_nowcast.py`), not a CI
gate, because real-data backtesting found nowcast correction does NOT reliably reduce error (see
that script's docstring and the project plan's "Empirical Validation Results" section for the
full analysis). Nowcasting remains implemented and opt-in but is not recommended for production
use pending a v2.
"""

import datetime

import pandas as pd

from iddata.enums import Disease
from iddata.loader import DiseaseDataLoader
from iddata.nowcast.base import NowcastConfig
from iddata.sources.nhsn import NHSNDataSource


def test_nhsn_nowcast_runs_end_to_end_without_error():
    as_of = datetime.date(2025, 1, 18)
    disease = Disease.FLU

    raw_df = NHSNDataSource(disease=disease).load(as_of=as_of)
    corrected_df = DiseaseDataLoader().load(
        sources=[NHSNDataSource(disease=disease)],
        as_of=as_of,
        nowcast=NowcastConfig(min_vintages=8),
    )

    assert list(corrected_df.columns) == list(raw_df.columns)
    assert corrected_df.shape == raw_df.shape
    assert corrected_df["wk_end_date"].max() == raw_df["wk_end_date"].max()

    national = corrected_df[(corrected_df["location"] == "US") & (corrected_df["agg_level"] == "national")]
    national = national.sort_values("wk_end_date")
    assert not national.empty

    # Only the trailing weeks nearest as_of are in scope for correction -- earlier history can
    # legitimately contain NaNs (e.g. weeks before NHSN reporting began), so restrict these
    # sanity checks to the recent tail rather than the entire historical series.
    recent = national.tail(8)
    assert recent["inc"].notna().all()
    assert (recent["inc"] >= 0).all()

    latest_week = national["wk_end_date"].max()
    corrected_latest = national.loc[national["wk_end_date"] == latest_week, "inc"].iloc[0]
    raw_national = raw_df[(raw_df["location"] == "US") & (raw_df["agg_level"] == "national")]
    raw_latest = raw_national.loc[raw_national["wk_end_date"] == pd.Timestamp(latest_week), "inc"].iloc[0]
    # The trailing (incomplete) week should actually have been touched by correction, confirming
    # the nowcaster ran rather than silently passing through (e.g. due to insufficient vintages).
    assert corrected_latest != raw_latest
