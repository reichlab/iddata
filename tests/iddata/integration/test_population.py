"""Tests for the PopulationData ancillary source. These require network access to SEER and census.gov."""

from iddata.ancillary.population import _load_hsa_populations


def test_hsa_populations():
    hsa = _load_hsa_populations()

    assert set(hsa.columns) == {"location", "season", "pop", "agg_level"}
    assert hsa["pop"].isna().sum() == 0
    assert (hsa["agg_level"] == "hsa").all()

    # Season format should be "YYYY/YY", not "YYYY.0/..." (float artifact)
    assert hsa["season"].str.match(r"^\d{4}/\d{2}$").all()

    # All previously-broken HSAs should have real population for a stable season
    season = hsa[hsa["season"] == "2023/24"]
    for hsa_id in ["4", "20", "85", "121",   # Connecticut HSAs (2010-2019 Census fallback)
                   "996", "997"]:              # AK/HI whole-state HSAs (state-total fallback)
        row = season[season["location"] == hsa_id]
        assert len(row) == 1, f"HSA {hsa_id} missing from 2023/24"
        assert row["pop"].iloc[0] > 0, f"HSA {hsa_id} has zero/negative population"

    # Plausibility checks on AK and HI state totals
    assert season[season["location"] == "996"]["pop"].iloc[0] > 700_000    # Alaska ~730k
    assert season[season["location"] == "997"]["pop"].iloc[0] > 1_400_000  # Hawaii ~1.4M

    # No (location, season) duplicates — HSA IDs like "20" must not collide with state FIPS "20"
    assert not hsa[["location", "season"]].duplicated().any()
