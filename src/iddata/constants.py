import datetime

from iddata.enums import Disease

S3_DATA_RAW_URL = "https://infectious-disease-data.s3.amazonaws.com/data-raw/"

PANDEMIC_SEASONS: tuple[str, ...] = ("2008/09", "2009/10", "2020/21", "2021/22")

# The date NHSNDataSource switches from routing to the legacy HHS-Protect archive
# (_load_from_hhs, flu only, covering through ~2024-05-01) to the current NHSN reporting source
# (_load_from_nhsn). The two pipelines have non-overlapping week coverage, so any code that
# reconstructs a series across multiple `as_of` vintages (e.g. nowcasting's reporting-triangle
# construction) must not let a requested vintage window straddle this boundary -- a legacy-
# archive vintage requested for a wk_end_date it doesn't cover returns NaN, which is
# indistinguishable from "not yet reported" and will corrupt any code assuming only the most
# recent (still-incomplete) cells are missing.
NHSN_SOURCE_CUTOVER_DATE = datetime.date(2024, 11, 15)

# Static per-disease defaults for how many trailing weeks of NHSN data are treated as
# incomplete/subject to reporting delay (and therefore nowcast-corrected).
#
# Derived by running `scripts/estimate_nhsn_max_delay.py --as-of 2025-04-19
# --training-window-weeks 17 --wide-max-delay-weeks 6 --completeness-threshold 0.99` (pooled
# across 53 location/agg_level series, 22 distinct vintages, on 2026-08-04). Two earlier attempts
# were tried and rejected:
#   - completeness_threshold=0.95 (the function default) rounded FLU's real (if small, <5%)
#     revision tail away entirely -- max_delay=1 week, equal to the earliest delay any data is
#     ever visible at all, making correction a structural no-op regardless of backtest date.
#   - --as-of 2025-09-13 with default --training-window-weeks/--wide-max-delay-weeks (0.99
#     threshold) gave FLU=4/COVID=2, but that as_of's (capped) 32-week training window landed on
#     2025-01-30 to 2025-09-13 -- mostly decline/off-season, capturing almost none of the actual
#     rise. `build_increment_triangle`'s reference weeks always end exactly at `--as-of` and
#     extend backward `--training-window-weeks`, so to concentrate the analysis on real season
#     dynamics, `--as-of` must be chosen deliberately (soon after the season, not a convenient
#     "recent" date), with `--wide-max-delay-weeks` no larger than needed (this project's own
#     max_delay is ~4-5 weeks, so a 6-week lookahead margin is already generous) so the training
#     window isn't needlessly shortened by the 2024-11-15 cutover cap. The chosen parameters
#     above land the training window on 2024-12-21 to 2025-04-19 -- the steepest part of the
#     rise (to a Feb 8 peak) through most of the decline. Under these season-concentrated
#     parameters BOTH diseases converge to max_delay=5 (vs. 4/2 from the mostly-off-season
#     window), consistent with the project's original premise that backfill is more substantial
#     during real upswings. Re-run this script periodically (e.g. once a season, or if backtest
#     results suggest NHSN's reporting-delay profile has drifted), choosing --as-of/
#     --training-window-weeks/--wide-max-delay-weeks deliberately per the reasoning above rather
#     than accepting the bare defaults, and update this dict with the reviewed result. IMPORTANT:
#     pass `--as-of` a Saturday (NHSN's wk_end_date grid); a misaligned as_of silently produces
#     an empty/unusable pooled triangle rather than an error.
NHSN_MAX_DELAY_WEEKS: dict[Disease, int] = {
    Disease.FLU: 5,
    Disease.COVID: 5,
}

# IMPORTANT: despite the calibration above, real-data backtesting found that nowcast correction
# using these (or other tried) max_delay values does NOT reliably reduce error -- it increased
# aggregate error across 25 backtest date/location combinations. Root causes: (1) individual
# locations' completion curves vary far more than a single global max_delay can account for
# (delay-1 completion ranges from 68% to 99.9% across a 12-location sample, with some locations
# plateauing early at a genuinely lower ceiling rather than just converging slower) -- NOTE this
# is specifically about `max_delay` (how many trailing weeks are treated as incomplete), not the
# delay-PMF *shape*, which `NHSNNowcaster` already fits per-(location, agg_level) at nowcast time
# (see `NHSN_MAX_DELAY_WEEKS_BY_LOCATION` below for the v2 attempt at per-location max_delay);
# (2) the completion profile is not stable enough over time for a fixed historical curve to
# transfer, and the direction/magnitude of its time-variation differs between the national
# aggregate and individual states (likely because the national pattern is partly an artifact of
# aggregating states with staggered peak timing). See the "Empirical Validation Results" section
# of the project plan for the full analysis. Nowcasting remains implemented and opt-in (off by
# default everywhere) but is NOT recommended for production use until a v2 addresses these.

# Per-(disease, location, agg_level) override of NHSN_MAX_DELAY_WEEKS, for locations whose own
# fitted delay distribution reaches `completeness_threshold` at a meaningfully different delay
# than the pooled/disease-level default above. The mechanism (`NHSNNowcaster._resolve_max_delay`
# checks this dict before falling back to `NHSN_MAX_DELAY_WEEKS[disease]`) is implemented and
# tested, but this dict is deliberately left EMPTY: a real attempt to populate it (below) was
# tried and made things substantially worse, not better.
#
# What was tried: running `scripts/estimate_nhsn_max_delay.py --as-of 2025-04-19
# --training-window-weeks 17 --wide-max-delay-weeks 6 --completeness-threshold 0.99
# --per-location` (same parameters as the pooled default, for direct comparability) fits a
# separate chain-ladder delay-PMF for each of the ~53 location/agg_level series and picks each
# one's own max_delay. This seemed promising going in, since `NHSNNowcaster` already fits its
# delay-PMF *shape* per-location at nowcast time -- only `max_delay` itself had been a single
# pooled/global constant -- and NJ's own result (max_delay=2) matched the independent,
# truth-vintage-based finding that NJ genuinely converges almost immediately.
#
# Re-running `scripts/backtest_nhsn_nowcast.py` with these per-location values (30 date/location
# combinations, the original 5 locations plus PA) showed nowcast correction made aggregate error
# MORE THAN DOUBLE WORSE (total_raw_error=26.68, total_corrected_error=59.36) -- worse than the
# pooled-default result this was meant to improve on. Critically, this wasn't only PA (added
# specifically because its per-location max_delay=2 contradicted independent evidence that it
# needs a LONGER window, not a shorter one, due to a genuinely low completion ceiling): NJ, the
# case expected to benefit most, ALSO got dramatically worse on 4 of 5 backtest dates despite its
# own fitted max_delay matching independent ground truth (e.g. 2025-02-08: raw_error=0.000 ->
# corrected_error=5.125). The likely explanation: fitting a delay-PMF from one location's own
# ~17-week training window is far noisier than fitting it from the ~53-location pooled sample
# (`estimate_delay`'s chain-ladder ratios are sums over reference weeks -- pooling gives many more
# effective observations), and that extra noise costs more accuracy than correctly capturing
# genuine per-location structure gains back, at least at this training-window size and without
# any shrinkage toward the pooled estimate. A real fix would likely need partial pooling/shrinkage
# (weight each location's own fit against the pooled default by how much history it has, rather
# than trusting either alone), a substantially wider per-location training window, or both -- not
# just running the existing pooled-analysis machinery once per location. See the project plan's
# "Empirical Validation Results" section for the full backtest output.
NHSN_MAX_DELAY_WEEKS_BY_LOCATION: dict[Disease, dict[tuple[str, str], int]] = {}
