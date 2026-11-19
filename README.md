# GTA VI Trophy Rarity Forecast

Forecast, for every one of Grand Theft Auto VI's 55 PlayStation 5 trophies,
the percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Entry for make.fun `gta6-trophy-rarity-forecast-01` (free, $250 prize pool).
`main.py` (Python 3.13, standard library only) reads every `<id>.in` in
`/input` (or `$MAKEFUN_INPUT_DIR`) and writes `<id>.out` to `/output`
(or `$MAKEFUN_OUTPUT_DIR`). Fully deterministic: no randomness, no network,
same input bytes give the same output bytes.

## Model

Per trophy, from the case file only:

1. **Dedupe the history.** Exophase refreshes every few hours, so
   consecutive reads are usually identical plateaus. We collapse them to
   distinct snapshots.
2. **Log-log trend.** With `x = ln(1 + hours since 2026-11-19T00:00Z)` and
   `y = logit(rate)`, fit a least-squares line per trophy. The slope is the
   net of two forces: progression (existing players earning the trophy,
   pushes the rate up) vs dilution (new players joining with few trophies,
   pushes it down). Each slope is shrunk toward the cross-trophy median
   slope (most trophies share the dilution factor) and the intercept is
   refit through the data centre, then read at the target.
3. **Flat-history dilution shift.** When the history is completely flat
   (launch night: every read identical), the trend is zero — but a refresh
   is overdue. Exophase jumps the tracked-player count and rates fall, so
   we shift the median down 1.2 logit units for every trophy that is not
   already ~100%.
4. **The ~100% trophy stays ~100%.** ("Just Like Paradise": every new
   player earns it immediately, so no dilution applies to it.)
5. **Spread (logit units).** `sqrt((0.5*sqrt(horizon_days))^2 + (fit SE)^2)`,
   floored at 0.35 and capped at 2.5. The `0.5*sqrt(days)` term is the
   house baseline's random-walk spread; the fit-SE term widens the
   interval when the fit is noisy or the target is far outside the
   history's time range. Honest intervals matter: the score is a pinball
   loss on the logit scale.

## Local validation

`score_local.py` implements the contest's pinball loss (logit scale) and
scores an output file against a practice answer. The kit ships no answer
file, so the practice answers were reconstructed from the refresh the
competition description documents (638 → 3,868 players tracked,
943 → 4,276 trophies earned, every rate but the 100% one fell), under
three different distributions of the new players' trophies:

| practice case | this model | persistence baseline (house) |
|---|---|---|
| p00 (proportional) | loss 0.243, miss 0.07 pp, coverage 1.000 | loss 1.401, miss 0.68 pp, coverage 0.000 |
| p00 (uniform) | loss 0.241 | loss 1.387 |
| p00 (easy-concentrated) | loss 0.288 | loss 1.534 |
| p01 (proportional) | loss 0.243 | loss 1.401 |

(House reports the persistence baseline at loss 1.51 on the real practice
answers; our reconstruction gives 1.40, close enough to trust the
comparison.)

A synthetic final-like case (120 reads, 15 distinct snapshots, checkpoint
2026-11-21T11:30Z, target 2026-11-26T00:00Z) also beats the baseline:
loss 0.239 vs 0.312, typical miss 1.07 pp vs 7.56 pp.

Determinism: two runs on the practice cases produce byte-identical
outputs. Runtime: ~0.06 s per case (limit 8 s).

## Files

- `main.py` — the entry program.
- `score_local.py` — dev-only local scorer (not part of the entry).
- `LICENSE` — MIT.
