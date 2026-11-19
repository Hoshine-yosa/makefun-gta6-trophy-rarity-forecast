#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01: forecast Exophase trophy earn rates at target.

Model (per trophy), all from the case file only, fully deterministic:
  * History reads are plateaus: Exophase refreshes every few hours, so
    consecutive reads are often identical. We dedupe to distinct snapshots.
  * x = ln(1 + hours since 2026-11-19T00:00Z); y = logit(rate).
    Fit a least-squares line y = a + b*x per trophy (log-log trend).
    The slope b is the net of two forces: progression (existing players
    earning the trophy, pushes the rate up) vs dilution (new players
    joining with few trophies, pushes it down). We shrink each slope
    toward the cross-trophy median slope (most trophies share the
    dilution factor) and refit the intercept through the data centre.
  * Flat history (all reads identical, as on launch night): the trend
    is zero, but a refresh is overdue -- Exophase jumps the tracked
    player count and rates fall. We apply a downward dilution shift to
    the median for every trophy that is not already ~100%.
  * The ~100% trophy ("start the game") stays ~100%: every new player
    earns it immediately, so no dilution applies to it.
  * Spread (logit): sqrt( (0.5*sqrt(horizon_days))^2 + (fit SE)^2 ),
    floored/capped; the 0.5*sqrt(days) term is the house baseline's
    random-walk spread. Wider when the fit is noisy or the target is
    far outside the history's time range.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- helpers ----------------------------------------------------------------

def clamp(x):
    return min(99.99, max(0.01, x))

def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))

def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)

def round4(x):
    """Round to 4 decimals, half up (matches JS Math.round(x*10000)/10000)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def hours_since_release(ts):
    return (ts - RELEASE).total_seconds() / 3600.0

# --- per-case forecast --------------------------------------------------------

def distinct_snapshots(history):
    """Collapse consecutive identical reads to distinct snapshots.
    A snapshot is distinct when (players_tracked, rates) changes."""
    out = []
    prev = None
    for r in history:
        key = (r.get("players_tracked"), tuple(r["rates"]))
        if key != prev:
            out.append(r)
            prev = key
    return out

def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels mismatch")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_ts(case["target"])
    x_target = math.log(1.0 + hours_since_release(target))

    snaps = distinct_snapshots(history)
    n_snap = len(snaps)
    last = history[-1]
    last_rates = last["rates"]

    # x values of the distinct snapshots
    xs = [math.log(1.0 + hours_since_release(parse_ts(s["at"]))) for s in snaps]

    # Raw per-trophy least-squares slope/intercept on (x, logit rate).
    raw = []  # (slope, intercept, resid_sd, n, x_mean, sxx)
    for i in range(len(trophies)):
        pts = []
        for j, s in enumerate(snaps):
            r = s["rates"][i]
            if r is None:
                continue
            pts.append((xs[j], logit(r)))
        if len(pts) < 2:
            raw.append(None)
            continue
        n = len(pts)
        xm = sum(p[0] for p in pts) / n
        ym = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - xm) ** 2 for p in pts)
        if sxx <= 1e-12:
            raw.append((0.0, ym, 0.0, n, xm, 0.0))
            continue
        sxy = sum((p[0] - xm) * (p[1] - ym) for p in pts)
        b = sxy / sxx
        a = ym - b * xm
        resid = [p[1] - (a + b * p[0]) for p in pts]
        dof = max(n - 2, 1)
        sd = math.sqrt(sum(e * e for e in resid) / dof) if n > 2 else 0.0
        raw.append((b, a, sd, n, xm, sxx))

    # Cross-trophy median slope (only trophies with a real fit).
    slopes = sorted(r[0] for r in raw if r is not None)
    med_slope = slopes[len(slopes) // 2] if slopes else 0.0

    # Is the history effectively flat? (one distinct snapshot, or no rate
    # moved at all). Then a refresh is overdue and rates will dilute.
    flat = n_snap <= 1
    if not flat:
        first_rates = snaps[0]["rates"]
        flat = all(
            (a == b) or (a is None and b is None)
            for a, b in zip(first_rates, last_rates)
        )

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        # The ~100% trophy stays ~100%: every new player earns it at once.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.95, 100, 100, 100]})
            continue

        r = raw[i]
        if r is None:
            # No usable history for this trophy: persist the last rate.
            med_logit = logit(last_rate)
            spread = 0.5 * math.sqrt(horizon_days)
        else:
            b, a, sd, n, xm, sxx = r
            # Shrink the slope's deviation from the median toward the median.
            w = n / (n + 3.0)
            b2 = med_slope + (b - med_slope) * w
            # Refit the intercept through the data centre with the new slope.
            ym = a + b * xm
            y_pred = ym + b2 * (x_target - xm)
            med_logit = y_pred
            # Prediction standard error (grows with extrapolation distance).
            if sxx > 1e-12 and n > 2:
                se = sd * math.sqrt(1.0 + 1.0 / n + (x_target - xm) ** 2 / sxx)
            else:
                se = sd
            spread = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + se * se)

        # Flat-history dilution shift: expect an Exophase refresh to add
        # many new players holding few trophies, so most rates fall.
        if flat:
            med_logit += -1.2
            spread = max(spread, 0.9)

        spread = min(2.5, max(0.35, spread))
        qs = [round4(inv_logit(med_logit + z * spread)) for z in Z]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(INPUT_DIR, name), encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_path = os.path.join(OUTPUT_DIR, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(forecast, separators=(",", ":")) + "\n")

if __name__ == "__main__":
    main()
