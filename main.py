#!/usr/bin/env python3
"""Rank/borrowing ensemble forecaster for GTA VI trophy earn rates.

Idea in one paragraph: trophies do not move on their own.  The whole
distribution of earn rates slides together as Exophase refreshes its
tracked-player count (new players dilute every rate at once) and as the
launch cohort plays further into the game (early trophies climb, late
ones lag).  So instead of fitting 55 separate curves, this program fits
ONE global curve -- the median logit rate across all trophies as a
function of log-time -- and then places each trophy on it with a
per-trophy offset that is shrunk toward its tier's typical offset.
Small launch-night counts (a trophy held by 4 players versus 5) are
mostly noise, so the shrinkage is strong when few players hold a
trophy and fades as the holder count grows.  The offset is also allowed
to stretch with the horizon: over a week, easy trophies drift up
relative to the pack while rare ones drift further down, which is what
the launch-night refresh actually showed (the easiest trophy fell
least, the platinum fell most).

Everything the program needs comes from the case file it is given:
its history of reads.  A handful of constants below are priors taken
from the competition's own published facts (the launch-night refresh:
638 -> 3,868 tracked players, every rate but the 100% one falling);
when the case history already contains refresh jumps, the history
overrides the priors.

Standard library only.  No network, no randomness, no model calls.
Two runs on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# ---- Priors, from the launch-night refresh published in the rules ----
# Logit fall per unit of log growth in tracked players.  The first
# refresh took players 638 -> 3,868 (log growth 1.80) and the median
# trophy's logit fell roughly 0.6-0.9, so about -0.4 to -0.5 per unit;
# the live 11:27Z snapshot (median logit -1.63 over ~8 h, with the
# tracked population up several-fold) implies the stronger end, -0.6.
BETA_PRIOR = -0.6
# Log growth of tracked players per unit of log-time (hours since the
# release day began, on a ln(1 + h) scale).  638 -> 3,868 in ~4.7 hours
# is a slope near 1.0.
ALPHA_PRIOR = 1.0
# How fast offsets stretch apart with log-time: an easy trophy's offset
# grows by this fraction of itself per ln-hour.  The 8-hour live
# snapshot showed the easiest trophy falling ~0.6 logit while the
# platinum fell ~1.9, i.e. offsets roughly doubled over that window;
# backtests below put the fitted value near 0.65.
GAMMA = 0.65
# Mean gap between Exophase refreshes, in hours.  The page sat
# unchanged for ~4.5 hours on launch night, then jumped.
TAU_REFRESH_HOURS = 5.0
# Log growth of tracked players at the first refresh (638 -> 3,868).
FIRST_JUMP_LOGN = 1.80
# Cross-trophy dispersion of logit moves at a refresh, when the case
# history has not shown one yet.  Keeps short-horizon bands honest:
# the practice showed a flat history can still hide a refresh jump.
# Backtests on the launch-night refresh put it near 0.25.
RESID_PRIOR = 0.25
# Small-count shrinkage: a trophy held by k players keeps k / (k + K)
# of its own offset; the rest is borrowed from its tier.
SHRINK_K = 8.0
# Release-day origin for the log-time axis (midnight UTC, 19 Nov 2026).
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(text):
    """ISO-8601 text (with a Z suffix) -> aware datetime, or None."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def logit(pct):
    """Logit of a percent, clamped so 0 and 100 stay finite."""
    p = min(max(pct / 100.0, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit -> percent in (0, 100)."""
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return 100.0 * p


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def least_squares(xs, ys):
    """Slope and intercept of the least-squares line y ~ a + b*x."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mx = mean(xs)
    my = mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def hours_since_origin(moment):
    if moment is None:
        return 0.0
    delta = (moment - ORIGIN).total_seconds() / 3600.0
    return max(delta, 0.0)


def log_time(moment):
    return math.log1p(hours_since_origin(moment))


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    days = max(horizon_min / 1440.0, 0.0)
    horizon_hours = max(horizon_min / 60.0, 0.0)

    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if target is None and checkpoint is not None:
        target = checkpoint
    x_last = log_time(checkpoint)
    x_target = log_time(target)
    dx = max(x_target - x_last, 0.0)

    n_trophies = len(trophies)
    # Last non-null rate per trophy, and the last read's player count.
    last_rate = [None] * n_trophies
    last_players = None
    for read in history:
        rates = read.get("rates") or []
        for i in range(min(n_trophies, len(rates))):
            if rates[i] is not None:
                last_rate[i] = float(rates[i])
        if read.get("players_tracked") is not None:
            last_players = float(read["players_tracked"])

    # Fallback rates for trophies never seen: their tier's typical rate.
    seen = [r for r in last_rate if r is not None]
    global_rate = median(seen) if seen else 1.0
    tier_rates = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [last_rate[i] for i, t in enumerate(trophies)
                if t.get("tier") == tier and last_rate[i] is not None]
        if vals:
            tier_rates[tier] = median(vals)
    for i, t in enumerate(trophies):
        if last_rate[i] is None:
            last_rate[i] = tier_rates.get(t.get("tier"), global_rate)

    # ---- The global curve: median logit rate per read, over log-time.
    read_x = []
    read_g = []
    read_n = []
    for read in history:
        moment = parse_time(read.get("at"))
        rates = read.get("rates") or []
        logits = [logit(float(r)) for r in rates
                  if r is not None and 0.01 < float(r) < 99.99]
        if not logits:
            continue
        read_x.append(log_time(moment))
        read_g.append(median(logits))
        n = read.get("players_tracked")
        read_n.append(float(n) if n is not None else None)

    if read_g:
        g_last = read_g[-1]
    else:
        g_last = logit(global_rate)

    slope_fit, _intercept = least_squares(read_x, read_g)

    # ---- Refresh jumps seen in the history: dilution per log-N growth.
    jumps = []
    beta_samples = []
    for j in range(1, len(read_g)):
        n_prev, n_cur = read_n[j - 1], read_n[j]
        if n_prev and n_cur and n_cur > n_prev:
            dln_n = math.log(n_cur / n_prev)
            dg = read_g[j] - read_g[j - 1]
            if dln_n > 0.05:
                jumps.append((dln_n, dg))
                beta_samples.append(dg / dln_n)
    n_jumps = len(jumps)
    if beta_samples:
        beta_hat = median(beta_samples)
        # A refresh can only dilute (or leave rates flat); never let a
        # noisy positive sample flip the sign of the dilution.
        beta_hat = min(beta_hat, 0.0)
    else:
        beta_hat = None
    # Blend the history's dilution with the prior; more jumps -> more
    # trust in the history.
    if beta_hat is None:
        beta = BETA_PRIOR
    else:
        beta = (n_jumps * beta_hat + 2.0 * BETA_PRIOR) / (n_jumps + 2.0)

    # ---- Tracked-player growth: fitted log-N slope, blended with prior.
    ln_n_x = []
    ln_n_y = []
    for x, n in zip(read_x, read_n):
        if n and n > 0:
            ln_n_x.append(x)
            ln_n_y.append(math.log(n))
    alpha_fit, _ = least_squares(ln_n_x, ln_n_y)
    if ln_n_y and max(ln_n_y) - min(ln_n_y) > 0.05:
        alpha = (n_jumps * alpha_fit + 2.0 * ALPHA_PRIOR) / (n_jumps + 2.0)
        alpha = min(max(alpha, 0.0), 3.0)
    else:
        alpha = ALPHA_PRIOR

    # Net global slope: the fitted line through the median logits, pulled
    # toward the prior slope implied by dilution (beta * alpha) when the
    # history shows few or no refreshes.
    slope_prior = beta * alpha
    slope = (n_jumps * slope_fit + 2.0 * slope_prior) / (n_jumps + 2.0)
    shift_smooth = slope * dx

    # ---- Refresh imminence: a jump may land inside the horizon even
    # when the history is flat.  Its expected size shrinks as the
    # tracked population grows (early jumps were ~6x, later ones ~1.2x).
    if last_players and last_players > 0:
        dln_next = FIRST_JUMP_LOGN * math.sqrt(638.0 / last_players)
        dln_next = min(max(dln_next, 0.15), FIRST_JUMP_LOGN)
    else:
        dln_next = 1.0
    p_refresh = 1.0 - math.exp(-horizon_hours / TAU_REFRESH_HOURS)
    jump_weight = 2.0 / (n_jumps + 2.0)
    shift_jump = p_refresh * beta * dln_next * jump_weight
    global_shift = shift_smooth + shift_jump

    # Cross-trophy dispersion of logit moves at observed refreshes; this
    # widens the spread when refreshes have actually been jumpy.
    resid = RESID_PRIOR
    if n_jumps >= 2:
        spreads = []
        for j in range(1, len(read_g)):
            n_prev, n_cur = read_n[j - 1], read_n[j]
            if n_prev and n_cur and n_cur > n_prev:
                rates_prev = (history[j - 1].get("rates") or [])
                rates_cur = (history[j].get("rates") or [])
                moves = []
                for a, b in zip(rates_prev, rates_cur):
                    if (a is not None and b is not None
                            and 0.01 < float(a) < 99.99
                            and 0.01 < float(b) < 99.99):
                        moves.append(logit(float(b)) - logit(float(a)))
                if len(moves) >= 10:
                    med = median(moves)
                    spreads.append(median([abs(m - med) for m in moves]))
        if spreads:
            resid = (median(spreads) + RESID_PRIOR) / 2.0

    # ---- Per-trophy offsets, shrunk toward the tier's typical offset.
    tier_offsets = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        offs = [logit(last_rate[i]) - g_last for i, t in enumerate(trophies)
                if t.get("tier") == tier and 0.01 < last_rate[i] < 99.99]
        if len(offs) >= 3:
            tier_offsets[tier] = median(offs)
    # Tiers with too few members borrow from the rarest quarter instead.
    rare_offs = sorted(logit(r) - g_last for r in last_rate
                       if 0.01 < r < 99.99)
    rare_median = median(rare_offs[:max(1, len(rare_offs) // 4)]) if rare_offs else 0.0

    sigma_jump = abs(beta) * dln_next * math.sqrt(max(p_refresh * (1.0 - p_refresh), 0.0)) * jump_weight
    sigma_rw = 0.5 * math.sqrt(days)
    base_sigma = math.sqrt(sigma_rw ** 2 + resid ** 2 + sigma_jump ** 2)
    base_sigma = max(base_sigma, 0.30)

    tier_widen = {"platinum": 1.30, "gold": 1.20, "silver": 1.10, "bronze": 1.0}

    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id")
        rate = last_rate[i]
        tier = trophy.get("tier") or "bronze"

        # The 100% trophy can only stay or fall, and it falls slowly:
        # every new player earns it within their first hours.  Keep its
        # band high and tight, widening only a little with the horizon.
        if rate >= 99.99:
            root_days = math.sqrt(days)
            qs = [99.99 - 0.70 * root_days,
                  99.99 - 0.35 * root_days,
                  99.99 - 0.15 * root_days,
                  99.99 - 0.05 * root_days,
                  99.99]
            qs = [round(min(max(q, 0.01), 99.99), 4) for q in qs]
            rows.append((tid, qs))
            continue

        own_offset = logit(rate) - g_last
        tier_offset = tier_offsets.get(tier, rare_median)
        # Holder count: how many tracked players actually hold it.
        if last_players and last_players > 0:
            holders = max(rate / 100.0 * last_players, 0.0)
        else:
            holders = rate * 6.38  # launch-night scale, when N unknown
        w = holders / (holders + SHRINK_K)
        offset = w * own_offset + (1.0 - w) * tier_offset

        # Median logit: global curve at the target, plus the shrunk
        # offset, stretched by the rank-divergence factor.
        median_logit = g_last + global_shift + offset * (1.0 + GAMMA * dx)

        sigma = base_sigma * tier_widen.get(tier, 1.0)
        # Rarer-than-median trophies get a wider band.
        sigma *= 1.0 + 0.15 * max(0.0, -offset)
        sigma = min(sigma, 3.0)

        qs = []
        prev = None
        for z in Z_VALUES:
            q = round(inv_logit(median_logit + z * sigma), 4)
            q = min(max(q, 0.01), 99.99)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append((tid, qs))

    rows.sort(key=lambda row: row[0])
    return {"trophies": [{"id": tid, "q": qs} for tid, qs in rows]}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        text = json.dumps(result, separators=(",", ":"))
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
