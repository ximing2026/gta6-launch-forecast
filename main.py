"""Launch-day Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

Model: a launch's day-1 curve, hour by hour, is forecast as ratios to its own
hour-1 print. The ratios come from the case's comparable past launches: for
each comparable, r(h) = viewers(h) / viewers(1) on its release day; the
forecast for hour h is the case's hour-1 print times a weighted log-normal
fit of those ratios (per-hour location and spread in log space), with donors
weighted by unlock-structure and platform similarity to the target, and an
adjustment for how strong the hour before the day (hour 0, the first regional
unlocks) already was relative to hour 1. Deterministic, stdlib only.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Tuned by leave-one-out cross-validation over the comparable launches.
SPREAD_K = 1.0
SIGMA_FLOOR = 0.30
W_UNLOCK_MATCH = 4.0
W_CONSOLE_MATCH = 1.5
HARD_UNLOCK_FILTER = True
HARD_CONSOLE_FILTER = False
MIN_DONORS = 6
USE_HOUR0 = True
HOUR0_BETA_CAP = 1.0


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def hour_ts(day_start, h):
    return (parse_ts(day_start) + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def unlock_kind(text):
    t = (text or "").lower()
    if "midnight" in t and ("local" in t or "rolling" in t):
        return "rolling"
    if "global" in t or "simultaneous" in t or "single" in t:
        return "global"
    if "midnight" in t:
        return "rolling"
    return "other"


def console_only(platforms):
    return "pc" not in (platforms or "").lower()


def day_hour(mapping, day_start, h):
    return mapping.get(hour_ts(day_start, h))


def target_hour0(case):
    """Viewers in the hour just before the case's day (hour 0), from the
    case's own hourly history, or None."""
    want = hour_ts(case["day_start"], 0)
    for pt in case.get("history_hourly", []):
        if pt.get("from") == want:
            return pt.get("viewers")
    return None


def forecast_case(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    notes = case.get("notes", "")
    target_unlock = unlock_kind(notes)
    if case["game"]["name"] == "Grand Theft Auto VI":
        target_unlock = "rolling"
    target_console = case["game"]["name"] == "Grand Theft Auto VI" or "PS5" in notes

    h0 = target_hour0(case)
    a_target = math.log(h0 / p1) if h0 and h0 > 0 else None

    # Donor pool: with the hard filters on, only donors matching the target's
    # unlock structure (and platform class, if that filter is on) are used;
    # if that leaves too few, fall back to the weighted full pool.
    def pool_ok(comp):
        if HARD_UNLOCK_FILTER and unlock_kind(comp.get("unlock", "")) != target_unlock:
            return False
        if HARD_CONSOLE_FILTER and console_only(comp.get("platforms", "")) != target_console:
            return False
        return True

    usable = []
    for comp in case["comparables"]:
        mapping = {pt["from"]: pt["viewers"] for pt in comp["hourly"]}
        base = day_hour(mapping, comp["day_start"], 1)
        if base and base > 0:
            usable.append((comp, mapping, base))
    filtered = [u for u in usable if pool_ok(u[0])]
    donors = filtered if len(filtered) >= MIN_DONORS else usable

    # Donor log-ratio samples per hour, plus each donor's hour-0 log ratio.
    samples = {h: [] for h in range(2, 25)}  # h -> [(log ratio, weight, a_donor)]
    a_vals = []
    for comp, mapping, base in donors:
        cu = unlock_kind(comp.get("unlock", ""))
        w = W_UNLOCK_MATCH if cu == target_unlock else 1.0
        if console_only(comp.get("platforms", "")) == target_console:
            w *= W_CONSOLE_MATCH
        v0 = day_hour(mapping, comp["day_start"], 0)
        a_d = math.log(v0 / base) if v0 and v0 > 0 else None
        if a_d is not None:
            a_vals.append((a_d, w))
        for h in range(2, 25):
            v = day_hour(mapping, comp["day_start"], h)
            if v and v > 0:
                samples[h].append((math.log(v / base), w, a_d))

    a_bar = None
    if a_vals:
        sw = sum(w for _, w in a_vals)
        a_bar = sum(a * w for a, w in a_vals) / sw

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        pts = samples.get(h, [])
        if pts:
            sw = sum(w for _, w, _ in pts)
            mu = sum(x * w for x, w, _ in pts) / sw
            var = sum(w * (x - mu) ** 2 for x, w, _ in pts) / sw
            if USE_HOUR0 and a_target is not None and a_bar is not None:
                sub = [(x, w, a) for x, w, a in pts if a is not None]
                if len(sub) >= 8:
                    sww = sum(w for _, w, _ in sub)
                    mx = sum(x * w for x, w, _ in sub) / sww
                    ma = sum(a * w for _, w, a in sub) / sww
                    cov = sum(w * (x - mx) * (a - ma) for x, w, a in sub) / sww
                    vaa = sum(w * (a - ma) ** 2 for _, w, a in sub) / sww
                    if vaa > 1e-6:
                        beta = max(-HOUR0_BETA_CAP, min(HOUR0_BETA_CAP, cov / vaa))
                        mu += beta * (a_target - a_bar)
            sigma = max(SIGMA_FLOOR, math.sqrt(var) * SPREAD_K)
        else:
            mu, sigma = 0.0, 0.15 * math.sqrt(h - 1)
        q = [p1 * math.exp(mu + z * sigma) for z in Z]
        q = [min(1e9, max(1e-6, v)) for v in q]
        q.sort()
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
