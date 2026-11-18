"""gta6-twitch-ccu-forecast-01 entry: a scale-matched comparable-curve forecaster.

Written for this competition (Python 3.13, standard library only, no network).
The grader runs it once per case:  python3 -P -s /bot/main.py
with the case's <id>.in alone in the input directory; it writes <id>.out to the
output directory. Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output.

Method (see METHODOLOGY.md for the full write-up and the leave-one-out numbers):

  A launch day's Twitch curve, hour by hour, is mostly a scaled copy of other big
  launches' curves - but the shape depends on the launch's scale. A launch whose
  first hour already draws half a million viewers (a game already unlocked across
  most of the world, with an audience waiting) stays front-loaded and decays;
  a launch whose first hour draws a few thousand is still unlocking somewhere
  and ramps for hours. So this program does not pool all 40-odd comparable
  launches equally. For every comparable in the case file it rebuilds the
  release-day curve, takes the log ratio of each hour to that launch's own
  hour 1, and pools those log ratios per hour with a Gaussian kernel weight on
  the distance between the comparable's log hour-1 print and this case's:
  launches of a similar size speak loudest.

  Each hour's forecast distribution is then log-normal around a centre that
  blends the kernel-weighted median and mean of the pooled log ratios, with the
  kernel-weighted standard deviation as its spread (floored so the band never
  collapses below a gentle random-walk width). The five scored quantiles are
  centre + z_tau * spread, exponentiated back to viewers and scaled by this
  case's hour-1 print.

  Everything is a deterministic function of the case file: no randomness, no
  clock, no network, so two runs write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the same constants the rules use).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Kernel bandwidth, in natural-log units of the hour-1 print: a comparable whose
# first hour is e^0.75 ~ 2.1x this case's (or 1/2.1x of it) counts exp(-1/2) ~ 0.61
# as much as an exactly size-matched one. Chosen by leave-one-out over the
# comparable launches (analysis/validate4.py): the best of the sweep on the
# contest's own loss, overall and on the big, mature launches GTA VI resembles.
BANDWIDTH = 0.75
# The centre of each hour's log-ratio distribution blends its weighted median
# and weighted mean half and half: the median resists the mixture's outliers,
# the mean keeps the centre honest when the weighted pool is small.
CENTRE_BLEND = 0.5
# Spread floor: an hour's log-spread never goes below SPREAD_FLOOR * sqrt(h - 1),
# so late hours keep a sane band even where the weighted pool happens to agree.
SPREAD_FLOOR = 0.09


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def hour_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def day_curve(comparable: dict) -> list:
    """The comparable's release-day hourly viewers, hours 1..24 (None if absent)."""
    start = parse_ts(comparable["day_start"])
    by_from = {p["from"]: p["viewers"] for p in comparable["hourly"]}
    return [by_from.get(hour_key(start + timedelta(hours=h))) for h in range(24)]


def round2(x: float):
    """Round to 2 decimals, half up; whole numbers come back as ints so the JSON
    writes 1240 where JavaScript would write 1240, not 1240.0."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def weighted_centre_spread(points: list) -> tuple:
    """(centre, spread) of a weighted sample of log ratios.

    points: list of (log_ratio, weight). Centre blends the weighted mean with
    the weighted median; spread is the weighted standard deviation.
    """
    total_w = sum(w for _x, w in points)
    mean = sum(w * x for x, w in points) / total_w
    var = sum(w * (x - mean) ** 2 for x, w in points) / total_w
    ordered = sorted(points)
    acc = 0.0
    median = ordered[-1][0]
    for x, w in ordered:
        acc += w
        if acc >= 0.5 * total_w:
            median = x
            break
    centre = CENTRE_BLEND * median + (1.0 - CENTRE_BLEND) * mean
    return centre, math.sqrt(var)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Rebuild every comparable's release-day curve once, with its kernel weight.
    # Two kinds of comparable are left out: the case's own game (the final case
    # lists the Grand Theft Auto VI category's pre-release history among the
    # comparables; its 'release day' is a trailer-premiere day, not a launch,
    # and it must never be fitted as one) and any comparable whose release day
    # is too incompletely sampled to describe a curve.
    own_name = case["game"]["name"].lower()
    weighted_curves = []
    for comparable in case["comparables"]:
        if own_name in comparable["name"].lower():
            continue
        curve = day_curve(comparable)
        if not curve[0] or curve[0] <= 0:
            continue  # no usable hour 1
        if sum(1 for v in curve if v) < 12:
            continue  # release day too sparse to be evidence about shape
        dist = (math.log(curve[0]) - log_p1) / BANDWIDTH
        weight = math.exp(-0.5 * dist * dist)
        if weight > 1e-6:
            weighted_curves.append((curve, weight))
    if not weighted_curves:
        raise ValueError(f"{name} has no comparable with a usable release day")

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        points = [
            (math.log(curve[h - 1] / curve[0]), w)
            for curve, w in weighted_curves
            if curve[h - 1]
        ]
        if not points:
            raise ValueError(f"{name} has no comparable covering hour {h}")
        centre, spread = weighted_centre_spread(points)
        spread = max(spread, SPREAD_FLOOR * math.sqrt(h - 1))
        quantiles = [
            round2(min(1e9, max(0.01, p1 * math.exp(centre + z * spread))))
            for z in Z
        ]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
