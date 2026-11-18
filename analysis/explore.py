"""Exploration of the practice case data (run during development, not by the grader).

For every comparable launch in a practice case: extract its release-day hourly curve,
compute hour-1 level, day peak, the hour-1/peak ratio a, the pre-day level b (mean of the
6 hours before day_start, relative to hour 1), and the growth into hour 1 g = v1 / v0.
Question: which of these runtime-computable features separate 'mature hour 1' launches
(game already out in most regions by 00:00 UTC) from 'ramping hour 1' launches?
"""
import json
import math
import sys
from datetime import datetime, timedelta

KIT = "/home/hatch/workspace/tools/makefun/kits/gta6-twitch-ccu-forecast-01-staging/gta6-twitch-ccu-forecast-01-kit"


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def day_curve(comp):
    ds = parse(comp["day_start"])
    by_from = {p["from"]: p["viewers"] for p in comp["hourly"]}
    curve = []
    for h in range(24):
        curve.append(by_from.get(fmt(ds + timedelta(hours=h))))
    pre = []
    for k in range(1, 13):
        v = by_from.get(fmt(ds - timedelta(hours=k)))
        if v is not None:
            pre.append(v)
    return curve, pre


def main():
    case = json.load(open(f"{KIT}/input/p00.in"))
    rows = []
    for c in case["comparables"]:
        curve, pre = day_curve(c)
        if curve[0] is None or curve[0] <= 0:
            continue
        peak = max(v for v in curve if v is not None)
        a = curve[0] / peak
        b = (sum(pre[:6]) / len(pre[:6])) / curve[0] if pre else float("nan")
        g = curve[0] / pre[0] if pre and pre[0] > 0 else float("nan")
        rows.append((c["name"], curve[0], peak, a, b, g))
    rows.sort(key=lambda r: r[3])
    print(f"{'launch':46s} {'h1':>9s} {'peak':>9s} {'a=h1/peak':>9s} {'b=pre6/h1':>9s} {'g=h1/h0':>8s}")
    for name, h1, peak, a, b, g in rows:
        print(f"{name[:46]:46s} {h1:9.0f} {peak:9.0f} {a:9.3f} {b:9.3f} {g:8.2f}")
    # correlation of log(b), log(g) with a
    import statistics
    for idx, label in ((4, "b"), (5, "g")):
        xs = [math.log(r[idx]) for r in rows if r[idx] == r[idx] and r[idx] > 0]
        ys = [r[3] for r in rows if r[idx] == r[idx] and r[idx] > 0]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        print(f"corr(log {label}, a) = {cov / math.sqrt(vx * vy):.3f}  (n={len(xs)})")


if __name__ == "__main__":
    main()
