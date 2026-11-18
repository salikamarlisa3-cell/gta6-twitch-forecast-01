"""Leave-one-out validation of candidate forecast models on the comparable launches.

Each comparable with a complete release-day curve becomes a pseudo-case in turn:
its hour-1 print is the 'known', its pre-day hours are the 'history', and the model
is fitted on the remaining comparables. Score = the contest's own score: mean pinball
loss on the log scale over hours 2..24 and the five levels.
"""
import json
import math
from datetime import datetime, timedelta

KIT = "/home/hatch/workspace/tools/makefun/kits/gta6-twitch-ccu-forecast-01-staging/gta6-twitch-ccu-forecast-01-kit"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def extract(comp):
    """Return (curve[24], pre[24]) hourly viewers on release day and the day before."""
    ds = parse(comp["day_start"])
    by_from = {p["from"]: p["viewers"] for p in comp["hourly"]}
    curve = [by_from.get(fmt(ds + timedelta(hours=h))) for h in range(24)]
    pre = [by_from.get(fmt(ds - timedelta(hours=k))) for k in range(24, 0, -1)]
    return curve, pre


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def pinball(y, q, tau):
    d = math.log(max(y, 1)) - math.log(max(q, 1e-9))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def unlock_hour(curve, pre):
    """First hour index (0 = hour 1 of release day, negative = day before) at which
    viewers reach 30% of the release-day peak, searching from 24h before day start."""
    peak = max(v for v in curve if v is not None)
    series = pre + curve  # index 24 == hour 1
    for i, v in enumerate(series):
        if v is not None and v >= 0.30 * peak:
            return i - 24
    return 0


# ---------- candidate models: each returns {hour: [q05..q95]} ----------

def model_baseline(case, train):
    p1 = case["known"]
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in range(2, 25)}


def pooled_quantiles(train, hours_pool=0):
    """Per-hour empirical quantiles of log(v_h / v_1) over training curves."""
    out = {}
    for h in range(2, 25):
        logs = []
        for curve, _pre in train:
            if curve[0] and curve[h - 1]:
                logs.append(math.log(curve[h - 1] / curve[0]))
        logs.sort()
        out[h] = [quantile(logs, t) for t in LEVELS]
    return out


def model_pooled(case, train):
    qs = pooled_quantiles(train)
    p1 = case["known"]
    return {h: [p1 * math.exp(x) for x in qs[h]] for h in range(2, 25)}


def model_regime(case, train):
    """Pooled quantiles, but only over comparables whose inferred unlock hour
    regime matches the case's: unlocked by day start (u <= 0) vs during the day.
    The case's regime is inferred from its pre-day trajectory: if the hour before
    day start already holds >= 45% of hour 1's print, the launch was live before
    the day began."""
    pre = case["pre"]
    v0 = pre[-1] if pre and pre[-1] else 0
    case_early = v0 >= 0.45 * case["known"]
    sub = []
    for curve, cpre in train:
        u = unlock_hour(curve, cpre)
        if (u <= 0) == case_early:
            sub.append((curve, cpre))
    if len(sub) < 8:
        sub = train
    qs = pooled_quantiles(sub)
    p1 = case["known"]
    return {h: [p1 * math.exp(x) for x in qs[h]] for h in range(2, 25)}


def model_knn(case, train, k=12):
    """Kernel-weighted per-hour quantiles of log-ratios. Distance between the case
    and a comparable: RMS difference of log viewers relative to hour 1 over the
    12 hours before day start (missing hours skipped), plus the log growth into
    hour 1. Weight = 1 / (0.35 + dist)."""
    def features(pre, v1):
        f = []
        for v in pre[-12:]:
            f.append(math.log(max(v, 1) / v1) if v else None)
        return f

    cf = features(case["pre"], case["known"])
    weighted = []
    for curve, cpre in train:
        tf = features(cpre, curve[0])
        diffs = [(a - b) ** 2 for a, b in zip(cf, tf) if a is not None and b is not None]
        if len(diffs) < 6:
            continue
        dist = math.sqrt(sum(diffs) / len(diffs))
        weighted.append((1.0 / (0.35 + dist), curve))
    weighted.sort(key=lambda w: -w[0])
    top = weighted[:k]
    p1 = case["known"]
    out = {}
    for h in range(2, 25):
        pts = []
        for w, curve in top:
            if curve[0] and curve[h - 1]:
                pts.append((math.log(curve[h - 1] / curve[0]), w))
        qs = []
        for tau in LEVELS:
            pts_sorted = sorted(pts)
            total_w = sum(w for _, w in pts_sorted)
            acc = 0.0
            val = pts_sorted[-1][0]
            for v, w in pts_sorted:
                acc += w
                if acc >= tau * total_w:
                    val = v
                    break
            qs.append(val)
        out[h] = [p1 * math.exp(x) for x in qs]
    return out


MODELS = {
    "baseline": model_baseline,
    "pooled": model_pooled,
    "regime": model_regime,
    "knn12": model_knn,
}


def main():
    case_file = json.load(open(f"{KIT}/input/p00.in"))
    comps = []
    for c in case_file["comparables"]:
        curve, pre = extract(c)
        if curve[0] is None or curve[0] <= 0:
            continue  # skips the GTA VI pre-release entry (no launch-day curve)
        comps.append((c["name"], curve, pre))
    print(f"{len(comps)} comparable launches usable as pseudo-cases")
    totals = {name: [] for name in MODELS}
    for i, (name, curve, pre) in enumerate(comps):
        train = [(cv, pr) for j, (_n, cv, pr) in enumerate(comps) if j != i]
        case = {"known": curve[0], "pre": pre}
        for mname, fn in MODELS.items():
            try:
                fc = fn(case, train)
            except Exception as e:
                print(f"  {mname} failed on {name}: {e}")
                continue
            losses = []
            for h in range(2, 25):
                y = curve[h - 1]
                if y is None:
                    continue
                for tau, q in zip(LEVELS, fc[h]):
                    losses.append(pinball(y, q, tau))
            totals[mname].append(sum(losses) / len(losses))
    for mname, scores in totals.items():
        scores_sorted = sorted(scores)
        print(f"{mname:10s} mean LOO loss = {sum(scores)/len(scores):.4f}  median = {scores_sorted[len(scores)//2]:.4f}  worst = {scores_sorted[-1]:.4f}")


if __name__ == "__main__":
    main()
