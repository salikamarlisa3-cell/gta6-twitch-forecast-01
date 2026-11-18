"""Tuning sweep: shrunk/calibrated variants of the pooled model, scale-conditioned
subsets, and regime blends. Same LOO protocol as validate.py."""
import json
import math
import sys
sys.path.insert(0, "analysis")
from validate import extract, pinball, quantile, unlock_hour, LEVELS, Z, KIT


def log_ratio_stats(train, subset=None):
    """Per-hour mean and sd of log(v_h/v_1) over the (sub)set of training curves."""
    stats = {}
    for h in range(2, 25):
        logs = []
        for curve, _pre in train:
            if subset is not None and not subset(curve):
                continue
            if curve[0] and curve[h - 1]:
                logs.append(math.log(curve[h - 1] / curve[0]))
        if logs:
            mu = sum(logs) / len(logs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in logs) / len(logs))
            logs.sort()
            med = quantile(logs, 0.5)
            stats[h] = (mu, med, sd, len(logs))
    return stats


def make_shrunk(alpha, beta, subset=None, floor=0.15):
    def fn(case, train):
        stats = log_ratio_stats(train, subset)
        p1 = case["known"]
        out = {}
        for h in range(2, 25):
            mu, med, sd, _n = stats[h]
            center = alpha * med
            sigma = max(beta * sd, floor * math.sqrt(h - 1) * 0.6)
            out[h] = [p1 * math.exp(center + z * sigma) for z in Z]
        return out
    return fn


def make_blend(w_regime):
    def fn(case, train):
        from validate import pooled_quantiles
        pre = case["pre"]
        v0 = pre[-1] if pre and pre[-1] else 0
        case_early = v0 >= 0.45 * case["known"]
        sub = [(cv, pr) for cv, pr in train if (unlock_hour(cv, pr) <= 0) == case_early]
        if len(sub) < 8:
            sub = train
        qa = pooled_quantiles(train)
        qb = pooled_quantiles(sub)
        p1 = case["known"]
        out = {}
        for h in range(2, 25):
            row = []
            for i in range(5):
                la, lb = qa[h][i], qb[h][i]
                row.append(p1 * math.exp((1 - w_regime) * la + w_regime * lb))
            out[h] = row
        return out
    return fn


CANDIDATES = {
    "shrunk_a.5_b1.0": make_shrunk(0.5, 1.0),
    "shrunk_a.3_b1.0": make_shrunk(0.3, 1.0),
    "shrunk_a.7_b1.0": make_shrunk(0.7, 1.0),
    "shrunk_a.5_b0.8": make_shrunk(0.5, 0.8),
    "shrunk_a.5_b1.2": make_shrunk(0.5, 1.2),
    "big_shrunk_a.5": make_shrunk(0.5, 1.0, subset=lambda cv: cv[0] >= 150_000),
    "big_shrunk_a.3": make_shrunk(0.3, 1.0, subset=lambda cv: cv[0] >= 150_000),
    "blend_r.5": make_blend(0.5),
}


def main():
    case_file = json.load(open(f"{KIT}/input/p00.in"))
    comps = []
    for c in case_file["comparables"]:
        curve, pre = extract(c)
        if curve[0] is None or curve[0] <= 0:
            continue
        comps.append((c["name"], curve, pre))
    results = {m: {"all": [], "m25": [], "m40": []} for m in CANDIDATES}
    for i, (name, curve, pre) in enumerate(comps):
        train = [(cv, pr) for j, (_n, cv, pr) in enumerate(comps) if j != i]
        case = {"known": curve[0], "pre": pre}
        peak = max(v for v in curve if v is not None)
        a = curve[0] / peak
        for mname, fn in CANDIDATES.items():
            fc = fn(case, train)
            losses = [pinball(curve[h - 1], q, tau) for h in range(2, 25) for tau, q in zip(LEVELS, fc[h])]
            loss = sum(losses) / len(losses)
            results[mname]["all"].append(loss)
            if a >= 0.25:
                results[mname]["m25"].append(loss)
            if a >= 0.4:
                results[mname]["m40"].append(loss)
    print(f"{'model':18s} {'ALL':>7s} {'a>=.25':>7s} {'a>=.4':>7s}")
    for mname, r in results.items():
        print(f"{mname:18s} {sum(r['all'])/len(r['all']):7.4f} {sum(r['m25'])/len(r['m25']):7.4f} {sum(r['m40'])/len(r['m40']):7.4f}")


if __name__ == "__main__":
    main()
