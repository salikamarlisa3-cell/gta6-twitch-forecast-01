"""Refinements on the scale-kernel model: + pre-day-peak kernel dimension,
+ smoothing of centre/sigma across neighbouring hours."""
import json
import math
import sys
sys.path.insert(0, "analysis")
from validate import extract, pinball, LEVELS, Z, KIT


def pre_peak_ratio(pre, v1):
    vals = [v for v in pre if v]
    return (max(vals) / v1) if vals and v1 else 1.0


def make_model(bw_scale=0.75, bw_pre=None, alpha=0.5, beta=1.0, floor=0.09, smooth=False):
    def fn(case, train):
        lp = math.log(case["known"])
        lpre = math.log(max(pre_peak_ratio(case["pre"], case["known"]), 1e-3))
        weights = []
        for curve, cpre in train:
            if not curve[0]:
                weights.append(0.0)
                continue
            d = (math.log(curve[0]) - lp) / bw_scale
            w = math.exp(-0.5 * d * d)
            if bw_pre:
                dp = (math.log(max(pre_peak_ratio(cpre, curve[0]), 1e-3)) - lpre) / bw_pre
                w *= math.exp(-0.5 * dp * dp)
            weights.append(w)
        p1 = case["known"]
        centers, sigmas = {}, {}
        for h in range(2, 25):
            pts = [(math.log(cv[h - 1] / cv[0]), w) for (cv, _), w in zip(train, weights) if cv[0] and cv[h - 1] and w > 1e-6]
            tw = sum(w for _, w in pts)
            mu = sum(w * x for x, w in pts) / tw
            sd = math.sqrt(sum(w * (x - mu) ** 2 for x, w in pts) / tw)
            srt = sorted(x for x, _w in pts)
            n = len(srt)
            med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
            centers[h] = alpha * med + (1 - alpha) * mu
            sigmas[h] = max(beta * sd, floor * math.sqrt(h - 1))
        if smooth:
            c2, s2 = {}, {}
            for h in range(2, 25):
                hs = [x for x in (h - 1, h, h + 1) if 2 <= x <= 24]
                c2[h] = sum(centers[x] for x in hs) / len(hs)
                s2[h] = sum(sigmas[x] for x in hs) / len(hs)
            centers, sigmas = c2, s2
        return {h: [p1 * math.exp(centers[h] + z * sigmas[h]) for z in Z] for h in range(2, 25)}
    return fn


CANDIDATES = {
    "base_k.75": make_model(),
    "+pre_bw1.0": make_model(bw_pre=1.0),
    "+pre_bw0.6": make_model(bw_pre=0.6),
    "+smooth": make_model(smooth=True),
    "+pre1.0+smooth": make_model(bw_pre=1.0, smooth=True),
}


def main():
    case_file = json.load(open(f"{KIT}/input/p00.in"))
    comps = []
    for c in case_file["comparables"]:
        curve, pre = extract(c)
        if curve[0] is None or curve[0] <= 0:
            continue
        comps.append((c["name"], curve, pre))
    results = {m: {"all": [], "m25": [], "big": []} for m in CANDIDATES}
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
            if a >= 0.25: results[mname]["m25"].append(loss)
            if curve[0] >= 150_000: results[mname]["big"].append(loss)
    print(f"{'model':16s} {'ALL':>7s} {'a>=.25':>7s} {'p1>=150k':>8s}")
    for mname, r in results.items():
        print(f"{mname:16s} {sum(r['all'])/len(r['all']):7.4f} {sum(r['m25'])/len(r['m25']):7.4f} {sum(r['big'])/len(r['big']):8.4f}")


if __name__ == "__main__":
    main()
