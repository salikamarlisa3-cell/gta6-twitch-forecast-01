"""Kernel-weighted-by-scale variants: weight each comparable by closeness of its
log hour-1 print to the case's log hour-1 print; weighted mean/sd of log-ratios
per hour; shrunk centre, calibrated spread. Sweeps bandwidth/alpha/beta."""
import json
import math
import sys
sys.path.insert(0, "analysis")
from validate import extract, pinball, LEVELS, Z, KIT


def make_kernel(bw, alpha, beta, floor=0.09):
    def fn(case, train):
        lp = math.log(case["known"])
        weights = []
        for curve, _pre in train:
            if curve[0] and curve[0] > 0:
                d = (math.log(curve[0]) - lp) / bw
                weights.append(math.exp(-0.5 * d * d))
            else:
                weights.append(0.0)
        p1 = case["known"]
        out = {}
        for h in range(2, 25):
            pts = [(math.log(cv[h - 1] / cv[0]), w) for (cv, _), w in zip(train, weights) if cv[0] and cv[h - 1] and w > 1e-6]
            tw = sum(w for _, w in pts)
            mu = sum(w * x for x, w in pts) / tw
            sd = math.sqrt(sum(w * (x - mu) ** 2 for x, w in pts) / tw)
            pts_sorted = sorted(x for x, _w in pts)
            med = pts_sorted[len(pts_sorted) // 2] if len(pts_sorted) % 2 else 0.5 * (pts_sorted[len(pts_sorted)//2 - 1] + pts_sorted[len(pts_sorted)//2])
            center = alpha * med + (1 - alpha) * mu
            sigma = max(beta * sd, floor * math.sqrt(h - 1))
            out[h] = [p1 * math.exp(center + z * sigma) for z in Z]
        return out
    return fn


CANDIDATES = {}
for bw in (0.75, 1.0, 1.5):
    for alpha in (0.5, 1.0):
        for beta in (0.8, 1.0):
            CANDIDATES[f"k{bw}_a{alpha}_b{beta}"] = make_kernel(bw, alpha, beta)


def main():
    case_file = json.load(open(f"{KIT}/input/p00.in"))
    comps = []
    for c in case_file["comparables"]:
        curve, pre = extract(c)
        if curve[0] is None or curve[0] <= 0:
            continue
        comps.append((c["name"], curve, pre))
    results = {m: {"all": [], "m25": [], "m40": [], "big": []} for m in CANDIDATES}
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
            if a >= 0.4: results[mname]["m40"].append(loss)
            if curve[0] >= 150_000: results[mname]["big"].append(loss)
    print(f"{'model':16s} {'ALL':>7s} {'a>=.25':>7s} {'a>=.4':>7s} {'p1>=150k':>8s}")
    for mname, r in sorted(results.items(), key=lambda kv: sum(kv[1]['big'])/len(kv[1]['big'])):
        print(f"{mname:16s} {sum(r['all'])/len(r['all']):7.4f} {sum(r['m25'])/len(r['m25']):7.4f} {sum(r['m40'])/len(r['m40']):7.4f} {sum(r['big'])/len(r['big']):8.4f}  (nbig={len(r['big'])})")


if __name__ == "__main__":
    main()
