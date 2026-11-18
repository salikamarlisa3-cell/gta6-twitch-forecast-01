"""Per-case LOO detail + evaluation restricted to 'mature hour 1' pseudo-cases
(a = h1/peak >= 0.25), the regime GTA VI's rolling unlock puts it in."""
import json
import math
import sys
sys.path.insert(0, "analysis")
from validate import extract, pinball, MODELS, LEVELS, KIT


def main():
    case_file = json.load(open(f"{KIT}/input/p00.in"))
    comps = []
    for c in case_file["comparables"]:
        curve, pre = extract(c)
        if curve[0] is None or curve[0] <= 0:
            continue
        comps.append((c["name"], curve, pre))
    per = {}
    for i, (name, curve, pre) in enumerate(comps):
        train = [(cv, pr) for j, (_n, cv, pr) in enumerate(comps) if j != i]
        case = {"known": curve[0], "pre": pre}
        peak = max(v for v in curve if v is not None)
        a = curve[0] / peak
        row = {"a": a}
        for mname, fn in MODELS.items():
            fc = fn(case, train)
            losses = [pinball(curve[h - 1], q, tau) for h in range(2, 25) for tau, q in zip(LEVELS, fc[h])]
            row[mname] = sum(losses) / len(losses)
        per[name] = row
    print(f"{'launch':44s} {'a':>6s} " + " ".join(f"{m:>9s}" for m in MODELS))
    for name, row in sorted(per.items(), key=lambda kv: kv[1]["a"]):
        print(f"{name[:44]:44s} {row['a']:6.3f} " + " ".join(f"{row[m]:9.3f}" for m in MODELS))
    for label, subset in (("ALL", list(per.values())), ("a>=0.25", [r for r in per.values() if r["a"] >= 0.25]), ("a>=0.4", [r for r in per.values() if r["a"] >= 0.4])):
        print(f"\n{label} (n={len(subset)}):")
        for m in MODELS:
            vals = [r[m] for r in subset]
            print(f"  {m:10s} mean = {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
