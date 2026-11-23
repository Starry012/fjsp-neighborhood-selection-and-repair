"""Recompute the four early-handoff arms and compare them to the original benchmark."""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from statistics import mean, median

from summarize import RESULTS, ROOT, SEEDS, outcomes, read_json, stratified_ci


COUNT = ("Stall05-MinOptions2", "Stall10-MinOptions2")
TREE = ("Stall05-Tree2WithoutClock", "Stall10-Tree2WithoutClock")
DIRECT = "MinOptions2"


def compare(index, cases, method, baseline):
    """Take medians over seeds within each case, then over equally weighted cases."""
    effects = {}
    for name in cases:
        deltas = []
        for seed in SEEDS:
            a = index[name, seed, method]["credited_Cmax"]
            b = index[name, seed, baseline]["credited_Cmax"]
            deltas.append(Fraction(100 * (b - a), b))
        effects[name] = median(deltas)
    middle = median(effects.values())
    by_stratum = {}
    for name, effect in effects.items():
        by_stratum.setdefault(cases[name], []).append(float(effect))
    values = {k: float(v) for k, v in effects.items()}
    return {"method": method, "baseline": baseline, **outcomes(list(values.values())),
            "median": float(middle), "interval95": stratified_ci(list(by_stratum.values()), .95),
            "strata": {k: outcomes(v) for k, v in by_stratum.items()}, "case_effects": values}


def summarize_followup():
    def records(name, expected):
        raw = (RESULTS / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Public data checksum mismatch: {name}")
        return [json.loads(line) for line in raw.splitlines()]

    audit = read_json(RESULTS / "early-switch-audit.json")
    new = records("early-switch.jsonl", audit["observations_sha256"])
    old = records("observations.jsonl", read_json(RESULTS / "audit.json")["public_data_sha256"]["observations.jsonl"])
    strata = {c["instance"]: c["stratum"] for c in read_json(ROOT / "engine/cases.json")}
    cases = {r["instance"]: strata[r["instance"]] for r in new}
    index = {(r["instance"], r["seed"], r["method"]): r for r in old + new}
    if (len(cases) != 48 or len(index) != 1872 or len(old) != 1296 or
            Counter(r["method"] for r in new) != dict.fromkeys((*COUNT, *TREE), 144)):
        raise ValueError("Incomplete benchmark or follow-up")
    first = compare(index, cases, COUNT[1], COUNT[0])
    chosen = int(first["median"] > 0)
    selected = COUNT[chosen]
    pairs = [(m, DIRECT) for m in (*COUNT, *TREE, "Stall20-MinOptions2", "Stall35-MinOptions2")]
    pairs += [(m, baseline) for m in COUNT for baseline in ("Stall20-MinOptions2", "Stall35-MinOptions2")]
    pairs += list(zip(TREE, COUNT)) + [(TREE[1-chosen], selected), (TREE[chosen], "Tree2WithoutClock")]
    comparisons = [first] + [compare(index, cases, m, b) for m, b in pairs]
    curves = {}
    for method in (DIRECT, *COUNT, *TREE, "Stall20-MinOptions2", "Stall35-MinOptions2"):
        rows = [index[n, s, method] for n in cases for s in SEEDS]
        curves[method] = {t: mean(r["checkpoints"][t] for r in rows) for t in rows[0]["checkpoints"]}
    matches = {str(t): sum(index[n, s, m]["handoff_sha256"] == index[n, s, b]["handoff_sha256"]
                           for n in cases for s in SEEDS)
               for t, m, b in zip((5, 10), TREE, COUNT)}
    return {"selected_seconds": (5, 10)[chosen], "counts": dict(Counter(r["method"] for r in new)),
            "comparisons": comparisons, "mean_by_second": curves, "identical_handoffs": matches}


if __name__ == "__main__":
    result = summarize_followup()
    (RESULTS / "early-switch-statistics.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(f"576 qualified runs, 48 cases, selected handoff {result['selected_seconds']} s")
