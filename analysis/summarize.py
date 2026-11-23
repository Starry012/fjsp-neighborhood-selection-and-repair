"""Recompute the tables from the public score records, using the Python standard library."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean, median, quantiles


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ("Stall35-MinOptions2", "MinOptions2", "Stall80", "Stall20-MinOptions2",
           "Stall35-Tree2WithoutClock", "Tree2WithoutClock", "active-strict", "full-cp-sat-hinted", "Chuffed")
METHOD_LABELS = {
    "Stall35-MinOptions2": ("GLNSA 35 秒 → 计数规则选组修复", "GLNSA (35 s) → Count-LNS"),
    "MinOptions2": ("计数规则选组修复", "Count-LNS"),
    "Stall80": ("GLNSA", "GLNSA"),
    "Stall20-MinOptions2": ("GLNSA 20 秒 → 计数规则选组修复", "GLNSA (20 s) → Count-LNS"),
    "Stall35-Tree2WithoutClock": ("GLNSA 35 秒 → 树模型选组修复", "GLNSA (35 s) → Tree-LNS"),
    "Tree2WithoutClock": ("树模型选组修复", "Tree-LNS"),
    "active-strict": ("旧候选池选组修复", "Legacy-LNS"),
    "full-cp-sat-hinted": ("全模型 OR-Tools CP-SAT", "全模型 OR-Tools CP-SAT"),
    "Chuffed": ("全模型 MiniZinc＋Chuffed", "全模型 MiniZinc＋Chuffed"),
}
TIMES = (5, 10, 20, 35, 50, 60)
SEEDS = (42, 43, 44)
PRIMARY = ((1, 2), (1, 3), (1, 7), (1, 8))
REGISTERED = (*PRIMARY, (1, 4), (5, 1), (6, 2), (1, 9))
EXPLORATORY = tuple((2, j) for j in (1, 3, 4, 5, 6, 7, 8, 9))


def read_json(path):
    return json.loads(path.read_bytes())


def case_effects(index, names, first, second, time=60):
    result = {}
    for name in names:
        values = []
        for seed in SEEDS:
            a, b = (index[name, seed, arm]["checkpoints"][str(time)] for arm in (first, second))
            values.append(100 * (b - a) / b)
        result[name] = median(values)
    return result


def outcomes(values):
    values = list(values)
    return {"median": median(values), "wins": sum(v > 0 for v in values), "ties": values.count(0),
            "losses": sum(v < 0 for v in values)}


def stratified_ci(groups, level):
    rng = Random(20261121)
    draws = sorted(median([x for group in groups for x in rng.choices(group, k=len(group))]) for _ in range(10000))
    tail = (1 - level) / 2
    return [draws[int(tail * (len(draws) - 1))], draws[int((1 - tail) * (len(draws) - 1))]]


def spread(values):
    if not values:
        return None
    return {"observations": len(values), "quartiles": quantiles(values, n=4, method="inclusive"), "max": max(values)}


def summarize():
    audit = read_json(RESULTS / "audit.json")
    for name, expected in {**audit["public_data_sha256"], "../engine/cases.json": audit["registered_cases_sha256"]}.items():
        if sha256((RESULTS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    observations = [json.loads(line) for line in (RESULTS / "observations.jsonl").read_bytes().splitlines()]
    cases = {r["instance"]: r for r in read_json(ROOT / "engine/cases.json")}
    rows = [r for r in observations if r["stage"] == "main"]
    bounds = read_json(RESULTS / "bounds.json")
    index = {(r["instance"], r["seed"], METHODS.index(r["method"]) + 1): r for r in rows}
    groups = defaultdict(list)
    # Use the locked block order also for deterministic bootstrap resampling.
    for name in dict.fromkeys(r["instance"] for r in rows):
        groups[cases[name]["stratum"]].append(name)
    if (len(index) != 1296 or len(rows) != len(index) or len(rows) != len(observations)
            or set(bounds) != set(cases) or sorted(map(len, groups.values())) != [12] * 4):
        raise ValueError("Incomplete benchmark")

    methods = {}
    for i, method in enumerate(METHODS, 1):
        selected = [r for r in rows if r["method"] == method]
        # The six project/GLNSA arms also logged 30 s. Include a time only
        # when every run of this arm has a delivered score at that time.
        available_times = sorted(int(t) for t in selected[0]["checkpoints"]
                                 if all(r["checkpoints"].get(t) is not None for r in selected))
        if not set(TIMES).issubset(available_times):
            raise ValueError(f"Missing registered checkpoint for {method}")
        display_name, short_name = METHOD_LABELS[method]
        methods[f"G{i}"] = {
            "method": method, "display_name": display_name, "short_name": short_name,
            "n": len(selected), "mean_C60": mean(r["credited_Cmax"] for r in selected),
            "mean_by_second": {str(t): mean(r["checkpoints"][str(t)] for r in selected) for t in available_times},
            "mean_C60_by_stratum": {s: mean(r["credited_Cmax"] for r in selected if r["instance"] in names)
                                    for s, names in groups.items()},
            "wall_s_median": median(r["wall_ms"] for r in selected) / 1000,
            "repairs_median": median(r["repair_calls"] for r in selected) if "repair_calls" in selected[0] else None,
            "selection_s_median": median(r["selection_seconds"] for r in selected) if "selection_seconds" in selected[0] else None,
        }
    comparisons = {}
    for first, second in (*REGISTERED, *EXPLORATORY):
        effect = case_effects(index, cases, first, second)
        strata = {s: [effect[n] for n in names] for s, names in groups.items()}
        level = .9875 if (first, second) in PRIMARY else .95
        comparisons[f"G{first}-G{second}"] = {
            **outcomes(effect.values()), "level": level, "interval": stratified_ci(list(strata.values()), level),
            "strata": {s: outcomes(v) for s, v in strata.items()}, "case_effects": effect,
        }
    curves = {s: {str(t): outcomes(case_effects(index, names, 1, 2, t).values()) for t in TIMES}
              for s, names in {"all": list(cases), **groups}.items()}
    quality = {}
    timing = {}
    for s, names in {"all": list(cases), **groups}.items():
        upper, bks = [], []
        archive_lower_gap, archive_upper_gap = [], []
        initial_gains, repair_gains = [], []
        for name in names:
            b = bounds[name]
            paired_initial = []
            for seed in SEEDS:
                v = index[name, seed, 2]["credited_Cmax"]
                ect = index[name, seed, 8]["ECT"]
                paired_initial.append(100 * (ect - v) / ect)
                h = index[name, seed, 1]
                repair_gains.append(h["handoff_Cmax"] - h["credited_Cmax"])
                upper.append(100 * (v - b["L"]) / b["L"])
                if b["historical"]:
                    floor = b["historical"]["lower_bound"]
                    target = b["historical"]["upper_bound"]
                    if not 0 < floor <= min(target, b["best_verified"]):
                        raise ValueError(f"Archive bound conflicts with a feasible schedule: {name}")
                    bks.append(100 * (v - target) / target)
                    archive_lower_gap.append(max(0., bks[-1]))
                    archive_upper_gap.append(100 * (v - floor) / floor)
            initial_gains.append(median(paired_initial))
        quality[s] = {
            "upper_gap_pct": spread(upper), "archive_BKS_gap_pct": spread(bks),
            "archive_conditional_gap_lower_pct": spread(archive_lower_gap),
            "archive_conditional_gap_upper_pct": spread(archive_upper_gap),
            "archive_status": dict(Counter(bounds[n]["historical"]["status"] for n in names if bounds[n]["historical"])),
            "proved": sum(bounds[n]["proved_optimum"] is not None for n in names),
            "G2_over_initial_ECT": outcomes(initial_gains),
        }
        at35, at60 = (case_effects(index, names, 1, 2, t) for t in (35, 60))
        def outcome_label(effect):
            return "win" if effect > 0 else "loss" if effect < 0 else "tie"
        timing[s] = {"G1_35_to_60_signs": dict(Counter(outcome_label(at35[n]) + "_to_" + outcome_label(at60[n])
                                                        for n in names)),
                     "G1_improved_after_handoff_runs": sum(v > 0 for v in repair_gains)}
    matched = []
    for name in cases:
        for seed in SEEDS:
            a, b = index[name, seed, 1], index[name, seed, 5]
            if a["handoff_sha256"] and a["handoff_sha256"] == b["handoff_sha256"]:
                matched.append(100 * (a["credited_Cmax"] - b["credited_Cmax"]) / a["credited_Cmax"])
    return {"methods": methods, "comparisons": comparisons, "G1_vs_G2_by_second": curves,
            "handoff": timing, "G5_over_G1_identical_handoff": spread(matched), "G2_quality": quality}


if __name__ == "__main__":
    result = summarize()
    path = RESULTS / "statistics.json"
    path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode())
    print(f"Wrote {path.name}: 9 methods and {len(result['comparisons'])} comparisons.")
