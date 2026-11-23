"""Paired, stratified analysis of final benchmark evidence."""

from collections import Counter
from pathlib import Path
from random import Random
from statistics import mean, median, quantiles
import tarfile

from experiments.final60_job import METHODS, SEEDS, TIMES, accepted, digest, evidence, read
from experiments.offline_job_state import atomic_publish, durable


def effects(arms, index, first, second, by_case):
    values, pair_outcomes = {}, Counter()
    for name in by_case:
        delta = []
        for seed in SEEDS:
            a, b = (arms.get((name, seed, index[x])) for x in (first, second))
            if a is None or b is None:
                pair_outcomes["missing"] += 1
                continue
            delta.append(100 * (b - a) / b)
            pair_outcomes["wins" if a < b else "losses" if a > b else "ties"] += 1
        if len(delta) == len(SEEDS):
            values[name] = median(delta)
    return values, {key: pair_outcomes[key] for key in ("wins", "ties", "losses", "missing")}


def stratified_interval(values, cases, level):
    if not values:
        return None
    samples = {}
    for name, value in values.items():
        samples.setdefault(cases[name]["stratum"], []).append(value)
    generator = Random(20261121)
    estimates = sorted(median([x for rows in samples.values() for x in generator.choices(rows, k=len(rows))])
                       for _ in range(10_000))
    tail = (1 - level) / 2
    return [estimates[int(tail * (len(estimates)-1))], estimates[int((1-tail) * (len(estimates)-1))]]


def distribution(values):
    return {"count": len(values), "median": median(values), "max": max(values),
            "quartiles": quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
            } if values else {"count": 0}


def statistics(cases, rows):
    index = dict(enumerate(METHODS, 1))
    valid = [r for r in rows if r["engineering_passed"]]
    arms = {(r["instance"], r["seed"], r["method"]): r["credited_Cmax"] for r in valid}
    comparisons = {}
    for first, second in ((1, 2), (1, 3), (1, 7), (1, 8), (1, 4), (5, 1), (6, 2), (1, 9)):
        values, pairs = effects(arms, index, first, second, cases)
        level = .9875 if first == 1 and second in (2, 3, 7, 8) else .95
        groups = {}
        for key in ("stratum", "operations", "source"):
            for part in {row[key] for row in cases.values()}:
                sample = [v for name, v in values.items() if cases[name][key] == part]
                groups[str(part)] = {"n": len(sample), "median_pct": median(sample) if sample else None}
        ordered = sorted(values.values())
        result = {
            "methods": [index[first], index[second]], "pairs": pairs, "case_count": len(values),
            "case_outcomes": {"wins": sum(v > 0 for v in ordered), "ties": ordered.count(0),
                              "losses": sum(v < 0 for v in ordered), "missing": len(cases) - len(values)},
            "median_pct": median(ordered) if ordered else None, "confidence_level": level,
            "interval_pct": stratified_interval(values, cases, level), "descriptive_only": pairs["missing"] > 7,
            "strata": groups, "case_effects_pct": values,
            "worse_over_2pct": sum(v < -2 for v in ordered),
            "max_regression_pct": max(0, -ordered[0]) if ordered else None,
            "p90_regression_pct": max(0, -ordered[int((len(ordered)-1)*.1)]) if ordered else None,
            "worse_over_2pct_if_missing_adverse": sum(v < -2 for v in ordered) + len(cases) - len(values),
        }
        comparisons[f"G{first}-G{second}"] = result
    methods = {}
    for method in METHODS:
        attempts = [r for r in rows if r["method"] == method]
        passed = [r for r in valid if r["method"] == method]
        stats = {"qualifying": len(passed), "scheduled": len(cases)*len(SEEDS),
                 "timely_C60": sum(r["credited_Cmax"] is not None for r in attempts),
                 "engineering_failed": len(attempts) - len(passed), "wall_ms": distribution([r["wall_ms"] for r in attempts]),
                 "curves": {}}
        for t in TIMES:
            scores = [r["checkpoints"][str(t)] for r in passed if r["checkpoints"][str(t)] is not None]
            stats["curves"][str(t)] = {"delivered": len(scores), "mean_Cmax": mean(scores) if scores else None}
        for k in ("repair_calls", "selection_seconds", "handoff_Cmax", "post_handoff_gain", "first_native_cp_ms"):
            stats[k] = distribution([r[k] for r in passed if r.get(k) is not None])
        methods[method] = stats

    def supported(row):
        return not row["descriptive_only"] and row["interval_pct"] is not None and row["interval_pct"][0] > 0

    def material(row):
        return (supported(row) and row["median_pct"] >= 3 and row["case_outcomes"]["wins"] >= 32
                and row["worse_over_2pct_if_missing_adverse"] <= 4
                and all(row["strata"][size]["median_pct"] is not None and row["strata"][size]["median_pct"] >= 0
                        for size in ("250", "500")))

    matched = []
    grouped = {(r["instance"], r["seed"], r["method"]): r for r in valid}
    for name in cases:
        for seed in SEEDS:
            a, b = (grouped.get((name, seed, index[x])) for x in (1, 5))
            if a and b and a.get("handoff_sha256") and a["handoff_sha256"] == b.get("handoff_sha256"):
                matched.append(100*(a["credited_Cmax"]-b["credited_Cmax"])/a["credited_Cmax"])
    return {"methods": methods, "comparisons": comparisons,
            "matched_handoffs_G5_vs_G1_pct": distribution(matched),
            "hybrid_beats_both_components": all(supported(comparisons[f"G1-G{x}"]) for x in (2, 3)),
            "formal_gain_gate": (len(valid) == 1296 and all(material(comparisons[f"G1-G{x}"]) for x in (2, 7, 8)))}


def gap_report(cases, main, refs, archived):
    from math import ceil

    reference = {r["instance"]: r for r in refs if r["engineering_passed"]}
    valid = [r for r in main if r["engineering_passed"]]
    rows, bounds, contradictions = [], {}, []
    for name, case in cases.items():
        scores = [r for r in valid if r["instance"] == name]
        result = reference.get(name, {})
        sources = {"integer_structural": case["structure"]["bound"]}
        for row in scores + ([result] if result else []):
            if row.get("global_bound") is not None:
                sources[f"{row['method']}-{row['seed']}-{300 if row is result else 60}s"] = ceil(row["global_bound"] - 1e-6)
        lower = max(sources.values())
        upper = result.get("credited_Cmax")
        verified_values = [r["credited_Cmax"] for r in scores] + ([upper] if upper is not None else [])
        if verified_values and lower > min(verified_values):
            contradictions.append({"instance": name, "bound": lower, "lowest_verified": min(verified_values)})
            continue
        exact = lower if lower in verified_values else None
        bounds[name] = {"L": lower, "independent_U": upper, "verified_best": min(verified_values, default=None),
                        "L_sources": sources, "exact": exact, "reference_status": result.get("status"),
                        "historical": archived.get(name)}
        for row in scores:
            value = row["credited_Cmax"]
            hist = archived.get(name)
            rows.append({"instance": name, "method": row["method"], "seed": row["seed"], "C60": value,
                         "upper_gap_pct": 100*(value-lower)/lower,
                         "lower_gap_pct": max(0, 100*(value-upper)/upper) if upper else None,
                         "true_gap_pct": 100*(value-exact)/exact if exact else None,
                         "archive_BKS_gap_pct": 100*(value-hist["upper_bound"])/hist["upper_bound"]
                         if hist and hist.get("upper_bound") else None})
    summaries = {}
    for method in METHODS:
        summaries[method] = {key: distribution([r[key] for r in rows if r["method"] == method and r[key] is not None])
                             for key in ("upper_gap_pct", "lower_gap_pct", "true_gap_pct", "archive_BKS_gap_pct")}
    return {"bounds": bounds, "quality": rows, "summaries": summaries, "contradictions": contradictions}


def export(bundle, root, plan):
    cases = {b["case"]["instance"]: b["case"] for b in plan["main"]}
    report = {"stages": {}, "manifest_sha256": digest(bundle / "source-freeze.json")}
    selected = {}
    for stage, blocks in plan.items():
        histories = [{"id": b["id"], "expected_arms": b["arms"], "history": evidence(root, b)} for b in blocks]
        report["stages"][stage] = histories
        selected[stage] = [r for b in histories if (a := accepted(b["history"])) for r in a["rows"]]
    report["statistics"] = statistics(cases, selected["main"])
    gaps = gap_report(cases, selected["main"], selected["reference"], read(bundle / "archive-bks.json"))
    report["gaps"] = gaps
    counts = {"main_qualified": sum(r["engineering_passed"] for r in selected["main"]),
              "reference_qualified": sum(r["engineering_passed"] for r in selected["reference"]),
              "bound_conflicts": len(gaps["contradictions"])}
    durable(root / "report.json", {**report, **counts})
    files = {name: root / name for name in ("identity.json", "report.json", "presolve-lock.json", "runtime-lock.json",
             "imported-integration.json")
             if (root / name).is_file()}
    files.update({name: bundle / name for name in ("cases.json", "source-freeze.json")})
    if (bundle / "recovery-source.json").is_file():
        files["recovery-source.json"] = bundle / "recovery-source.json"
    files.update({p.relative_to(root).as_posix(): p for p in (root / "sessions").rglob("*.json")})
    durable(root / "report-manifest.json", {name: digest(path) for name, path in files.items()})
    tmp = root / ".brief.tmp"
    with tarfile.open(tmp, "w:gz") as archive:
        for name, path in {**files, "report-manifest.json": root / "report-manifest.json"}.items():
            archive.add(path, arcname=name, recursive=False)
    target = root / "brief.tar.gz"
    atomic_publish(tmp, target)
    checksum = digest(target)
    (root / "brief.tar.gz.sha256").write_bytes(f"{checksum}  brief.tar.gz\n".encode("ascii"))
    return checksum, counts
