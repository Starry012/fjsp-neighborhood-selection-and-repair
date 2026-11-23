"""Generate fixed development labels and replay measured repair costs."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from importlib.metadata import version
from itertools import combinations, product
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

from baseline import load_pinned_json, solve_ect
from fjsp import parse_instance, validate_schedule
from neighborhoods import select_release_pairs
from repair import _frozen_queues
from v3_chain_pilot import _measure, _plans


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _candidates(instance, schedule, cfg):
    groups = {}

    def add(ops, kind):
        groups.setdefault(tuple(sorted(ops)), set()).add(kind)

    pairs = select_release_pairs(instance, schedule, max_pairs=cfg["max_anchors_per_instance"], seed=cfg["seed"])
    for p in pairs.pairs:
        add((p.anchor, p.blocker), "blocker-pair")
        add((p.anchor, p.random_partner), "random-pair")
    for plan in _plans(instance, schedule, cfg):
        for group in plan["groups"]:
            add(group["chain"], "blocker-chain")
            add(group["random"], "random-extension")
    for draw in range(cfg["sweep_draws"]):
        sequence = sorted(instance.options)
        Random(f"sweep:{cfg['seed']}:{instance.name}:{draw}").shuffle(sequence)
        for size in cfg["sweep_sizes"]:
            if size < len(sequence):
                add(sequence[:size], "random-sweep")
    return [{"released": release, "sources": sorted(kinds)} for release, kinds in groups.items()]


def _prefix(order, results, remaining_ms):
    """Replay fixed calls in order; credit only those that finish in time."""
    best = tried = 0
    for candidate in order:
        elapsed, gain = results[candidate]
        if elapsed > remaining_ms:
            break
        remaining_ms -= elapsed
        tried += 1
        best = max(best, gain)
    return {"gain": best, "completed": tried, "exhausted": tried == len(order)}


def _analyze(output, plans, instances, cfg):
    expected = {(p["instance"], budget, repeat, tuple(c["released"]))
                for p in plans for c in p["candidates"]
                for budget, repeat in product(cfg["repair_budgets_ms"], range(1, p["repeats"] + 1))}
    initial = {p["instance"]: p["initial"] for p in plans}
    trials = defaultdict(dict)
    statuses = Counter()
    verified = late = returned = 0
    total_ms = 0
    with (output / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name, budget, repeat = (row[field] for field in ("instance", "budget_ms", "repeat"))
            release = tuple(row["released"])
            expected.remove((name, budget, repeat, release))
            elapsed = row["outer_seconds"] * 1000
            total_ms += elapsed
            statuses[row["status"]] += 1
            schedule = row.get("schedule")
            on_time = elapsed <= budget
            late += not on_time
            gain = 0
            if schedule is not None:
                returned += 1
                best = validate_schedule(instances[name], schedule)
                old = initial[name]
                scope = "global" if len(release) == len(instances[name].options) else "neighborhood"
                if row["bound_scope"] != scope or row["lower_bound"] > best or (
                    row["status"] == "OPTIMAL" and row["lower_bound"] != best
                ):
                    raise ValueError(f"{name}: invalid bound or scope")
                if best > old["makespan"] or _frozen_queues(schedule, frozenset(release)) != _frozen_queues(
                    old, frozenset(release)
                ):
                    raise ValueError(f"{name}: repair violated its neighborhood")
                verified += 1
                if on_time:
                    gain = old["makespan"] - best
            if row["gain"] is not None and gain != row["gain"]:
                raise ValueError(f"{name}: replay gain differs from recorded gain")
            trials[(name, budget, repeat)][release] = (elapsed, gain, on_time and schedule is not None)
    if expected:
        raise ValueError(f"{len(expected)} planned trials have no result")

    replay = []
    signals = []
    for p in plans:
        name = p["instance"]
        cmax = None if p["initial"] is None else p["initial"]["makespan"]
        order = [tuple(c["released"]) for c in p["candidates"]]
        kinds = {tuple(c["released"]): c["sources"] for c in p["candidates"]}
        structure = sorted(order, key=lambda c: not any(k.startswith("blocker-") for k in kinds[c]))
        random_orders = []
        for draw in range(cfg["random_replay_orders"]):
            shuffled = order.copy()
            Random(f"replay:{cfg['seed']}:{name}:{draw}").shuffle(shuffled)
            random_orders.append(shuffled)
        short, long = sorted(cfg["repair_budgets_ms"])
        repeats = range(1, p["repeats"] + 1)
        for budget, repeat, total in product((short, long), repeats, cfg["replay_budgets_ms"]):
            observed = trials[(name, budget, repeat)]
            costs = {release: value[:2] for release, value in observed.items()}
            remaining = max(0, total - p["prep_ms"])
            affordable = [(gain, -elapsed, release) for release, (elapsed, gain) in costs.items()
                          if elapsed <= remaining]
            reference = max(affordable, default=(0, 0, None))
            random = [_prefix(sequence, costs, remaining) for sequence in random_orders]
            replay.append({"instance": name, "split": p["split"], "cmax": cmax, "pool_size": len(order),
                           "repair_budget_ms": budget, "repeat": repeat, "total_budget_ms": total,
                           "structure": _prefix(structure, costs, remaining),
                           "random_median_gain": median(item["gain"] for item in random),
                           "random_exhausted": sum(item["exhausted"] for item in random),
                           "observed_best_gain": reference[0], "observed_best_release": reference[2]})
        eligible = [c for c in order if all(trials[(name, b, r)][c][2] for b, r in product((short, long), repeats))]
        complete = 0
        reversals = []
        for a, b in combinations(eligible, 2):
            complete += 1
            diffs = [[trials[(name, limit, r)][a][1] - trials[(name, limit, r)][b][1] for r in repeats]
                     for limit in (short, long)]
            if any(all(sign * delta > 0 for delta in diffs[0]) and all(sign * delta < 0 for delta in diffs[1])
                   for sign in (-1, 1)):
                reversals.append({"a": a, "b": b, "short_and_long_gain_differences": diffs})
        signals.append({"instance": name, "split": p["split"], "repeats": p["repeats"], "candidates": len(order),
                        "candidates_always_on_time": len(eligible), "complete_pairs": complete,
                        "strict_reversals": reversals})
    _write(output / "replay.json", replay)
    _write(output / "signals.json", signals)
    return {"labels": sum(statuses.values()), "statuses": dict(statuses), "late": late, "returned": returned,
            "independently_verified": verified, "cumulative_repair_ms": total_ms}


def run(config, output):
    cfg = json.loads(config.read_bytes())
    pins = dict(line.split("==") for line in Path("requirements.txt").read_text().splitlines())
    if platform.python_version() != cfg["python"] or any(version(k) != v for k, v in pins.items()):
        raise ValueError("use the pinned Python and package versions")
    with Path(cfg["archive"]).open("rb") as handle:
        if file_digest(handle, "sha256").hexdigest() != cfg["archive_sha256"]:
            raise ValueError("archive fingerprint mismatch")
    output.mkdir(parents=True)
    tracked = (config, Path(__file__), *(Path(name) for name in (
        "v3_chain_pilot.py", "blocking_chains.py", "neighborhoods.py", "move_scan.py", "repair.py",
        "baseline.py", "fjsp.py", "requirements.txt",
    )))
    _write(output / "run.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                "processor": platform.processor(), "packages": pins},
                                "sha256": {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in tracked}})
    plans = []
    instances = {}
    for case in cfg["cases"]:
        name = case["instance"]
        raw = load_pinned_json(Path(cfg["instance_dir"]) / f"{name}.json", case["sha256"])
        problem = instances[name] = parse_instance(raw)
        if case["family"] != raw["family_long"] or case["operations"] != len(problem.options):
            raise ValueError(f"{name}: unexpected family or size")
        start = perf_counter()
        ect = solve_ect(problem, start + cfg["ect_budget_ms"] / 1000)
        candidates = []
        if ect.schedule is not None:
            validate_schedule(problem, ect.schedule)
            candidates = _candidates(problem, ect.schedule, cfg)
        plans.append({**case, "ect_status": ect.status, "initial": ect.schedule, "candidates": candidates,
                      "prep_ms": (perf_counter() - start) * 1000,
                      "repeats": cfg["validation_repeats"] if case["split"] == "validation-development" else 1})
    _write(output / "plan.json", plans)

    with (output / "labels.jsonl").open("x", encoding="utf-8") as labels:
        for p in plans:
            name = p["instance"]
            tasks = list(product([c["released"] for c in p["candidates"]], cfg["repair_budgets_ms"],
                                 range(1, p["repeats"] + 1)))
            Random(f"label:{cfg['seed']}:{name}").shuffle(tasks)
            for release, budget, repeat in tasks:
                row = _measure(instances[name], p["initial"], release, budget, repeat, cfg)
                labels.write(json.dumps(row, separators=(",", ":")) + "\n")
            labels.flush()
            print(f"{name}: {len(p['candidates'])} candidates, {len(tasks)} repairs", flush=True)
    summary = _analyze(output, plans, instances, cfg)
    summary["ect_failures"] = [p["instance"] for p in plans if p["initial"] is None]
    summary["output_sha256"] = {name: sha256((output / name).read_bytes()).hexdigest() for name in (
        "plan.json", "labels.jsonl", "replay.json", "signals.json",
    )}
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(bool(summary["ect_failures"] or set(summary["statuses"]) - {"OPTIMAL", "FEASIBLE", "UNKNOWN", "TIME_LIMIT"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("experiments/v3_dev_replay.json"))
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output))
