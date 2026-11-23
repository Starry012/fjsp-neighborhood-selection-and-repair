"""Frozen, exact-cardinality paired probes of the three existing operators.

Every physical request is independent of every other request. Identical inputs
are measured once, with explicit logical aliases rather than extra replicates.
"""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter

from alns_neighborhoods import OPERATORS, anchor_hash, select_neighborhood
from experiments import conditional_selection_cost_pilot as cost
from experiments import random_acquisition as acquisition


def content_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_tasks(instance, anchor, cfg):
    """Generate once; the same groups and CP seeds are used at both budgets."""
    fingerprint = anchor_hash(anchor)
    tasks, physical, pairs, selections = [], {}, [], []
    for repeat, (generation_seed, solver_seed) in enumerate(zip(cfg["generation_seeds"], cfg["solver_seeds"], strict=True)):
        for index, operator in enumerate(OPERATORS):
            seed = content_hash([fingerprint, operator, generation_seed])
            started = perf_counter()
            proposal = select_neighborhood(instance, anchor, operator, cfg["rho"], Random(seed + ":operator"))
            selection_seconds = perf_counter() - started
            started = perf_counter()
            random_group = sorted(Random(seed + ":control").sample(sorted(instance.options), len(proposal["released"])))
            control_seconds = perf_counter() - started
            group = {"operator": proposal["released"], "random": random_group}
            selection_id = f"r{repeat}-{operator}"
            selections.append({"id": selection_id, "generation_seed": generation_seed,
                               "solver_seed": solver_seed, "proposal": proposal,
                               "random_released": random_group, "operator_seconds": selection_seconds,
                               "random_seconds": control_seconds})
            # Balance arm order and budget order. Never use a result to form the next task.
            targets = cfg["repair_targets_seconds"][::(-1 if (repeat + index) % 2 else 1)]
            for target in targets:
                arms = ["operator", "random"][::(-1 if (repeat + index + int(target)) % 2 else 1)]
                ids = {}
                for arm in arms:
                    released = group[arm]
                    key = content_hash([fingerprint, released, solver_seed, target, 1, False])
                    if key not in physical:
                        task = {"id": f"t{len(tasks):04d}", "released": released, "solver_seed": solver_seed,
                                "repair_target_seconds": target, "anchor_hash": fingerprint,
                                "physical_key": key, "workers": 1, "presolve": False}
                        tasks.append(task)
                        physical[key] = task["id"]
                    ids[arm] = physical[key]
                pairs.append({"id": f"{selection_id}-s{target}", "selection_id": selection_id,
                              "operator": operator, "repeat": repeat, "solver_seed": solver_seed,
                              "generation_seed": generation_seed, "repair_target_seconds": target,
                              "k": len(proposal["released"]), "tasks": ids})
    return {"anchor_hash": fingerprint, "tasks": tasks, "pairs": pairs, "selections": selections}


def validate_unit(unit, instance, anchor):
    """Assert pairing invariants independently of the task generator."""
    cost.check(unit["anchor_hash"] == anchor_hash(anchor), "unit anchor changed")
    by_id = {t["id"]: t for t in unit["tasks"]}
    cost.check(len(by_id) == len(unit["tasks"]), "duplicate physical task ID")
    for task in unit["tasks"]:
        released = task["released"]
        cost.check(released == sorted(set(released)) and set(released) <= instance.options.keys(), "invalid released IDs")
        cost.check(task["workers"] == 1 and task["presolve"] is False and task["anchor_hash"] == unit["anchor_hash"], "CP configuration differs")
        cost.check(task["physical_key"] == content_hash([unit["anchor_hash"], released, task["solver_seed"],
                   task["repair_target_seconds"], 1, False]), "physical task key differs")
    grouped = defaultdict(list)
    for pair in unit["pairs"]:
        a, b = [by_id[pair["tasks"][arm]] for arm in ("operator", "random")]
        cost.check(len(a["released"]) == len(b["released"]) == pair["k"], "paired actual cardinalities differ")
        cost.check(a["solver_seed"] == b["solver_seed"] == pair["solver_seed"] and
                   a["repair_target_seconds"] == b["repair_target_seconds"] == pair["repair_target_seconds"], "paired repair configuration differs")
        grouped[pair["selection_id"]].append((pair, a, b))
    for group in grouped.values():
        cost.check(len(group) == 2 and {p[0]["repair_target_seconds"] for p in group} == {1, 3}, "expected fresh 1s and 3s branches")
        cost.check(all(group[0][i]["released"] == group[1][i]["released"] for i in (1, 2)), "budget arms use different groups")


def history(root, unit, instance, anchor):
    """Reuse only raw-verified requests whose owned warm worker exited normally."""
    root, completed = Path(root), {}
    tasks = {task["id"]: task for task in unit["tasks"]}
    attempts = sorted((root / "attempts").glob("*"))
    for attempt in attempts:
        if not (attempt / "cleanup.json").exists():
            continue
        cleanup = cost.read(attempt / "cleanup.json")
        if not (cleanup.get("reaped") and cleanup.get("exit_code") == 0 and not cleanup.get("forced")):
            continue
        for directory in sorted((attempt / "requests").glob("*")):
            cost.check(directory.name in tasks, "unplanned request directory")
            if not (directory / "COMPLETED.json").exists():
                continue
            cost.check(directory.name not in completed, "duplicate successful physical request")
            row = acquisition.verify_completed(directory, tasks[directory.name], instance, anchor)
            report = cost.read(directory / "worker-report.json")
            cost.check(report["task"] == tasks[directory.name], "raw report task differs")
            native = report["result"]
            if native["schedule"] is not None:
                value = cost.verify_candidate(instance, anchor, tasks[directory.name]["released"], native["schedule"])
                cost.check(native["status"] in ("OPTIMAL", "FEASIBLE") and value >= row["value"], "native result differs")
            row = {**row, "native_lower_bound": native["lower_bound"], "native_seconds": native["seconds"],
                   "native_search_seconds": native["search_seconds"], "model_sha256": report["model_sha256"],
                   "receipt_path": str(directory / "receipt.json"),
                   "completion_sha256": cost.digest(directory / "COMPLETED.json"),
                   "cleanup_path": str(attempt / "cleanup.json"), "cleanup_sha256": cost.digest(attempt / "cleanup.json")}
            completed[directory.name] = row
    return completed, attempts


def paired_rows(units, records):
    rows = []
    for unit in units:
        for pair in unit["pairs"]:
            a, b = [records.get((unit["id"], pair["tasks"][arm])) for arm in ("operator", "random")]
            if a is None or b is None:
                continue
            cost.check(a["before"] == b["before"] == unit["case"]["before"], "paired initial objectives differ")
            before = a["before"]
            rows.append({"state_id": unit["id"], "instance": unit["case"]["instance"],
                         "operations": unit["case"]["operations"], "anchor_hash": unit["anchor_hash"],
                         **pair, "before": before, "operator_Cmax": a["value"], "random_Cmax": b["value"],
                         "operator_gain_pct": 100 * (before - a["value"]) / before,
                         "random_gain_pct": 100 * (before - b["value"]) / before,
                         "paired_advantage_pct_of_anchor": 100 * (b["value"] - a["value"]) / before,
                         "operator_seconds": a["request_seconds"], "random_seconds": b["request_seconds"],
                         "operator_status": a["native_status"], "random_status": b["native_status"],
                         "same_physical_task": pair["tasks"]["operator"] == pair["tasks"]["random"]})
    return rows


def comparison(rows):
    """Aggregate physical states, then instances; aliases never add evidence."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["instance"], row["state_id"], row["operator"], row["repair_target_seconds"])].append(row)
    state_cells = []
    for key, repeats in sorted(cells.items()):
        complete = len(repeats) == 3
        state_cells.append({"instance": key[0], "state_id": key[1], "operator": key[2], "seconds": key[3],
                            "completed_repeats": len(repeats), "complete": complete,
                            "median_advantage_pct": median(r["paired_advantage_pct_of_anchor"] for r in repeats) if complete else None,
                            "mean_operator_gain_pct": mean(r["operator_gain_pct"] for r in repeats) if complete else None,
                            "mean_random_gain_pct": mean(r["random_gain_pct"] for r in repeats) if complete else None,
                            "operator_improves": sum(r["operator_Cmax"] < r["before"] for r in repeats),
                            "random_improves": sum(r["random_Cmax"] < r["before"] for r in repeats)})
    instance_groups = defaultdict(list)
    for cell in state_cells:
        instance_groups[(cell["instance"], cell["operator"], cell["seconds"])].append(cell)
    instance_cells = []
    for key, states in sorted(instance_groups.items()):
        complete = all(s["complete"] for s in states)
        instance_cells.append({"instance": key[0], "operator": key[1], "seconds": key[2], "physical_states": len(states),
                               "observed_cells_complete": complete,
                               "median_state_advantage_pct": median(s["median_advantage_pct"] for s in states) if complete else None})
    return {"state_cells": state_cells, "instance_cells": instance_cells,
            "scope": "Exposed six-case local mechanism diagnosis. Completeness is reported separately; no training, holdout, causal attribution to missing paper mechanisms, or 60s policy claim.",
            "aggregation": "Three joint generation/CP repeats, then median over unique physical states within each instance. Source aliases are descriptive only; no independent-instance confidence claim."}
