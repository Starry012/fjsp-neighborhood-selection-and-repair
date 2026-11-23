"""Frozen candidate-level diagnostic. Features never change candidate choice."""

from collections import defaultdict, deque
from random import Random
from statistics import mean
from time import perf_counter

from alns_neighborhoods import anchor_hash, select_neighborhood
from experiments.alns_matched import content_hash, history
from experiments import conditional_selection_cost_pilot as cost
from fjsp import validate_schedule
from repair import _frozen_queues

ACTIONS = ("random-operations", "time-window", "parallel-block", "machine-removal")


def graph(instance, anchor, released=()):
    released = set(released)
    rows = {r["operation"]: r for r in anchor["assignments"]}
    durations = {op: min(opts.values()) if op in released else opts[rows[op]["machine"]]
                 for op, opts in instance.options.items()}
    edges = set()
    for sequence in (*instance.jobs.values(), *_frozen_queues(anchor, frozenset(released)).values()):
        edges.update(zip(sequence, sequence[1:]))
    children, degree = defaultdict(list), dict.fromkeys(rows, 0)
    for a, b in edges:
        children[a].append(b)
        degree[b] += 1
    queue = deque(sorted(op for op, d in degree.items() if d == 0))
    order, head = [], dict.fromkeys(rows, 0)
    while queue:
        a = queue.popleft()
        order.append(a)
        head[a] += durations[a]
        for b in children[a]:
            head[b] = max(head[b], head[a])
            degree[b] -= 1
            if degree[b] == 0:
                queue.append(b)
    cost.check(len(order) == len(rows), "precedence graph has a cycle")
    tail = {}
    for a in reversed(order):
        tail[a] = durations[a] + max((tail[b] for b in children[a]), default=0)
    span = max(head.values())
    slack = {a: span - (head[a] + tail[a] - durations[a]) for a in rows}
    return span, slack, edges


def features(instance, anchor, released):
    """Only pre-repair data. Relaxed precedence/load bounds are one-sided."""
    begun = perf_counter()
    horizon = validate_schedule(instance, anchor)
    group = set(released)
    cost.check(group <= instance.options.keys(), "foreign released operation")
    rows = {r["operation"]: r for r in anchor["assignments"]}
    span, slack, edges = graph(instance, anchor)
    lower, _, _ = graph(instance, anchor, group)
    load, fixed = defaultdict(int), defaultdict(int)
    for op, row in rows.items():
        d = instance.options[op][row["machine"]]
        load[row["machine"]] += d
        if op not in group:
            fixed[row["machine"]] += d
    lower = max(lower, max(fixed.values(), default=0))
    cost.check(lower <= horizon and span <= horizon, "invalid relaxed bound")
    critical = {op for op, v in slack.items() if v == 0}
    near = {op for op, v in slack.items() if v <= .05 * span}
    machines = {m for opts in instance.options.values() for m in opts}
    covered = {rows[op]["machine"] for op in group}
    eligible = {m for op in group for m in instance.options[op]}
    top = {m for m, v in load.items() if v == max(load.values())}
    boundary = sum((a in group) != (b in group) for a, b in edges)
    return {"actual_k": len(group), "released_fraction": len(group) / len(rows),
            "critical_coverage": len(group & critical) / len(critical),
            "near_critical_coverage": len(group & near) / len(near),
            "current_machine_coverage": len(covered) / len(machines),
            "eligible_machine_coverage": len(eligible) / len(machines),
            "max_load_machine_coverage": len(top & covered) / len(top),
            "mean_eligible_fraction": mean(len(instance.options[op]) / len(machines) for op in group) if group else 0,
            "processing_reduction_fraction": sum(instance.options[op][rows[op]["machine"]] - min(instance.options[op].values()) for op in group) / horizon,
            "time_span_fraction": (max(rows[op]["end"] for op in group) - min(rows[op]["start"] for op in group)) / horizon if group else 0,
            "boundary_edge_fraction": boundary / len(edges) if edges else 0,
            "current_order_span_fraction": span / horizon,
            "relaxed_lower_bound": lower, "relaxed_gap_fraction": (horizon - lower) / horizon,
            "certified_no_strict_improvement": lower == horizon,
            "feature_seconds": perf_counter() - begun,
            "scope": "Current-order slack; relaxed retained-order/min-duration/load bound. No CP result inputs; no filtering."}


def candidates(instance, anchor, rho, cfg, context_index=0):
    fingerprint = anchor_hash(anchor)
    selections, tasks, aliases, physical = [], [], [], {}
    for draw, generation_seed in enumerate(cfg["generation_seeds"]):
        generated = {}
        for op in ACTIONS:
            seed = content_hash([fingerprint, rho, op, generation_seed])
            t = perf_counter()
            proposal = select_neighborhood(instance, anchor, op, rho, Random(seed))
            selection_seconds = perf_counter() - t
            sid = f"d{draw}-{op}"
            selected = {"id": sid, "draw": draw, "generation_seed": generation_seed,
                        "proposal": proposal, "selection_seconds": selection_seconds,
                        "features": features(instance, anchor, proposal["released"])}
            selections.append(selected)
            generated[op] = selected
        for seed_index, solver_seed in enumerate(cfg["solver_seeds"]):
            offset = (context_index + draw + seed_index) % 4
            order = ACTIONS[offset:] + ACTIONS[:offset]
            if (context_index + draw + seed_index) % 2:
                order = order[::-1]
            for op in order:
                selected = generated[op]
                released = selected["proposal"]["released"]
                for sec in cfg["repair_targets_seconds"]:
                    key = content_hash([fingerprint, released, solver_seed, sec, 1, False])
                    if key not in physical:
                        task = {"id": f"t{len(tasks):04d}", "released": released, "solver_seed": solver_seed,
                                "repair_target_seconds": sec, "anchor_hash": fingerprint,
                                "physical_key": key, "workers": 1, "presolve": False}
                        tasks.append(task)
                        physical[key] = task["id"]
                    aliases.append({"selection_id": selected["id"], "operator": op, "draw": draw,
                                    "generation_seed": selected["generation_seed"], "solver_seed": solver_seed,
                                    "repair_target_seconds": sec, "task_id": physical[key]})
    return {"anchor_hash": fingerprint, "rho": rho, "tasks": tasks, "pairs": aliases, "selections": selections}


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in ("selection_seconds", "feature_seconds")}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def validate_unit(unit, instance, anchor, cfg):
    expected = candidates(instance, anchor, unit["rho"], cfg, int(unit["id"][1:]))
    cost.check(all(semantic(unit[k]) == semantic(v) for k, v in expected.items()), "frozen candidates/features differ")
    cost.check(len(unit["pairs"]) == 32, "expected 4 operators x 2 draws x 2 seeds x 2 budgets")


def paired_rows(units, records):
    rows = []
    for unit in units:
        selected = {s["id"]: s for s in unit["selections"]}
        for alias in unit["pairs"]:
            r = records.get((unit["id"], alias["task_id"]))
            if r is None:
                continue
            f = selected[alias["selection_id"]]["features"]
            cost.check(not f["certified_no_strict_improvement"] or r["gain"] == 0, "observed gain contradicts relaxed bound")
            rows.append({"state_id": unit["id"], "instance": unit["case"]["instance"],
                         "operations": unit["case"]["operations"], "anchor_hash": unit["anchor_hash"],
                         "rho": unit["rho"], **alias, "before": r["before"], "after": r["value"],
                         "gain": r["gain"], "gain_pct": r["G_pct"], "request_seconds": r["request_seconds"],
                         "native_status": r["native_status"], "native_lower_bound": r["native_lower_bound"],
                         "native_seconds": r["native_seconds"], "native_search_seconds": r["native_search_seconds"],
                         "native_nonsearch_seconds": None if r["native_search_seconds"] is None else r["native_seconds"] - r["native_search_seconds"],
                         "certified_no_strict_improvement": f["certified_no_strict_improvement"]})
    return rows


def comparison(rows):
    cells, pools = defaultdict(list), []
    for r in rows:
        cells[r["instance"], r["state_id"], r["draw"], r["solver_seed"], r["repair_target_seconds"]].append(r)
    for key, rs in sorted(cells.items()):
        complete = len(rs) == 4 and {r["operator"] for r in rs} == set(ACTIONS)
        pools.append({"instance": key[0], "state_id": key[1], "draw": key[2], "solver_seed": key[3], "seconds": key[4],
                      "complete": complete, "positive_candidates": sum(r["gain"] > 0 for r in rs),
                      "best_observed_gain_pct": max(r["gain_pct"] for r in rs) if complete else None,
                      "uniform_observed_gain_pct": mean(r["gain_pct"] for r in rs) if complete else None,
                      "observed_selection_headroom_pct": max(r["gain_pct"] for r in rs) - mean(r["gain_pct"] for r in rs) if complete else None})
    return {"four_candidate_pools": pools,
            "scope": "Dependent development observations; no fitted selector, eight-candidate oracle or 60s policy claim.",
            "nonsearch_clock_scope": "Native total minus native solve includes validation/model/decode; not an exact model-build clock."}
