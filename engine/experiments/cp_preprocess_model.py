"""Equivalent retained-order domain tightening; frozen repair.py is untouched."""

from collections import defaultdict, deque
from hashlib import sha256
import json
from time import perf_counter

from ortools.sat.python import cp_model

from experiments import conditional_selection_cost_pilot as cost
from fjsp import validate_schedule
from repair import RepairResult, _frozen_queues


def preprocess(instance, initial, released):
    """One DAG pass. Head/tail exclude the operation's own processing time."""
    horizon = validate_schedule(instance, initial)
    released = frozenset(released)
    cost.check(released <= instance.options.keys(), "foreign operation")
    rows = {r["operation"]: r for r in initial["assignments"]}
    durations = {op: min(opts.values()) if op in released else opts[rows[op]["machine"]]
                 for op, opts in instance.options.items()}
    edges = set()
    for sequence in (*instance.jobs.values(), *_frozen_queues(initial, released).values()):
        edges.update(zip(sequence, sequence[1:]))
    children, degree = defaultdict(list), dict.fromkeys(instance.options, 0)
    for a, b in sorted(edges):
        children[a].append(b)
        degree[b] += 1
    queue, order = deque(sorted(op for op, n in degree.items() if n == 0)), []
    head = dict.fromkeys(instance.options, 0)
    while queue:
        a = queue.popleft()
        order.append(a)
        for b in children[a]:
            head[b] = max(head[b], head[a] + durations[a])
            degree[b] -= 1
            if degree[b] == 0:
                queue.append(b)
    cost.check(len(order) == len(instance.options), "retained-order cycle")
    tail = {}
    for a in reversed(order):
        tail[a] = max((durations[b] + tail[b] for b in children[a]), default=0)
    bounds, allowed = {}, {}
    for op, opts in instance.options.items():
        d = durations[op]
        bounds[op] = {"start": [head[op], horizon - tail[op] - d],
                      "end": [head[op] + d, horizon - tail[op]]}
        allowed[op] = [m for m, p in opts.items() if head[op] + p + tail[op] <= horizon] if op in released else [rows[op]["machine"]]
        row = rows[op]
        cost.check(row["machine"] in allowed[op] and all(bounds[op][key][0] <= row[key] <= bounds[op][key][1]
                   for key in ("start", "end")), "preprocessing removed the feasible anchor")
    certificate = {"horizon": horizon, "head": head, "tail": tail, "bounds": bounds,
                   "allowed_machines": allowed, "minimum_durations": durations, "edges": sorted(edges)}
    before = sum(len(instance.options[op]) for op in released)
    after = sum(len(allowed[op]) for op in released)
    old_domain_values = 2 * len(instance.options) * (horizon + 1)
    new_domain_values = sum(hi - lo + 1 for b in bounds.values() for lo, hi in b.values())
    summary = {"certificate_sha256": sha256(json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "release_options_before": before, "release_options_after": after, "pruned_options": before - after,
               "time_domain_values_before": old_domain_values, "time_domain_values_after": new_domain_values,
               "time_domain_reduction_fraction": 1 - new_domain_values / old_domain_values,
               "tightened_time_variables": sum(lo > 0 or hi < horizon for b in bounds.values() for lo, hi in b.values()),
               "retained_edges": len(edges), "precedence_lower_bound": max(head[op] + durations[op] for op in order),
               "passes": 1, "unreleased_times_fixed": False}
    return certificate, summary


def build_model(instance, initial, released, variant, certificate):
    cost.check(variant in ("P1", "P2"), "strengthened model requires P1/P2")
    released = frozenset(released)
    horizon = initial["makespan"]
    rows = {r["operation"]: r for r in initial["assignments"]}
    model = cp_model.CpModel()
    starts, ends, choices = {}, {}, {}
    intervals, load_terms = defaultdict(list), defaultdict(list)
    fixed_load = defaultdict(int)
    for op, options in instance.options.items():
        b = certificate["bounds"][op]
        start = starts[op] = model.new_int_var(*b["start"], f"start_{op}")
        end = ends[op] = model.new_int_var(*b["end"], f"end_{op}")
        row = rows[op]
        model.add_hint(start, row["start"])
        model.add_hint(end, row["end"])
        if op in released:
            eligible = choices[op] = {}
            for machine in certificate["allowed_machines"][op]:
                duration = options[machine]
                present = eligible[machine] = model.new_bool_var(f"use_{op}_{machine}")
                intervals[machine].append(model.new_optional_interval_var(start, duration, end, present, f"interval_{op}_{machine}"))
                model.add_hint(present, int(machine == row["machine"]))
                if variant == "P2":
                    load_terms[machine].append(duration * present)
            model.add_exactly_one(eligible.values())
            if variant == "P2":
                values = [options[m] for m in eligible]
                duration = model.new_int_var(min(values), max(values), f"duration_{op}")
                model.add(duration == sum(options[m] * v for m, v in eligible.items()))
                model.add(end == start + duration)
                model.add_hint(duration, options[row["machine"]])
        else:
            machine = row["machine"]
            intervals[machine].append(model.new_interval_var(start, options[machine], end, f"fixed_{op}_{machine}"))
            fixed_load[machine] += options[machine]
    # Keep original constraint order, objective and all remaining NoOverlap terms.
    for sequence in (*instance.jobs.values(), *_frozen_queues(initial, released).values()):
        for a, b in zip(sequence, sequence[1:]):
            model.add(ends[a] <= starts[b])
    for machine_intervals in intervals.values():
        model.add_no_overlap(machine_intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends.values())
    model.minimize(makespan)
    model.add_hint(makespan, horizon)
    if variant == "P2":
        for machine in sorted(intervals):
            model.add(fixed_load[machine] + sum(load_terms[machine]) <= makespan)
    return model, starts, ends, choices, makespan


def repair_schedule(instance, initial, released, deadline, *, variant, metrics, workers=1, seed=42, presolve=False):
    started = perf_counter()
    horizon = validate_schedule(instance, initial)
    released = tuple(released)
    cost.check(all(type(op) is int and op in instance.options for op in released), "invalid released IDs")
    released = frozenset(released)
    scope = "global" if released == instance.options.keys() else "neighborhood"

    def result(status, schedule=None, lower_bound=None, search_seconds=None, detail=None):
        finished = perf_counter()
        return RepairResult(status, schedule, horizon, lower_bound, scope, finished - started,
                            search_seconds, finished > deadline or status == "TIME_LIMIT", detail)

    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted before preprocessing")
    t = perf_counter()
    certificate, summary = preprocess(instance, initial, released)
    metrics.update(preprocessing_seconds=perf_counter() - t, preprocessing=summary)
    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted during preprocessing")
    model, starts, ends, choices, makespan = build_model(instance, initial, released, variant, certificate)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.cp_model_presolve = presolve
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return result("TIME_LIMIT", detail="budget exhausted during model construction")
    solver.parameters.max_time_in_seconds = remaining
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        lower = solver.best_objective_bound if status == cp_model.UNKNOWN and solver.best_objective_bound > 0 else None
        detail = solver.response_proto.solution_info if status == cp_model.MODEL_INVALID else (
            "unexpected: feasible anchor rejected" if status == cp_model.INFEASIBLE else None)
        return result(status_name, lower_bound=lower, search_seconds=solver.wall_time, detail=detail)
    initial_rows = {r["operation"]: r for r in initial["assignments"]}
    assignments = []
    for op in instance.options:
        machine = next(m for m, v in choices[op].items() if solver.boolean_value(v)) if op in released else initial_rows[op]["machine"]
        assignments.append({"operation": op, "machine": machine, "start": solver.value(starts[op]), "end": solver.value(ends[op])})
    schedule = {"instance": instance.name, "assignments": assignments, "makespan": solver.value(makespan)}
    cost.verify_candidate(instance, initial, released, schedule)
    return result(status_name, schedule=schedule, lower_bound=solver.best_objective_bound, search_seconds=solver.wall_time)
