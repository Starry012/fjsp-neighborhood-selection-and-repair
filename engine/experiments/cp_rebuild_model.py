"""Equivalent encodings of the SAME retained-order neighborhood, without presolve."""

from collections import defaultdict, deque
from hashlib import sha256
import json
from time import perf_counter

from ortools.sat.python import cp_model

from experiments import conditional_selection_cost_pilot as cost
from experiments import cp_preprocess_model as base
from fjsp import validate_schedule
from repair import RepairResult, _frozen_queues

VARIANTS = ("P0", "P1", "C", "J", "M", "D", "N", "ALL")
CONTRASTS = (("P0", "P1"), *(("P1", v) for v in VARIANTS[2:]),
             ("J", "D"), ("M", "D"), ("C", "ALL"), ("D", "ALL"))


def chain_lags(sequences, durations):
    """O(n log n) dyadic-span shortcuts. Lag excludes BOTH endpoint operations."""
    lags = {}
    for sequence in sequences:
        prefix = [0]
        for op in sequence:
            prefix.append(prefix[-1] + durations[op])
        span = 2
        while span < len(sequence):
            for i in range(len(sequence) - span):
                j = i + span
                pair = sequence[i], sequence[j]
                lag = prefix[j] - prefix[i + 1]
                lags[pair] = max(lags.get(pair, 0), lag)
            span *= 2
    return lags


def dag_lags(certificate, pairs):
    """Longest path for a bounded endpoint set, not a sum over overlapping paths."""
    d = certificate["minimum_durations"]
    children, indegree = defaultdict(list), dict.fromkeys(d, 0)
    for a, b in certificate["edges"]:
        children[a].append(b)
        indegree[b] += 1
    queue = deque(sorted(a for a, degree in indegree.items() if degree == 0))
    order = []
    while queue:
        a = queue.popleft()
        order.append(a)
        for b in children[a]:
            indegree[b] -= 1
            if indegree[b] == 0:
                queue.append(b)
    cost.check(len(order) == len(d), "retained-order cycle")
    selected = set(pairs)
    # Mixed job/machine paths of length two are cheap additional endpoints.
    for a in order:
        for b in children[a]:
            selected.update((a, c) for c in children[b])
    by_source = defaultdict(set)
    for a, b in selected:
        by_source[a].add(b)
    result = {}
    for a, targets in sorted(by_source.items()):
        distance = {b: 0 for b in children[a]}
        for b in order:
            if b not in distance:
                continue
            for c in children[b]:
                distance[c] = max(distance.get(c, -1), distance[b] + d[b])
        for b in sorted(targets):
            cost.check(b in distance and distance[b] > 0, "invalid transitive endpoint")
            result[a, b] = distance[b]
    return result


def preprocess(instance, initial, released, variant="P1"):
    cost.check(variant in VARIANTS[1:], "unknown strengthened encoding")
    cert, summary = base.preprocess(instance, initial, released)
    if variant == "P1":
        return cert, summary
    released = frozenset(released)
    rows = {r["operation"]: r for r in initial["assignments"]}
    queues = _frozen_queues(initial, released)
    d = cert["minimum_durations"]
    compact = variant in ("C", "ALL")
    singleton = sorted(op for op in released if len(cert["allowed_machines"][op]) == 1) if compact else []
    removed_ends = sorted(set(instance.options) - released | set(singleton)) if compact else []
    job = chain_lags(instance.jobs.values(), d) if variant in ("J", "D", "ALL") else {}
    machine = chain_lags(queues.values(), d) if variant in ("M", "D", "ALL") else {}
    lags = job if variant == "J" else machine if variant == "M" else (
        dag_lags(cert, job.keys() | machine.keys()) if variant in ("D", "ALL") else {})
    affected = {m for op in released for m in cert["allowed_machines"][op]}
    skip = sorted(set(queues) - affected) if variant in ("N", "ALL") else []
    encoding = {"variant": variant, "removed_end_operations": removed_ends,
        "collapsed_singletons": singleton, "skip_no_overlap_machines": skip,
        "job_terminal_objective": compact,
        "lags": [[a, b, lag] for (a, b), lag in sorted(lags.items())]}
    cost.check(all(rows[b]["start"] >= rows[a]["end"] + lag for a, b, lag in encoding["lags"]),
               "shortcut removed the feasible anchor")
    cert["encoding"] = encoding
    summary = {**summary, "encoding_certificate_sha256": sha256(json.dumps(encoding, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "eliminated_end_variables": len(removed_ends), "collapsed_singletons": len(singleton),
        "skipped_no_overlap_machines": len(skip), "added_lag_constraints": len(lags),
        "max_lag": max(lags.values(), default=0), "job_terminal_objective": compact}
    return cert, summary


def build_model(instance, initial, released, variant, certificate):
    if variant == "P1":
        return base.build_model(instance, initial, released, "P1", certificate)
    cost.check(variant in VARIANTS[2:] and certificate["encoding"]["variant"] == variant, "encoding differs")
    released = frozenset(released)
    encoding = certificate["encoding"]
    removed = set(encoding["removed_end_operations"])
    singleton = set(encoding["collapsed_singletons"])
    rows = {r["operation"]: r for r in initial["assignments"]}
    model = cp_model.CpModel()
    starts, ends, choices, intervals = {}, {}, {}, defaultdict(list)
    for op, options in instance.options.items():
        b, row = certificate["bounds"][op], rows[op]
        allowed = certificate["allowed_machines"][op]
        if op in removed:
            cost.check(len(allowed) == 1, "cannot eliminate variable-duration end")
            duration = options[allowed[0]]
            lo = max(b["start"][0], b["end"][0] - duration)
            hi = min(b["start"][1], b["end"][1] - duration)
            start = starts[op] = model.new_int_var(lo, hi, f"start_{op}")
            end = ends[op] = start + duration
        else:
            start = starts[op] = model.new_int_var(*b["start"], f"start_{op}")
            end = ends[op] = model.new_int_var(*b["end"], f"end_{op}")
        model.add_hint(start, row["start"])
        if op not in removed:
            model.add_hint(end, row["end"])
        if op in released and op not in singleton:
            eligible = choices[op] = {}
            for machine in allowed:
                present = eligible[machine] = model.new_bool_var(f"use_{op}_{machine}")
                intervals[machine].append(model.new_optional_interval_var(start, options[machine], end, present, f"interval_{op}_{machine}"))
                model.add_hint(present, int(machine == row["machine"]))
            model.add_exactly_one(eligible.values())
        else:
            machine = allowed[0]
            if op in singleton:
                choices[op] = {machine: True}
            interval = (model.new_fixed_size_interval_var(start, options[machine], f"fixed_{op}_{machine}") if op in removed else
                        model.new_interval_var(start, options[machine], end, f"fixed_{op}_{machine}"))
            intervals[machine].append(interval)
    # Critically, singleton released operations stay RELEASED in these queues.
    for sequence in (*instance.jobs.values(), *_frozen_queues(initial, released).values()):
        for a, b in zip(sequence, sequence[1:]):
            model.add(ends[a] <= starts[b])
    for machine, items in intervals.items():
        if machine not in encoding["skip_no_overlap_machines"]:
            model.add_no_overlap(items)
    makespan = model.new_int_var(0, initial["makespan"], "makespan")
    objective_ends = [ends[job[-1]] for job in instance.jobs.values()] if encoding["job_terminal_objective"] else list(ends.values())
    model.add_max_equality(makespan, objective_ends)
    model.minimize(makespan)
    model.add_hint(makespan, initial["makespan"])
    for a, b, lag in encoding["lags"]:
        model.add(starts[b] >= ends[a] + lag)
    return model, starts, ends, choices, makespan


def repair_schedule(instance, initial, released, deadline, *, variant, metrics, workers=1, seed=42, presolve=False):
    # Preserve P1's exact implementation and model ordering as a fresh control.
    if variant == "P1":
        return base.repair_schedule(instance, initial, released, deadline, variant="P1", metrics=metrics,
                                    workers=workers, seed=seed, presolve=presolve)
    started = perf_counter()
    horizon = validate_schedule(instance, initial)
    released = frozenset(released)
    scope = "global" if released == instance.options.keys() else "neighborhood"
    def result(status, schedule=None, lower_bound=None, search_seconds=None, detail=None):
        now = perf_counter()
        return RepairResult(status, schedule, horizon, lower_bound, scope, now - started,
                            search_seconds, now > deadline or status == "TIME_LIMIT", detail)
    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted before preprocessing")
    t = perf_counter()
    cert, summary = preprocess(instance, initial, released, variant)
    metrics.update(preprocessing_seconds=perf_counter() - t, preprocessing=summary)
    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted during preprocessing")
    model, starts, ends, choices, makespan = build_model(instance, initial, released, variant, cert)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.cp_model_presolve = presolve
    solver.parameters.relative_gap_limit = solver.parameters.absolute_gap_limit = 0
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return result("TIME_LIMIT", detail="budget exhausted during model construction")
    solver.parameters.max_time_in_seconds = remaining
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        lower = solver.best_objective_bound if status == cp_model.UNKNOWN and solver.best_objective_bound > 0 else None
        return result(solver.status_name(status), lower_bound=lower, search_seconds=solver.wall_time)
    rows = {r["operation"]: r for r in initial["assignments"]}
    assignments = []
    for op in instance.options:
        machine = next(m for m, v in choices[op].items() if v is True or solver.boolean_value(v)) if op in released else rows[op]["machine"]
        assignments.append({"operation": op, "machine": machine, "start": solver.value(starts[op]), "end": solver.value(ends[op])})
    schedule = {"instance": instance.name, "assignments": assignments, "makespan": solver.value(makespan)}
    cost.verify_candidate(instance, initial, released, schedule)
    return result(solver.status_name(status), schedule, solver.best_objective_bound, solver.wall_time)
