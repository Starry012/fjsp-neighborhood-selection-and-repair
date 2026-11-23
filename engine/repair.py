"""CP-SAT repair for a supplied set of released operations."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

from ortools.sat.python import cp_model

from fjsp import Instance, validate_schedule


@dataclass(frozen=True)
class RepairResult:
    status: str
    schedule: dict | None
    initial_makespan: int
    lower_bound: float | None
    bound_scope: str
    seconds: float
    search_seconds: float | None
    budget_exceeded: bool
    detail: str | None


def _frozen_queues(schedule: dict, released: frozenset[int]) -> dict[int, tuple[int, ...]]:
    queues = defaultdict(list)
    for row in sorted(schedule["assignments"], key=lambda row: row["start"]):
        if row["operation"] not in released:
            queues[row["machine"]].append(row["operation"])
    return {machine: tuple(operations) for machine, operations in queues.items()}


def repair_schedule(
    instance: Instance, initial: dict, released: Iterable[int], deadline: float, *, workers: int = 1, seed: int = 42,
    presolve: bool = True, on_incumbent=None
) -> RepairResult:
    """Optimize the supplied neighborhood until an absolute perf_counter deadline.

    Non-released operations retain their machines and mutual machine order.
    All times can change. A missing solver schedule is returned as None.
    """
    started = perf_counter()
    horizon = validate_schedule(instance, initial)
    released = tuple(released)
    if any(type(op) is not int or op not in instance.options for op in released):
        raise ValueError("released operations must be known operation IDs")
    released = frozenset(released)
    scope = "global" if released == instance.options.keys() else "neighborhood"

    def result(status, *, schedule=None, lower_bound=None, search_seconds=None, detail=None):
        finished = perf_counter()
        return RepairResult(status, schedule, horizon, lower_bound, scope, finished - started,
                            search_seconds, finished > deadline or status == "TIME_LIMIT", detail)

    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted before model construction")

    incumbent = {row["operation"]: row for row in initial["assignments"]}
    frozen_queues = _frozen_queues(initial, released)
    model = cp_model.CpModel()
    starts, ends, choices = {}, {}, {}
    intervals = defaultdict(list)
    for op, options in instance.options.items():
        # The validated initial schedule remains feasible, so the horizon is a safe cutoff.
        start = starts[op] = model.new_int_var(0, horizon, f"start_{op}")
        end = ends[op] = model.new_int_var(0, horizon, f"end_{op}")
        row = incumbent[op]
        model.add_hint(start, row["start"])
        model.add_hint(end, row["end"])
        if op in released:
            eligible = choices[op] = {}
            for machine, duration in options.items():
                present = eligible[machine] = model.new_bool_var(f"use_{op}_{machine}")
                intervals[machine].append(model.new_optional_interval_var(
                    start, duration, end, present, f"interval_{op}_{machine}"
                ))
                model.add_hint(present, int(machine == row["machine"]))
            model.add_exactly_one(eligible.values())
        else:
            machine = row["machine"]
            intervals[machine].append(model.new_interval_var(start, options[machine], end, f"fixed_{op}_{machine}"))

    for sequence in (*instance.jobs.values(), *frozen_queues.values()):
        for before, after in zip(sequence, sequence[1:]):
            model.add(ends[before] <= starts[after])
    for machine_intervals in intervals.values():
        model.add_no_overlap(machine_intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends.values())
    model.minimize(makespan)
    model.add_hint(makespan, horizon)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.cp_model_presolve = presolve
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return result("TIME_LIMIT", detail="budget exhausted during model construction")
    def decode(reader):
        assignments = []
        for op in instance.options:
            if op in released:
                machine = next(m for m, present in choices[op].items() if reader.boolean_value(present))
            else:
                machine = incumbent[op]["machine"]
            assignments.append({"operation": op, "machine": machine, "start": reader.value(starts[op]),
                                "end": reader.value(ends[op])})
        schedule = {"instance": instance.name, "assignments": assignments, "makespan": reader.value(makespan)}
        verified = validate_schedule(instance, schedule)
        if verified > horizon or _frozen_queues(schedule, released) != frozen_queues:
            raise ValueError("repair result violates the neighborhood restrictions")
        return schedule

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            if perf_counter() < deadline:
                on_incumbent(decode(self))
            else:
                self.stop_search()

    solver.parameters.max_time_in_seconds = remaining
    status = solver.solve(model) if on_incumbent is None else solver.solve(model, Incumbent())
    status_name = solver.status_name(status)
    search_seconds = solver.wall_time
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        lower_bound = solver.best_objective_bound if status == cp_model.UNKNOWN and solver.best_objective_bound > 0 else None
        detail = None
        if status == cp_model.MODEL_INVALID:
            detail = solver.response_proto.solution_info
        elif status == cp_model.INFEASIBLE:
            detail = "unexpected: the validated initial schedule belongs to this neighborhood"
        return result(status_name, lower_bound=lower_bound, search_seconds=search_seconds, detail=detail)

    schedule = decode(solver)
    return result(status_name, schedule=schedule, lower_bound=solver.best_objective_bound, search_seconds=search_seconds)
