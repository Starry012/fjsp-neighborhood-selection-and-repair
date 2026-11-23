"""Select few releases using a necessary timing relaxation and certified cuts."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
from time import perf_counter

from ortools.sat.python import cp_model

from fjsp import validate_schedule
from repair import _frozen_queues
from retained_bounds import retained_path_bound


@dataclass(frozen=True)
class FailureProof:
    anchor_sha256: str
    target: int
    released: tuple[int, ...]
    certified_lower_bound: float
    repair_status: str


@dataclass(frozen=True)
class SelectionResult:
    status: str
    released: tuple[int, ...] | None
    release_count: int | None
    count_lower_bound: float | None
    optimistic_makespan: int | None
    target: int
    max_released: int
    seconds: float
    search_seconds: float | None
    budget_exceeded: bool
    machine_pairs: int
    cuts: int
    detail: str | None


def _operations(instance, released):
    released = tuple(released)
    if any(type(o) is not int or o not in instance.options for o in released):
        raise ValueError("release sets must contain known integer operation IDs")
    return tuple(sorted(set(released)))


def anchor_fingerprint(instance, initial):
    value = validate_schedule(instance, initial)
    content = {"instance": instance.name, "makespan": value,
               "jobs": sorted((j, list(q)) for j, q in instance.jobs.items()),
               "options": sorted((o, sorted(ms.items())) for o, ms in instance.options.items()),
               "assignments": sorted((r["operation"], r["machine"], r["start"], r["end"])
                                     for r in initial["assignments"])}
    return sha256(json.dumps(content, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def certify_failure(instance, initial, released, target, repair, *, on_time):
    """Return a cut certificate only when this target is validly excluded.

    The caller supplies the hard, outer timing verdict; a solver's return
    reserve is not itself a new experimental deadline.
    """
    if type(target) is not int or target < 0 or type(on_time) is not bool:
        raise ValueError("use an integer target and a Boolean deadline verdict")
    released = _operations(instance, released)
    current = validate_schedule(instance, initial)
    if repair.initial_makespan != current:
        raise ValueError("repair uses a different initial objective")
    expected_scope = "global" if set(released) == instance.options.keys() else "neighborhood"
    if repair.bound_scope != expected_scope:
        raise ValueError("invalid repair bound scope")
    if repair.status in ("INFEASIBLE", "MODEL_INVALID"):
        raise ValueError("repair must contain its validated feasible initial schedule")
    value = None
    if repair.schedule is not None:
        value = validate_schedule(instance, repair.schedule)
        if value > current or _frozen_queues(repair.schedule, frozenset(released)) != _frozen_queues(initial, frozenset(released)):
            raise ValueError("repair does not belong to the supplied neighborhood")
    bound = repair.lower_bound
    if bound is not None and (not isfinite(bound) or bound > current or (value is not None and bound > value)):
        raise ValueError("invalid repair lower bound")
    if repair.status == "OPTIMAL" and (value is None or bound != value):
        raise ValueError("optimal repair lacks a matching verified objective and bound")
    if not on_time or bound is None or bound <= target:
        return None
    if repair.status not in ("OPTIMAL", "FEASIBLE", "UNKNOWN"):
        return None
    return FailureProof(anchor_fingerprint(instance, initial), target, released, float(bound), repair.status)


def select_release_group(instance, initial, target, max_released, deadline, *, failures=(), workers=1, seed=42,
                         observations=None):
    """Minimize release count subject to retained-path makespan <= target.

    All originally ordered machine pairs are conditional on both endpoints
    remaining frozen. Released-operation resource competition is relaxed.
    INFEASIBLE concerns this target and cap, not the original FJSP's feasibility.
    """
    started = perf_counter()
    current = validate_schedule(instance, initial)
    if type(target) is not int or target < 0 or type(max_released) is not int or max_released < 0:
        raise ValueError("target and release cap must be nonnegative integers")
    if type(workers) is not int or workers <= 0 or type(seed) is not int or seed < 0 or not isfinite(deadline):
        raise ValueError("invalid solver configuration")
    if observations is not None and not isinstance(observations, list):
        raise ValueError("observations must be a list or None")
    failures = tuple(failures)
    anchor = anchor_fingerprint(instance, initial)
    unique = {}
    for proof in failures:
        if not isinstance(proof, FailureProof) or proof.anchor_sha256 != anchor or proof.target != target:
            raise ValueError("failure proof belongs to a different schedule or target")
        released = _operations(instance, proof.released)
        if (proof.released != released or not isfinite(proof.certified_lower_bound)
                or not target < proof.certified_lower_bound <= current
                or proof.repair_status not in ("OPTIMAL", "FEASIBLE", "UNKNOWN")):
            raise ValueError("invalid failure proof")
        unique[released] = proof
    machine_pairs = 0

    def result(status, *, released=None, count_bound=None, optimistic=None, search_seconds=None, detail=None):
        finished = perf_counter()
        return SelectionResult(status, released, None if released is None else len(released), count_bound,
                               optimistic, target, max_released, finished - started, search_seconds,
                               finished > deadline or status == "TIME_LIMIT", machine_pairs, len(unique), detail)

    if perf_counter() >= deadline:
        return result("TIME_LIMIT", detail="budget exhausted before selection model construction")
    model = cp_model.CpModel()
    rows = {r["operation"]: r for r in initial["assignments"]}
    choose, starts, ends = {}, {}, {}
    for o, options in sorted(instance.options.items()):
        choose[o] = model.new_bool_var(f"release_{o}")
        starts[o] = model.new_int_var(0, target, f"optimistic_start_{o}")
        ends[o] = model.new_int_var(0, target, f"optimistic_end_{o}")
        fixed, fastest = options[rows[o]["machine"]], min(options.values())
        model.add(ends[o] == starts[o] + fixed + (fastest - fixed) * choose[o])
    for chain in instance.jobs.values():
        for a, b in zip(chain, chain[1:]):
            model.add(ends[a] <= starts[b])
    for chain in _frozen_queues(initial, frozenset()).values():
        for a, b in combinations(chain, 2):
            model.add(ends[a] <= starts[b]).only_enforce_if([choose[a].Not(), choose[b].Not()])
            machine_pairs += 1
            if machine_pairs % 256 == 0 and perf_counter() >= deadline:
                return result("TIME_LIMIT", detail="budget exhausted while constructing ordered machine pairs")
    model.add(sum(choose.values()) <= max_released)
    for failed in unique:
        model.add(sum(choose[o] for o in choose if o not in failed) >= 1)
    model.minimize(sum(choose.values()))
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return result("TIME_LIMIT", detail="budget exhausted during selection model construction")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    solver.parameters.max_time_in_seconds = remaining
    if observations is None:
        status = solver.solve(model)
    else:
        class ObserveSolutions(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                released = tuple(o for o in choose if self.boolean_value(choose[o]))
                observations.append({"available_seconds": perf_counter() - started,
                                     "solver_seconds": self.wall_time,
                                     "release_count": len(released), "released": released,
                                     "count_lower_bound": self.best_objective_bound})
        # Observation only: never stop search or change its objective/configuration.
        status = solver.solve(model, ObserveSolutions())
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(name, count_bound=solver.best_objective_bound if status == cp_model.UNKNOWN else None,
                      search_seconds=solver.wall_time,
                      detail=solver.response_proto.solution_info if status == cp_model.MODEL_INVALID else None)
    released = tuple(o for o in choose if solver.boolean_value(choose[o]))
    bound = retained_path_bound(instance, initial, released).lower_bound
    count_bound = solver.best_objective_bound
    if (len(released) > max_released or bound > target or count_bound > len(released)
            or (status == cp_model.OPTIMAL and count_bound != len(released))
            or any(set(released) <= set(failed) for failed in unique)):
        raise ValueError("selected group fails independent relaxation or feedback checks")
    return result(name, released=released, count_bound=count_bound, optimistic=bound,
                  search_seconds=solver.wall_time)
