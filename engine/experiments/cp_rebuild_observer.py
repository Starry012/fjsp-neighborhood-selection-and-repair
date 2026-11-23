"""Common instrumentation and projected callback decoding for eight encodings."""

from dataclasses import asdict
from hashlib import sha256
import math
from time import perf_counter
from unittest.mock import patch

from ortools.sat.python import cp_model

from experiments import conditional_selection_cost_pilot as cost
from experiments import cp_rebuild_model
from repair import repair_schedule as original_repair


def observed_repair(instance, initial, released, deadline, *, seed, variant, publish, request_started):
    cost.check(variant in cp_rebuild_model.VARIANTS, "unknown repair variant")
    entered = perf_counter()
    original_solve = cp_model.CpSolver.solve
    released = frozenset(released)
    initial_rows = {r["operation"]: r for r in initial["assignments"]}
    best, captured = initial["makespan"], {}
    metrics = {"variant": variant, "preprocessing_seconds": 0., "preprocessing": None,
               "prepare_to_solve_seconds": None, "other_prepare_seconds": None, "solve_started_after_request_seconds": None,
               "native_call_seconds": None, "native_stats": None, "model_counts": None,
               "first_worker_improvement_seconds": None, "first_native_solution_seconds": None}
    stats = {"callbacks": 0, "callback_seconds": 0., "published": 0}

    class Progress(cp_model.CpSolverSolutionCallback):
        def __init__(self, model):
            super().__init__()
            self.model, self.variables = model, None

        def on_solution_callback(self):
            nonlocal best
            begun = perf_counter()
            stats["callbacks"] += 1
            try:
                if metrics["first_native_solution_seconds"] is None:
                    metrics["first_native_solution_seconds"] = begun - request_started
                value = round(self.objective_value)
                if value >= best:
                    return
                if self.variables is None:
                    self.variables = {v.name: self.model.get_int_var_from_proto_index(i) for i, v in enumerate(self.model.proto.variables)}
                rows = []
                for op, options in instance.options.items():
                    machine = initial_rows[op]["machine"]
                    if op in released:
                        present = [m for m in options if f"use_{op}_{m}" in self.variables]
                        if present:
                            machine = next(m for m in present if self.boolean_value(self.variables[f"use_{op}_{m}"]))
                        else:
                            cost.check(variant in ("C", "ALL"), "missing machine choices")
                            # A collapsed singleton retains the anchor's machine,
                            # but NOT its position in the retained machine queue.
                    start = self.value(self.variables[f"start_{op}"])
                    end = start + options[machine]
                    if f"end_{op}" in self.variables:
                        cost.check(self.value(self.variables[f"end_{op}"]) == end, "callback interval differs")
                    rows.append({"operation": op, "machine": machine, "start": start, "end": end})
                schedule = {"instance": instance.name, "assignments": rows, "makespan": value}
                cost.check(cost.verify_candidate(instance, initial, released, schedule) == value, "callback value differs")
                lower = self.best_objective_bound
                cost.check(math.isfinite(lower) and lower <= value + 1e-6, "invalid callback bound")
                verified = perf_counter()
                if metrics["first_worker_improvement_seconds"] is None:
                    metrics["first_worker_improvement_seconds"] = begun - request_started
                credited = publish(schedule, "incumbent", {"found_at": begun, "worker_verified_at": verified,
                    "worker_validation_seconds": verified - begun, "lower_bound": lower, "bound_scope": "neighborhood"})
                best = value
                stats["published"] += credited is not False
            finally:
                stats["callback_seconds"] += perf_counter() - begun

    def intercepted(solver, model, solution_callback=None):
        cost.check(solution_callback is None, "unexpected existing callback")
        captured["model"] = model
        proto = model.proto
        metrics["model_counts"] = {"variables": len(proto.variables), "constraints": len(proto.constraints),
            "intervals": sum(c.has_interval() for c in proto.constraints),
            "optional_intervals": sum(c.has_interval() and bool(c.enforcement_literal) for c in proto.constraints),
            "linear_constraints": sum(c.has_linear() for c in proto.constraints),
            "no_overlap_constraints": sum(c.has_no_overlap() for c in proto.constraints)}
        now = perf_counter()
        metrics["prepare_to_solve_seconds"] = now - entered
        metrics["other_prepare_seconds"] = now - entered - metrics["preprocessing_seconds"]
        metrics["solve_started_after_request_seconds"] = now - request_started
        # Common instrumentation is charged before giving the solver its remaining
        # budget, including P0. A tiny exhausted solve permits a valid UNKNOWN response.
        solver.parameters.max_time_in_seconds = max(1e-9, deadline - perf_counter())
        result = original_solve(solver, model, Progress(model))
        metrics["native_call_seconds"] = perf_counter() - now
        response = solver.response_proto
        metrics["native_stats"] = {name: getattr(response, name) for name in
            ("num_booleans", "num_branches", "num_conflicts", "num_binary_propagations", "num_integer_propagations", "deterministic_time")}
        return result

    with patch.object(cp_model.CpSolver, "solve", intercepted):
        if variant == "P0":
            result = original_repair(instance, initial, released, deadline, workers=1, seed=seed, presolve=False)
        else:
            result = cp_rebuild_model.repair_schedule(instance, initial, released, deadline,
                variant=variant, metrics=metrics, workers=1, seed=seed, presolve=False)
    if result.schedule is not None:
        cost.verify_candidate(instance, initial, released, result.schedule)
        publish(result.schedule, "final", {"worker_verified_at": perf_counter(), "lower_bound": result.lower_bound,
                                          "bound_scope": result.bound_scope, "status": result.status})
    cost.check(result.status not in ("INFEASIBLE", "MODEL_INVALID"), "preprocessing or solver rejected a feasible anchor")
    model_hash = sha256(str(captured["model"].proto).encode()).hexdigest() if captured else None
    metrics["observed_seconds"] = perf_counter() - entered
    return {"result": asdict(result), "model_sha256": model_hash, "metrics": metrics, **stats}
