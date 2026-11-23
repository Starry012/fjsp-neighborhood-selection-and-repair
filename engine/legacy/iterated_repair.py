"""Shared strict-improvement rounds around one incumbent at a time."""

from math import isfinite
from time import perf_counter

from baseline import solve_ect
from fjsp import validate_schedule
from repair import repair_schedule


def search_rounds(instance, make_candidates, rank_candidates, *, total_ms=60_000, repair_ms=200,
                  retry_ms=None, global_reserve_ms=20, return_reserve_ms=5, workers=1, seed=42, initial=None,
                  presolve=True, on_incumbent=None):
    """Rebuild candidates after each improving round; stop on no improvement.

    Callbacks read the round-start schedule. Every repair in that round uses
    that same schedule. Only independently verified, on-time strict gains can
    become incumbents. Stopping without a gain is not a global optimality claim.
    Supplying initial is useful for validation; production runs construct ECT.
    With retry_ms, a complete nonimproving first pass is followed by one longer
    pass over candidates not proved OPTIMAL on time. Both passes use the same
    round-start schedule; proof status never survives an incumbent change.
    """
    if (not isfinite(total_ms) or not isfinite(repair_ms) or total_ms <= 0 or repair_ms <= 0 or
            global_reserve_ms < 0 or return_reserve_ms < 0):
        raise ValueError("use positive finite budgets and nonnegative reserves")
    if retry_ms is not None and (not isfinite(retry_ms) or retry_ms <= repair_ms):
        raise ValueError("retry budget must be finite and greater than the first-pass budget")
    start = perf_counter()
    deadline = start + total_ms / 1000
    work_deadline = deadline - global_reserve_ms / 1000
    rounds, trace = [], []
    current = initial if initial is not None else solve_ect(instance, deadline).schedule
    if current is not None:
        current = {**current, "makespan": validate_schedule(instance, current)}
    initial_finished = perf_counter()
    if current is None or initial_finished > deadline:
        return {"stop": "INITIAL_DEADLINE", "initial": current, "schedule": None,
                "rounds": [], "trace": [], "elapsed_ms": (initial_finished - start) * 1000}
    original = current
    trace.append({"available_ms": (initial_finished - start) * 1000, "makespan": current["makespan"]})
    if on_incumbent is not None:
        on_incumbent(current, (initial_finished - start) * 1000)
    stop = "GLOBAL_DEADLINE"
    while perf_counter() < work_deadline:
        round_initial, best = current, current
        prepared = perf_counter()
        record = {"initial": round_initial, "candidates": [], "order": [], "calls": [], "complete": False}
        rounds.append(record)
        try:
            candidates = make_candidates(instance, round_initial)
            record["candidates"] = candidates
            record["generation_ms"] = (perf_counter() - prepared) * 1000
            if perf_counter() >= work_deadline:
                record["stop"] = "GLOBAL_DEADLINE"
                break
            ranked = perf_counter()
            order = list(rank_candidates(instance, round_initial, candidates))
            if (len(order) != len(candidates) or any(type(i) is not int for i in order)
                    or set(order) != set(range(len(candidates)))):
                raise ValueError("candidate ranking must be a permutation of all indices")
            record.update(order=order, ranking_ms=(perf_counter() - ranked) * 1000)
            complete, passes, pass_id = True, [(repair_ms, order)], 0
            record["passes"] = []
            while pass_id < len(passes):
                budget, pass_order = passes[pass_id]
                record["passes"].append({"budget_ms": budget, "order": list(pass_order)})
                unresolved = []
                for idx in pass_order:
                    call_start = perf_counter()
                    end_by = min(deadline, call_start + budget / 1000)
                    solver_deadline = min(work_deadline, end_by - return_reserve_ms / 1000)
                    if call_start >= solver_deadline:
                        complete = False
                        break
                    released = candidates[idx]["released"]
                    repaired = repair_schedule(instance, round_initial, released, solver_deadline,
                                               workers=workers, seed=seed, presolve=presolve)
                    value = None if repaired.schedule is None else validate_schedule(instance, repaired.schedule)
                    if value is not None and value > round_initial["makespan"]:
                        raise ValueError("repair worsened its validated initial upper bound")
                    finish = perf_counter()
                    on_time = finish <= end_by
                    record["calls"].append({"candidate": idx, "released": released,
                                            "pass": pass_id, "budget_ms": budget,
                                            "start_ms": (call_start - start) * 1000,
                                            "finish_ms": (finish - start) * 1000,
                                            "effective_ms": (end_by - call_start) * 1000,
                                            "on_time": on_time, **vars(repaired)})
                    if repaired.status in ("INFEASIBLE", "MODEL_INVALID"):
                        raise RuntimeError(f"unexpected {repaired.status}: the validated initial schedule belongs to this neighborhood")
                    if not (on_time and repaired.status == "OPTIMAL"):
                        unresolved.append(idx)
                    if value is not None and on_time and value < best["makespan"]:
                        best = repaired.schedule
                        trace.append({"available_ms": (finish - start) * 1000, "makespan": value})
                        if on_incumbent is not None:
                            on_incumbent(best, (finish - start) * 1000)
                    if finish >= deadline:
                        complete = False
                        break
                if not complete:
                    break
                if (pass_id == 0 and retry_ms is not None and unresolved
                        and best["makespan"] == round_initial["makespan"]):
                    passes.append((retry_ms, unresolved))
                pass_id += 1
            improved = best["makespan"] < round_initial["makespan"]
            current = best if improved else round_initial
            record.update(complete=complete, accepted=improved, best_makespan=current["makespan"],
                          finished_ms=(perf_counter() - start) * 1000)
            if not complete:
                record["stop"] = "GLOBAL_DEADLINE"
                break
            if not improved:
                stop = "NO_STRICT_IMPROVEMENT"
                record["stop"] = stop
                break
            record["stop"] = "IMPROVED"
        except Exception as exc:
            # Preserve already validated gains from this partial round as well.
            current = best
            stop = "ERROR"
            record.update(stop=stop, error=f"{type(exc).__name__}: {exc}",
                          best_makespan=current["makespan"], accepted=best["makespan"] < round_initial["makespan"])
            break
    elapsed = (perf_counter() - start) * 1000
    return {"stop": stop, "initial": original, "schedule": current, "rounds": rounds,
            "trace": trace, "elapsed_ms": elapsed, "delivered_on_time": elapsed <= total_ms}
