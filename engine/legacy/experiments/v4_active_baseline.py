"""Frozen, serial 60-second development comparison from timed local input reads."""

import argparse
from collections import Counter
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

# Set before importing numerical libraries. Affinity also limits native helper threads.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import torch

from baseline import load_pinned_json, solve_ect
from experiments.v3_ranker_dev import _save, _sha
from experiments.v3_ranker_live import _rank
from experiments.v4_multiround import _load_v4_ranker, audit as audit_old, make_pool
from fjsp import parse_instance, validate_schedule
from iterated_repair import search_rounds
from repair import RepairResult, _frozen_queues, repair_schedule
from retained_bounds import retained_path_bound


CONFIG = Path("experiments/v4_active_baseline.json")
METHODS = ("original-strong", "active-strict", "full-cp-sat-hinted")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def configure_cpu():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    affinity = None
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        check(kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)), "cannot read CPU affinity")
        selected = process_mask.value & -process_mask.value
        check(kernel.SetProcessAffinityMask(handle, selected), "cannot set single-CPU affinity")
        affinity = {"original_mask": process_mask.value, "selected_mask": selected,
                    "logical_processor": selected.bit_length() - 1}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "affinity": affinity,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "dependencies": {k: version(k) for k in ("ortools", "torch", "numpy", "pandas", "protobuf")}}


def verify_repair(instance, initial, released, result):
    check(result.status not in ("INFEASIBLE", "MODEL_INVALID"), "known feasible anchor was rejected")
    expected_scope = "global" if set(released) == set(instance.options) else "neighborhood"
    check(result.bound_scope == expected_scope, "incorrect bound scope")
    check(result.initial_makespan == validate_schedule(instance, initial), "stale repair anchor")
    if result.schedule is None:
        return None
    value = validate_schedule(instance, result.schedule)
    check(value <= initial["makespan"], "repair worsens its anchor")
    check(_frozen_queues(result.schedule, frozenset(released)) == _frozen_queues(initial, frozenset(released)), "retained order changed")
    check(result.lower_bound is None or result.lower_bound <= value, "invalid lower bound")
    check(result.status != "OPTIMAL" or result.lower_bound == value, "invalid optimality claim")
    return value


def credit(result, budget_ms):
    if result["delivered_on_time"]:
        return result["schedule"]["makespan"]
    if result.get("fallback_written_ms", math.inf) <= budget_ms:
        return result["initial"]["makespan"]
    return None


def execute(input_path, case, method, seed, cfg, base_cfg, delivery, *,
            trial_started=None, trial_deadline=None, prepared_initial=None, on_incumbent=None,
            ranker_loader=None):
    started = perf_counter() if trial_started is None else trial_started
    deadline = started + cfg["total_ms"] / 1000
    if trial_deadline is not None:
        deadline = min(deadline, trial_deadline)
    effective_ms = (deadline - started) * 1000
    work_deadline = deadline - cfg["final_reserve_ms"] / 1000
    elapsed = lambda: (perf_counter() - started) * 1000
    result = {"instance": case["instance"], "method": method, "seed": seed,
              "initial": None, "schedule": None, "trace": [], "random_calls": [],
              "old_result": None, "full_result": None, "stop": "INITIAL_DEADLINE"}
    instance = current = None
    try:
        instance = parse_instance(load_pinned_json(input_path, case["sha256"]))
        check(len(instance.options) == case["operations"], "instance size changed")
        result["input_finish_ms"] = elapsed()
        initial = prepared_initial if prepared_initial is not None else solve_ect(instance, work_deadline).schedule
        check(initial is not None, "ECT did not produce a schedule")
        initial = {**initial, "makespan": validate_schedule(instance, initial)}
        current = result["initial"] = initial
        _save(delivery.with_suffix(".fallback.json"), initial)
        result["fallback_written_ms"] = elapsed()
        result["trace"].append({"available_ms": result["fallback_written_ms"], "makespan": initial["makespan"], "source": "ECT"})
        if on_incumbent is not None:
            on_incumbent(initial, result["fallback_written_ms"], "ECT")
        if perf_counter() >= work_deadline:
            raise TimeoutError("initialization consumed the search budget")
        if method == "full-cp-sat-hinted":
            call_start = elapsed()
            full = repair_schedule(instance, initial, tuple(instance.options), work_deadline, workers=1, seed=seed,
                                   presolve=cfg["cp_model_presolve"])
            result["full_result"] = {"repair": asdict(full), "start_ms": call_start, "finish_ms": elapsed(), "on_time": elapsed() <= effective_ms}
            value = verify_repair(instance, initial, tuple(instance.options), full)
            finish = elapsed()
            on_time = finish <= effective_ms
            result["full_result"] = {"repair": asdict(full), "start_ms": call_start, "finish_ms": finish, "on_time": on_time}
            if value is not None and on_time and value < current["makespan"]:
                current = full.schedule
                result["trace"].append({"available_ms": finish, "makespan": value, "source": "full-cp"})
                if on_incumbent is not None:
                    on_incumbent(current, finish, "full-cp")
            result["stop"] = full.status
        else:
            support_start = elapsed()
            labels = load_pinned_json(Path("experiments/v3_dev_replay.json"), base_cfg["label_config_sha256"])
            model, weights = (ranker_loader or _load_v4_ranker)(base_cfg)
            result["weights"] = weights
            result["support_ms"] = elapsed() - support_start
            old_start = elapsed()
            remaining_ms = max(0.001, (work_deadline - perf_counter()) * 1000)
            result["old_start_ms"], result["old_total_ms"] = old_start, remaining_ms
            old = search_rounds(instance,
                lambda i, s: make_pool(i, s, labels, True),
                lambda i, s, c: _rank(i, s, c, "tabular", cfg["ranking_seed"], cfg["ranking_budget_input_ms"], model, base_cfg),
                initial=initial, total_ms=remaining_ms, repair_ms=cfg["repair_ms"],
                global_reserve_ms=cfg["old_global_reserve_ms"], return_reserve_ms=cfg["old_return_reserve_ms"],
                workers=1, seed=seed, presolve=cfg["cp_model_presolve"],
                on_incumbent=(None if on_incumbent is None else
                    lambda s, t: on_incumbent(s, old_start + t, "old-local")))
            old_finish = elapsed()
            old_credited = old_finish <= effective_ms - cfg["final_reserve_ms"]
            result.update(old_result=old, old_finish_ms=old_finish, old_credited=old_credited, stop=old["stop"])
            if old_credited and old["schedule"] is not None:
                current = old["schedule"]
                result["trace"].extend({"available_ms": old_start + t["available_ms"], "makespan": t["makespan"], "source": "old-local"} for t in old["trace"][1:])
            if method == "active-strict" and old["stop"] != "ERROR":
                rng = Random(seed)
                operations = sorted(instance.options)
                while perf_counter() + cfg["minimum_random_remaining_ms"] / 1000 < work_deadline:
                    prepared = elapsed()
                    released = tuple(sorted(rng.sample(operations, cfg["random_size"])))
                    lower = retained_path_bound(instance, current, released).lower_bound
                    call = {"released": released, "anchor": current["makespan"], "retained_lower_bound": lower,
                            "preparation_start_ms": prepared, "preparation_ms": elapsed() - prepared,
                            "status": "BOUND_REJECTED", "repair": None}
                    if lower < current["makespan"]:
                        call_start = perf_counter()
                        end_by = min(work_deadline, call_start + cfg["repair_ms"] / 1000)
                        if call_start + cfg["minimum_random_remaining_ms"] / 1000 >= end_by:
                            break
                        repaired = repair_schedule(instance, current, released, end_by - cfg["random_reserve_ms"] / 1000, workers=1, seed=seed,
                                                   presolve=cfg["cp_model_presolve"])
                        call.update(status=repaired.status, repair=asdict(repaired), start_ms=(call_start - started) * 1000,
                                    finish_ms=elapsed(), end_ms=(end_by - started) * 1000)
                        call["on_time"] = call["finish_ms"] <= call["end_ms"]
                        result["random_calls"].append(call)
                        value = verify_repair(instance, current, released, repaired)
                        finished = perf_counter()
                        on_time = finished <= end_by
                        call.update(status=repaired.status, repair=asdict(repaired), start_ms=(call_start - started) * 1000,
                                    finish_ms=(finished - started) * 1000, end_ms=(end_by - started) * 1000, on_time=on_time)
                        if value is not None and on_time and value < current["makespan"]:
                            current = repaired.schedule
                            result["trace"].append({"available_ms": call["finish_ms"], "makespan": value, "source": "random-64"})
                            if on_incumbent is not None:
                                on_incumbent(current, call["finish_ms"], "random-64")
                    else:
                        result["random_calls"].append(call)
                result["stop"] = "BUDGET_FINISHED"
    except Exception as exc:
        result.update(stop="ERROR", error=f"{type(exc).__name__}: {exc}")
    if instance is not None and current is not None:
        validate_schedule(instance, current)
        result["schedule"] = current
        _save(delivery, current)
    result["elapsed_ms"] = elapsed()
    result["delivered_on_time"] = result["schedule"] is not None and result["elapsed_ms"] <= effective_ms
    result["credited_makespan"] = credit(result, effective_ms)
    return result


def summarize(plan, rows):
    cases = []
    for case in plan["cases"]:
        selected = [r for r in rows if r["instance"] == case["instance"]]
        methods = {}
        for method in METHODS:
            trials = sorted((r for r in selected if r["method"] == method), key=lambda r: r["seed"])
            values = [r["credited_makespan"] for r in trials]
            methods[method] = {"values": values, "median": median(values) if len(values) == len(plan["config"]["seeds"]) and None not in values else None,
                               "elapsed_ms": [r["elapsed_ms"] for r in trials]}
        gains = {}
        for opponent in ("original-strong", "full-cp-sat-hinted"):
            a, b = methods["active-strict"]["values"], methods[opponent]["values"]
            paired = [100 * (y - x) / y for x, y in zip(a, b)] if len(a) == len(b) == len(plan["config"]["seeds"]) and None not in a + b else []
            gains[opponent] = {"paired_pct": paired, "median_pct": median(paired) if paired else None}
        cases.append({"instance": case["instance"], "operations": case["operations"], "methods": methods, "active_gains": gains})
    comparison = {}
    for opponent in ("original-strong", "full-cp-sat-hinted"):
        values = [c["active_gains"][opponent]["median_pct"] for c in cases]
        comparison[opponent] = {"instance_median_pct": median(values) if None not in values else None,
            "wins_ties_losses": [sum(v is not None and v > 0 for v in values), sum(v == 0 for v in values), sum(v is not None and v < 0 for v in values)],
            "by_scale": {str(n): median(c["active_gains"][opponent]["median_pct"] for c in cases if c["operations"] == n) for n in sorted({c["operations"] for c in cases})} if None not in values else {}}
    return {"runs": len(rows), "cases": cases, "comparison": comparison,
            "late_runs": sum(not r["delivered_on_time"] for r in rows), "error_runs": sum(r["stop"] == "ERROR" for r in rows),
            "missing_quality": sum(r["credited_makespan"] is None for r in rows)}


def prepare(output, environment):
    cfg = json.loads(CONFIG.read_bytes())
    check(version("ortools") == "9.15.6755", "use the isolated V4 runtime")
    base = load_pinned_json(Path(cfg["source_config"]), cfg["source_config_sha256"])
    cases = [next(c for c in base["cases"] if c["instance"] == name) for name in cfg["case_names"]]
    labels = load_pinned_json(Path("experiments/v3_dev_replay.json"), base["label_config_sha256"])
    _, weights = _load_v4_ranker(base)
    tasks = list(product(cfg["case_names"], METHODS, cfg["seeds"]))
    Random(cfg["order_seed"]).shuffle(tasks)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    (output / "inputs").mkdir()
    (output / "delivery").mkdir()
    sources = sorted([*Path(".").glob("*.py"), *Path("experiments").glob("*.py"), CONFIG,
                      Path(cfg["source_config"]), Path("experiments/v3_dev_replay.json"), Path("experiments/v3_ranker_dev.json"), Path("requirements-v4.txt"),
                      Path("experiments/V4_ACTIVE_BASELINE.md"), Path("tests/test_v4_active_baseline.py")])
    for source in sources:
        target = output / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for case in cases:
        path = Path(labels["instance_dir"]) / f"{case['instance']}.json"
        check(_sha(path) == case["sha256"], "local instance changed")
        (output / "inputs" / path.name).write_bytes(path.read_bytes())
    plan = {"config": cfg, "base_config": base, "cases": cases, "tasks": tasks, "environment": environment,
            "created_utc": datetime.now(timezone.utc).isoformat(), "weight_sha256": weights,
            "config_sha256": _sha(CONFIG), "code_sha256": {p.as_posix(): _sha(p) for p in sources},
            "baseline_note": "Both local methods retain frozen learned tabular-42 ranking; the intervention is strict random-64 continuation, not removal of all learning.",
            "old_results_reused": False}
    _save(output / "plan.json", plan)
    return plan


def run(output, environment):
    plan = prepare(output, environment)
    cases = {c["instance"]: c for c in plan["cases"]}
    rows = []
    for index, (name, method, seed) in enumerate(plan["tasks"], 1):
        key = f"{name}-{method}-{seed}"
        result = execute(output / "inputs" / f"{name}.json", cases[name], method, seed, plan["config"], plan["base_config"], output / "delivery" / f"{key}.json")
        _save(output / f"{key}.json", result)
        row = {k: result[k] for k in ("instance", "method", "seed", "stop", "elapsed_ms", "delivered_on_time", "credited_makespan")}
        row["file"] = f"{key}.json"
        rows.append(row)
        _save(output / "runs.json", rows)
        print(json.dumps({"completed": index, "planned": len(plan["tasks"]), **row}), flush=True)
    summary = summarize(plan, rows)
    summary["files_sha256"] = {p.relative_to(output).as_posix(): _sha(p) for p in [*output.glob("*.json"), *(output / "delivery").glob("*.json")] if p.name not in ("summary.json", "audit.json")}
    _save(output / "summary.json", summary)
    audit(output)


def audit(output):
    plan = json.loads((output / "plan.json").read_bytes())
    rows = json.loads((output / "runs.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    cfg = plan["config"]
    check([(r["instance"], r["method"], r["seed"]) for r in rows] == [tuple(t) for t in plan["tasks"]], "missing or reordered tasks")
    for path, digest in summary["files_sha256"].items():
        check(_sha(output / path) == digest, f"output changed: {path}")
    for path, digest in plan["code_sha256"].items():
        check(_sha(output / "source" / path) == digest, f"snapshot changed: {path}")
    labels = load_pinned_json(Path("experiments/v3_dev_replay.json"), plan["base_config"]["label_config_sha256"])
    instances = {c["instance"]: parse_instance(load_pinned_json(output / "inputs" / f"{c['instance']}.json", c["sha256"])) for c in plan["cases"]}
    starts, totals, statuses = {}, Counter(), Counter()
    for row in rows:
        result = json.loads((output / row["file"]).read_bytes())
        instance = instances[row["instance"]]
        initial = current = result["initial"]
        check(initial is not None, "missing ECT fallback")
        validate_schedule(instance, initial)
        if instance.name in starts:
            check(initial == starts[instance.name], "methods did not share the same ECT schedule")
        starts[instance.name] = initial
        expected_trace = [result["trace"][0]]
        check(expected_trace[0] == {"available_ms": result["fallback_written_ms"], "makespan": initial["makespan"], "source": "ECT"}, "invalid initial trace")
        old = result["old_result"]
        if old is not None:
            check(old["initial"] == initial, "old phase did not start at ECT")
            audit_cfg = {**plan["base_config"], "total_ms": result["old_total_ms"]}
            verified = audit_old(instance, old, labels, True, audit_cfg)
            totals.update({"old_calls": verified["calls"], "returned_verified": verified["returned_and_verified"]})
            statuses.update(verified["statuses"])
            check(result["weights"] == plan["weight_sha256"], "ranking weights changed")
            check(result["old_credited"] == (result["old_finish_ms"] <= cfg["total_ms"] - cfg["final_reserve_ms"]), "old phase credit mismatch")
            if result["old_credited"]:
                current = old["schedule"]
                expected_trace.extend({"available_ms": result["old_start_ms"] + t["available_ms"], "makespan": t["makespan"], "source": "old-local"} for t in old["trace"][1:])
        rng = Random(row["seed"])
        check(row["method"] == "active-strict" or not result["random_calls"], "control performed random continuation")
        for call in result["random_calls"]:
            released = sorted(rng.sample(sorted(instance.options), cfg["random_size"]))
            check(call["released"] == released and call["anchor"] == current["makespan"], "random stream or anchor mismatch")
            lower = retained_path_bound(instance, current, released).lower_bound
            check(call["retained_lower_bound"] == lower, "invalid retained bound")
            if call["repair"] is None:
                check(lower >= current["makespan"] and call["status"] == "BOUND_REJECTED", "invalid bound rejection")
                totals["bound_rejected"] += 1
            else:
                repaired = RepairResult(**call["repair"])
                value = verify_repair(instance, current, released, repaired)
                totals["random_calls"] += 1
                totals["returned_verified"] += value is not None
                totals["late_local_calls"] += not call["on_time"]
                statuses[repaired.status] += 1
                check(call["on_time"] == (call["finish_ms"] <= call["end_ms"]), "random call deadline mismatch")
                if value is not None and call["on_time"] and value < current["makespan"]:
                    current = repaired.schedule
                    expected_trace.append({"available_ms": call["finish_ms"], "makespan": value, "source": "random-64"})
        full = result["full_result"]
        if full is not None:
            check(row["method"] == "full-cp-sat-hinted" and old is None, "incorrect full-model arm")
            repaired = RepairResult(**full["repair"])
            value = verify_repair(instance, initial, tuple(instance.options), repaired)
            totals["full_calls"] += 1
            totals["returned_verified"] += value is not None
            statuses[repaired.status] += 1
            check(full["on_time"] == (full["finish_ms"] <= cfg["total_ms"]), "full deadline mismatch")
            if value is not None and full["on_time"] and value < current["makespan"]:
                current = repaired.schedule
                expected_trace.append({"available_ms": full["finish_ms"], "makespan": value, "source": "full-cp"})
        check(current == result["schedule"] and result["trace"] == expected_trace, "final schedule or quality trace mismatch")
        validate_schedule(instance, current)
        key = Path(row["file"])
        check(json.loads((output / "delivery" / key).read_bytes()) == current, "delivered file differs")
        check(json.loads((output / "delivery" / key.with_suffix(".fallback.json")).read_bytes()) == initial, "fallback differs")
        check(result["delivered_on_time"] == (result["elapsed_ms"] <= cfg["total_ms"]), "delivery flag mismatch")
        check(credit(result, cfg["total_ms"]) == row["credited_makespan"], "incorrect credited score")
        for field in ("instance", "method", "seed", "stop", "elapsed_ms", "delivered_on_time", "credited_makespan"):
            check(row[field] == result[field], f"row differs: {field}")
    check(summarize(plan, rows) == {k: v for k, v in summary.items() if k != "files_sha256"}, "summary mismatch")
    report = {"runs_audited": len(rows), "same_ect_cases": len(starts), "totals": dict(totals), "statuses": dict(statuses),
              "summary_sha256": _sha(output / "summary.json"), "completed_utc": datetime.now(timezone.utc).isoformat()}
    _save(output / "audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    environment = configure_cpu()
    (audit(args.output) if args.audit else run(args.output, environment))
