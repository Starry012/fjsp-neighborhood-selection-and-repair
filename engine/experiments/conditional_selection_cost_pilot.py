"""Bounded CPU-only cost diagnostic; no policy training or STOP labels."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from random import Random
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter
import traceback
from uuid import uuid4


CONFIG = Path("experiments/conditional_selection_cost_pilot.plan.json")
EXTRA_SOURCES = ["experiments/conditional_selection_cost_pilot.py",
                 "tests/test_conditional_selection_cost_pilot.py",
                 "experiments/resched_qualification.py", "requirements-py314.txt", "run-python.ps1"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Unique files and post-exit reads avoid mutable worker mailboxes."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def pinned(path, expected):
    raw = Path(path).read_bytes()
    check(sha256(raw).hexdigest() == expected, f"fingerprint mismatch: {path}")
    return json.loads(raw)


def frozen_queues(schedule, released):
    queues = defaultdict(list)
    for row in schedule["assignments"]:
        if row["operation"] not in released:
            queues[row["machine"]].append((row["start"], row["operation"]))
    return {machine: tuple(op for _, op in sorted(rows)) for machine, rows in queues.items()}


def verify_candidate(instance, initial, released, schedule):
    from fjsp import validate_schedule
    value = validate_schedule(instance, schedule)
    check(value <= initial["makespan"], "candidate exceeds anchor horizon")
    check(frozen_queues(initial, set(released)) == frozen_queues(schedule, set(released)),
          "candidate changed a frozen machine or relative order")
    return value


def planned_tasks(cases, cfg):
    tasks, permutations = [], []
    generation_started = perf_counter()
    for case in cases:
        operations = case["operation_ids"]
        check(len(operations) == len(set(operations)), "duplicate operation IDs")
        for construction_seed in cfg["release_generation"]["construction_seeds"]:
            sequence = sorted(operations)
            Random(construction_seed).shuffle(sequence)
            permutations.append({"instance": case["instance"], "construction_seed": construction_seed,
                                 "sequence": sequence})
            for size in cfg["release_generation"]["prefix_sizes"]:
                check(type(size) is int and 1 <= size <= len(sequence), "invalid nonempty prefix size")
                for solver_seed in cfg["repair"]["solver_seeds"]:
                    tasks.append({"instance": case["instance"], "construction_seed": construction_seed,
                                  "k": size, "solver_seed": solver_seed,
                                  "released": sorted(sequence[:size])})
    Random(cfg["batch"]["order_seed"]).shuffle(tasks)
    for number, task in enumerate(tasks, 1):
        task["id"] = f"task-{number:04d}"
    return tasks, permutations, perf_counter() - generation_started


def perform_request(spec, repair_function=None):
    from fjsp import parse_instance, validate_schedule
    if repair_function is None:
        from repair import repair_schedule
        repair_function = repair_schedule
    task, case, cfg = spec["task"], spec["case"], spec["repair"]
    started = perf_counter()
    instance = parse_instance(pinned(case["input"], case["sha256"]))
    initial = pinned(case["anchor"], case["anchor_sha256"])
    before = validate_schedule(instance, initial)
    check(instance.name == task["instance"] == case["instance"], "request instance mismatch")
    check(len(instance.options) == case["operations"], "operation count mismatch")
    released = task["released"]
    check(len(released) == task["k"] == len(set(released)) and set(released) <= instance.options.keys(),
          "invalid release set")
    loaded = perf_counter()
    result = repair_function(instance, initial, released,
                             started + (cfg["request_wall_budget_ms"] - cfg["return_reserve_ms"]) / 1000,
                             workers=cfg["workers"], seed=task["solver_seed"],
                             presolve=cfg["cp_model_presolve"])
    repaired = perf_counter()
    value = None
    if result.schedule is not None:
        check(result.status in ("FEASIBLE", "OPTIMAL"), "schedule with unexpected solver status")
        value = verify_candidate(instance, initial, released, result.schedule)
    check(result.status not in ("INFEASIBLE", "MODEL_INVALID"), "validated anchor must be feasible")
    if result.lower_bound is not None:
        check(math.isfinite(result.lower_bound) and result.lower_bound <= (value or before) + 1e-6,
              "invalid lower bound")
    if result.status == "OPTIMAL":
        check(value is not None and result.lower_bound is not None
              and abs(value - result.lower_bound) < 1e-6, "OPTIMAL without matching value and bound")
    verified = perf_counter()
    elapsed = verified - started
    return {"task": task, "before": before, "value": value,
            "on_time": elapsed <= cfg["request_wall_budget_ms"] / 1000,
            "request_seconds": elapsed, "input_seconds": loaded - started,
            "repair_call_seconds": repaired - loaded, "verification_seconds": verified - repaired,
            "search_seconds": result.search_seconds,
            "eligible_options": sum(len(instance.options[op]) for op in released),
            "result": asdict(result)}


def worker(request_path):
    boot = perf_counter()
    spec = read(request_path)
    response_path = Path(spec["response"])
    try:
        from experiments.resched_qualification import configure_cpu
        affinity = configure_cpu()
        # Native package imports are explicitly outside the request clock.
        import repair
        response = perform_request(spec, repair.repair_schedule)
        response.update({"affinity": affinity, "worker_seconds_before_serialization": perf_counter() - boot,
                         "request_sha256": digest(request_path)})
        save(response_path, response)
        return 0
    except Exception as error:
        save(response_path, {"task": spec["task"], "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc(), "request_sha256": digest(request_path)})
        return 1


def prepare(output):
    from experiments.resched_qualification import configure_cpu
    affinity = configure_cpu()
    cfg = read(CONFIG)
    actual = {"python": platform.python_version(), "ortools": version("ortools")}
    check(actual == cfg["resources"]["expected_environment"], "environment mismatch")
    for path, expected in cfg["source_fingerprints"].items():
        check(digest(path) == expected, f"draft source changed: {path}")
    sources = list(dict.fromkeys([str(CONFIG), *cfg["source_fingerprints"], *EXTRA_SOURCES]))
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in sources:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        source_hashes[name] = digest(target)
    (output / "inputs").mkdir()
    (output / "anchors").mkdir()
    from baseline import solve_ect
    from fjsp import parse_instance, validate_schedule
    source_cases = read(cfg["cases_config"])["cases"]
    cases = []
    for name in cfg["case_names"]:
        original = [case for case in source_cases if case["instance"] == name]
        check(len(original) == 1, f"missing or duplicate case: {name}")
        case = dict(original[0])
        raw = Path(case["input"]).read_bytes()
        check(sha256(raw).hexdigest() == case["sha256"], "input changed")
        instance = parse_instance(json.loads(raw))
        check(len(instance.options) == cfg["expected_operations"][name] == case["operations"],
              "unexpected scale")
        target = output / "inputs" / f"{name}.json"
        target.write_bytes(raw)
        case["original_input"] = case["input"]
        case["input"] = str(target.resolve())
        started = perf_counter()
        result = solve_ect(instance, started + cfg["anchor"]["preparation_budget_ms_per_case"] / 1000)
        check(result.schedule is not None, "ECT preparation failed")
        value = validate_schedule(instance, result.schedule)
        preparation_seconds = perf_counter() - started
        check(preparation_seconds <= cfg["anchor"]["preparation_budget_ms_per_case"] / 1000,
              "ECT preparation exceeded its budget")
        check(value == cfg["anchor"]["expected_cmax_from_previous_ect"][name], "ECT value changed")
        anchor = output / "anchors" / f"{name}.json"
        save(anchor, result.schedule)
        case.update({"anchor": str(anchor.resolve()), "anchor_sha256": digest(anchor),
                     "before": value, "preparation_seconds": preparation_seconds,
                     "operation_ids": sorted(instance.options)})
        cases.append(case)
    tasks, permutations, generation_seconds = planned_tasks(cases, cfg)
    check(len(tasks) == cfg["batch"]["planned_requests"], "task count changed")
    check(len(tasks) * cfg["repair"]["request_wall_budget_ms"]
          == cfg["batch"]["summed_request_budgets_ms"], "budget count changed")
    cfg["status"] = "frozen-for-measurement"
    cfg["protocol"] = "conditional-selection-cp-cost-calibration-1"
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
            "environment": actual, "platform": platform.platform(), "executable": sys.executable,
            "affinity": affinity, "cases": cases, "tasks": tasks, "permutations": permutations,
            "generation_seconds": generation_seconds, "source_sha256": source_hashes}
    save(output / "plan.json", plan)
    save(output / "qualification.json", {"passed": True, "environment": actual,
                                         "ECT": {case["instance"]: case["before"] for case in cases},
                                         "planned_requests": len(tasks), "affinity": affinity})
    return plan


def execute(output, task, case, cfg, batch_deadline):
    directory = output / "requests" / task["id"]
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    response = directory / "response.json"
    save(request, {"task": task, "case": case, "repair": cfg["repair"], "response": str(response.resolve())})
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    row = {"task": task, "request": str(request.relative_to(output)),
           "request_sha256": digest(request), "process_reaped": True}
    process = None
    started = perf_counter()
    try:
        with (directory / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "experiments.conditional_selection_cost_pilot",
                                        "--worker", str(request.resolve())],
                                       cwd=output / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                limit = min(cfg["batch"]["per_request_watchdog_seconds"], batch_deadline - perf_counter())
                if limit <= 0:
                    raise subprocess.TimeoutExpired(process.args, 0)
                row["exit_code"] = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                row["watchdog"] = True
                process.terminate()
                try:
                    process.wait(timeout=cfg["batch"]["cleanup_grace_seconds"] / 2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=cfg["batch"]["cleanup_grace_seconds"] / 2)
                row["exit_code"] = process.returncode
    except KeyboardInterrupt:
        row["error"] = "INTERRUPTED"
        row["interrupted"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=cfg["batch"]["cleanup_grace_seconds"])
            except subprocess.TimeoutExpired:
                row["process_reaped"] = False
        row["process_wall_seconds"] = perf_counter() - started
    if response.exists():
        row["response"] = str(response.relative_to(output))
        row["response_sha256"] = digest(response)
        try:
            reply = read(response)
            check(reply["task"] == task, "worker task identity mismatch")
            check(reply["request_sha256"] == row["request_sha256"], "worker request fingerprint mismatch")
            if reply.get("error"):
                row["error"] = reply["error"]
        except Exception as error:
            row["error"] = f"response validation: {error}"
    elif not row.get("error"):
        row["error"] = "worker did not return a response"
    if row.get("exit_code") != 0 and not row.get("error"):
        row["error"] = f"worker exit {row.get('exit_code')}"
    return row


def observation(task, case, row, reply, cfg):
    """Never credit a late value, an error, or an absent solver response."""
    reply = reply if isinstance(reply, dict) else {}
    successful = (not row.get("error") and not row.get("watchdog") and row.get("exit_code") == 0
                  and row.get("process_reaped") is True and isinstance(reply, dict) and not reply.get("error"))
    value = reply.get("value") if successful else None
    on_time = successful and reply.get("on_time") is True
    usable = on_time and type(value) is int and reply["result"]["status"] in ("FEASIBLE", "OPTIMAL")
    delivered = value if usable else case["before"]
    return {**task, "operations": case["operations"], "before": case["before"],
            "status": (reply.get("result") or {}).get("status", "ERROR"),
            "on_time": bool(on_time), "usable": bool(usable), "delivered_value": delivered,
            "gain": case["before"] - delivered, "raw_value": value,
            "request_seconds": reply.get("request_seconds") if successful else None,
            "search_seconds": reply.get("search_seconds") if successful else None,
            "eligible_options": reply.get("eligible_options") if successful else None,
            "process_wall_seconds": row.get("process_wall_seconds"),
            "error": row.get("error"), "watchdog": bool(row.get("watchdog")),
            "near_budget": bool(successful and reply["request_seconds"] >=
                                0.9 * cfg["repair"]["request_wall_budget_ms"] / 1000)}


def aggregate(observations):
    groups = defaultdict(list)
    for row in observations:
        groups[row["operations"], row["k"]].append(row)
    result = []
    for (scale, size), rows in sorted(groups.items()):
        times = sorted(row["request_seconds"] for row in rows if row["request_seconds"] is not None)
        paired = defaultdict(list)
        for row in rows:
            paired[row["instance"], row["construction_seed"]].append(row)
        stable = sum(len(group) == 3 and all(item["usable"] and item["gain"] > 0 for item in group)
                     for group in paired.values())
        result.append({"operations": scale, "k": size, "planned": len(rows), "timed": len(times),
                       "median_ms": 1000 * median(times) if times else None,
                       "p90_ms": 1000 * times[math.ceil(0.9 * len(times)) - 1] if times else None,
                       "timely_valid": sum(row["usable"] for row in rows),
                       "improved": sum(row["gain"] > 0 for row in rows),
                       "stable_improving_groups": stable,
                       "near_budget": sum(row["near_budget"] for row in rows),
                       "late": sum(row["request_seconds"] is not None and not row["on_time"] for row in rows),
                       "errors": sum(bool(row["error"]) or row["watchdog"] for row in rows),
                       "statuses": dict(Counter(row["status"] for row in rows))})
    return result


def audit(output):
    from fjsp import parse_instance, validate_schedule
    plan = read(output / "plan.json")
    cfg = plan["config"]
    issues = []
    for name, expected in plan["source_sha256"].items():
        check(digest(output / "source" / name) == expected, f"frozen source changed: {name}")
        check(digest(name) == expected, f"live source changed: {name}")
    cases = {case["instance"]: case for case in plan["cases"]}
    instances, anchors = {}, {}
    for name, case in cases.items():
        instances[name] = parse_instance(pinned(case["input"], case["sha256"]))
        check(digest(case["original_input"]) == case["sha256"], "original input changed")
        anchors[name] = pinned(case["anchor"], case["anchor_sha256"])
        check(validate_schedule(instances[name], anchors[name]) == case["before"], "anchor changed")
    expected_tasks, expected_permutations, _ = planned_tasks(plan["cases"], cfg)
    check(expected_tasks == plan["tasks"] and expected_permutations == plan["permutations"],
          "saved permutations or task order changed")
    check(len(expected_tasks) == cfg["batch"]["planned_requests"], "plan count mismatch")
    rows = [json.loads(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    check([row["task"] for row in rows] == expected_tasks, "missing, duplicate or reordered result row")
    observations, checked_schedules, file_hashes = [], 0, {}
    for task, row in zip(expected_tasks, rows):
        case = cases[task["instance"]]
        reply = None
        try:
            check(row.get("process_reaped") is True, "child not reaped")
            if "request" in row:
                path = output / row["request"]
                check(digest(path) == row["request_sha256"], "request changed")
                request = read(path)
                check(request["task"] == task and request["case"] == case
                      and request["repair"] == cfg["repair"], "request differs from plan")
                file_hashes[row["request"]] = row["request_sha256"]
            if "response" in row:
                path = output / row["response"]
                check(digest(path) == row["response_sha256"], "response changed")
                reply = read(path)
                check(reply["task"] == task and reply["request_sha256"] == row["request_sha256"],
                      "response identity mismatch")
                file_hashes[row["response"]] = row["response_sha256"]
                if "result" in reply:
                    check(reply["before"] == case["before"], "response anchor mismatch")
                    elapsed = reply["request_seconds"]
                    check(elapsed >= 0 and math.isfinite(elapsed), "invalid request time")
                    check(reply["on_time"] == (elapsed <= cfg["repair"]["request_wall_budget_ms"] / 1000),
                          "timeliness flag mismatch")
                    schedule = reply["result"]["schedule"]
                    if schedule is not None:
                        check(reply["result"]["status"] in ("FEASIBLE", "OPTIMAL"),
                              "schedule with unexpected status")
                        value = verify_candidate(instances[task["instance"]], anchors[task["instance"]],
                                                 task["released"], schedule)
                        check(value == reply["value"], "value differs from independent verification")
                        checked_schedules += 1
                    else:
                        check(reply["value"] is None, "fabricated objective without schedule")
                    check(reply["eligible_options"] == sum(len(instances[task["instance"]].options[op])
                                                           for op in task["released"]), "option count mismatch")
                    check(reply["result"]["bound_scope"] == "neighborhood", "unexpected proof scope")
                    check(reply["result"]["status"] not in ("INFEASIBLE", "MODEL_INVALID"), "invalid model status")
                    if reply["result"]["status"] == "OPTIMAL":
                        check(reply["value"] is not None and reply["result"]["lower_bound"] is not None
                              and abs(reply["value"] - reply["result"]["lower_bound"]) < 1e-6,
                              "invalid OPTIMAL certificate")
            if row.get("error") or row.get("watchdog"):
                issues.append({"task": task["id"], "kind": "execution", "detail": row.get("error", "watchdog")})
        except Exception as error:
            issues.append({"task": task["id"], "kind": "audit", "detail": str(error)})
            row = {**row, "error": f"audit: {error}"}
        observations.append(observation(task, case, row, reply, cfg))
    optimal_groups = defaultdict(list)
    for item in observations:
        if item["usable"] and item["status"] == "OPTIMAL":
            optimal_groups[item["instance"], item["construction_seed"], item["k"]].append(item["raw_value"])
    for key, values in optimal_groups.items():
        if len(set(values)) != 1:
            issues.append({"group": key, "kind": "audit", "detail": "conflicting OPTIMAL values"})
    for (name, seed, size), values in optimal_groups.items():
        for (other_name, other_seed, other_size), other in optimal_groups.items():
            if name == other_name and seed == other_seed and size < other_size and min(values) < max(other):
                issues.append({"group": [name, seed, size, other_size], "kind": "audit",
                               "detail": "OPTIMAL nested neighborhoods violate inclusion"})
    summary = {"protocol": cfg["protocol"], "planned": len(expected_tasks), "finished_rows": len(rows),
               "validated_schedules": checked_schedules, "timely_valid": sum(row["usable"] for row in observations),
               "improved_requests": sum(row["gain"] > 0 for row in observations),
               "late": sum(row["request_seconds"] is not None and not row["on_time"] for row in observations),
               "issues": issues, "by_scale_and_k": aggregate(observations), "observations": observations}
    summary_path = output / "summary.json"
    if summary_path.exists():
        check(read(summary_path) == summary, "re-audit summary differs from saved summary")
    else:
        save(summary_path, summary)
    report_path = output / "REPORT.md"
    if not report_path.exists():
        lines = ["# Conditional selection: fixed-budget CP cost diagnostic", "",
                 "CPU-only, exposed ECT anchors, random nested prefixes; no training or STOP labels.", "",
                 f"Planned: {len(rows)}; independently validated schedules: {checked_schedules}; "
                 f"timely valid: {summary['timely_valid']}; improved requests: {summary['improved_requests']}; "
                 f"late: {summary['late']}; execution/audit issues: {len(issues)}.", "",
                 "| Operations | K | N | Median ms | P90 ms | Timely valid | Improved | Near 1s | Late |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary["by_scale_and_k"]:
            med = f"{row['median_ms']:.3f}" if row["median_ms"] is not None else "missing"
            p90 = f"{row['p90_ms']:.3f}" if row["p90_ms"] is not None else "missing"
            lines.append(f"| {row['operations']} | {row['k']} | {row['planned']} | {med} | {p90} | "
                         f"{row['timely_valid']} | {row['improved']} | {row['near_budget']} | {row['late']} |")
        lines += ["", "Repeated prefixes/seeds are dependent; this is not a paired full online quality comparison.",
                  "Unknown or late outcomes are not proofs of no improvement. Observed P90 is descriptive.",
                  "Model-selected costs, graph/decoder time, server performance, and an optimal release cap are unmeasured.", ""]
        with report_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(lines))
    audit_result = {"clean": not issues, "issues": issues, "checked_schedules": checked_schedules,
                    "plan_sha256": digest(output / "plan.json"), "runs_sha256": digest(output / "runs.jsonl"),
                    "summary_sha256": digest(summary_path), "request_response_sha256": file_hashes}
    target = output / "audit.json"
    if target.exists():
        target = output / f"re-audit-{uuid4().hex}.json"
    save(target, audit_result)
    print(json.dumps({"audit_clean": not issues, "planned": len(rows), "validated": checked_schedules,
                      "timely_valid": summary["timely_valid"], "improved": summary["improved_requests"],
                      "late": summary["late"], "issues": len(issues), "summary": str(summary_path)}), flush=True)
    return not issues


def run(output):
    preparation_started = perf_counter()
    plan = prepare(output)
    check("torch" not in sys.modules, "unexpected torch import")
    print(json.dumps({"prepared": str(output), "requests": len(plan["tasks"]),
                      "ECT": {case["instance"]: case["before"] for case in plan["cases"]}}), flush=True)
    cfg = plan["config"]
    cases = {case["instance"]: case for case in plan["cases"]}
    started = perf_counter()
    deadline = started + cfg["batch"]["batch_wall_limit_seconds"]
    errors = 0
    interrupted = False
    with (output / "runs.jsonl").open("x", encoding="utf-8", buffering=1) as file:
        for number, task in enumerate(plan["tasks"], 1):
            if interrupted or perf_counter() >= deadline:
                row = {"task": task, "error": "NOT_STARTED: batch interrupted or wall limit reached",
                       "process_reaped": True}
            else:
                row = execute(output, task, cases[task["instance"]], cfg, deadline)
                interrupted = bool(row.get("interrupted"))
            errors += bool(row.get("error"))
            file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            if number % 12 == 0 or number == len(plan["tasks"]):
                print(json.dumps({"finished": number, "planned": len(plan["tasks"]), "errors": errors,
                                  "wall_seconds": round(perf_counter() - started, 2)}), flush=True)
    save(output / "runtime.json", {"main_wall_seconds": perf_counter() - started,
                                   "preparation_seconds": started - preparation_started})
    return audit(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    check(args.output is not None, "an output directory is required")
    return 0 if (audit(args.output.resolve()) if args.audit else run(args.output.resolve())) else 1


if __name__ == "__main__":
    sys.exit(main())
