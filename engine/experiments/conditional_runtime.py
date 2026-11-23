"""One hard global deadline with soft local CP allocations and saved incumbents."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from random import Random
import shutil
import sys
from time import perf_counter
import traceback

from delivery_guard import CandidatePublisher, atomic_json, run_guarded
from experiments import conditional_selection_cost_pilot as cost


CONFIG = Path("experiments/conditional_runtime.json")
SOURCES = [str(CONFIG), "experiments/conditional_runtime.py", "tests/test_conditional_runtime.py",
           "delivery_guard.py", "baseline.py", "fjsp.py", "repair.py", "retained_bounds.py",
           "experiments/conditional_selection_cost_pilot.py", "experiments/cp_streaming_delivery_probe.py",
           "experiments/resched_qualification.py"]


def execute_worker(request, publisher, repair_observer=None):
    from baseline import load_pinned_json
    from fjsp import parse_instance, validate_schedule
    from retained_bounds import retained_path_bound
    if repair_observer is None:
        from experiments.cp_streaming_delivery_probe import observed_repair
        repair_observer = observed_repair
    root = Path(request["output"])
    cfg = request["spec"]["config"]
    instance = parse_instance(load_pinned_json(Path(request["input_path"]), request["case"]["sha256"]))
    current = request["initial"]
    cost.check(validate_schedule(instance, current) == current["makespan"], "initial invalid")
    started, deadline = request["started"], request["worker_deadline"]
    work_until = deadline - cfg["worker_finish_reserve_ms"] / 1000
    operations, rng = sorted(instance.options), Random(request["spec"]["seed"])
    attempts, solved, locally_late, updates = 0, 0, 0, 0
    with (root / "calls.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while perf_counter() + cfg["minimum_remaining_ms"] / 1000 < work_until:
            if cfg.get("max_attempts") is not None and attempts >= cfg["max_attempts"]:
                break
            attempts += 1
            anchor = current
            released = sorted(rng.sample(operations, min(cfg["random_size"], len(operations))))
            lower = retained_path_bound(instance, anchor, released).lower_bound
            base = {"call": attempts, "anchor_Cmax": anchor["makespan"], "released": released,
                    "retained_lower_bound": lower, "prepared_ms": (perf_counter() - started) * 1000}
            if lower >= anchor["makespan"]:
                log.write(json.dumps({**base, "event": "BOUND_REJECTED"}) + "\n")
                continue
            directory = root / "calls" / f"call-{attempts:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "anchor.json", anchor)
            call_started = perf_counter()
            local_target = min(work_until, call_started + cfg["repair_target_ms"] / 1000)
            if local_target - call_started < cfg["minimum_remaining_ms"] / 1000:
                break
            solved += 1
            sequence = 0
            log.write(json.dumps({**base, "event": "CALL_STARTED", "started": call_started,
                                  "local_target": local_target}) + "\n")

            def publish(schedule, kind, details):
                nonlocal current, updates, sequence
                value = cost.verify_candidate(instance, anchor, released, schedule)
                if value >= current["makespan"]:
                    return False
                sequence += 1
                scope = "global" if len(released) == len(instance.options) else "neighborhood"
                atomic_json(directory / f"candidate-{sequence:06d}.json",
                            {"call": attempts, "kind": kind, "released": released, "schedule": schedule,
                             **details, "bound_scope": scope})
                # The global cutoff applies even when a useful local result arrives late.
                if perf_counter() >= deadline:
                    return False
                source = f"random64:{attempts}:{sequence}"
                publisher(schedule, (perf_counter() - started) * 1000, source)
                current = schedule
                updates += 1
                return True

            report = repair_observer(instance, anchor, released, local_target, seed=request["spec"]["seed"],
                                     stream=True, publish=publish)
            result = report["result"]
            if result["schedule"] is not None:
                publish(result["schedule"], "returned", {"lower_bound": result["lower_bound"], "status": result["status"]})
            finished = perf_counter()
            # No local finish<=target gate: real elapsed time consumes the global budget.
            locally_late += finished > local_target
            atomic_json(directory / "report.json", report)
            log.write(json.dumps({**base, "event": "CALL_COMPLETED", "started": call_started,
                                  "finished": finished, "local_target": local_target,
                                  "local_target_exceeded": finished > local_target,
                                  "native_status": result["status"], "current_Cmax": current["makespan"]}) + "\n")
    return {"attempts": attempts, "repair_calls": solved, "local_target_exceeded": locally_late,
            "worker_updates": updates, "schedule": current, "torch_loaded": "torch" in sys.modules,
            "worker_finished_ms": (perf_counter() - started) * 1000}


def worker(request_path):
    from experiments.resched_qualification import configure_cpu
    request = cost.read(request_path)
    root = Path(request["output"])
    try:
        affinity = configure_cpu()
        atomic_json(root / "worker_ready.json", {"pid": os.getpid(), "affinity": affinity,
                                                  "python": platform.python_version(), "ortools": version("ortools")})
        publisher = CandidatePublisher(root / "worker", request["started"])
        result = execute_worker(request, publisher)
        atomic_json(root / "worker_result.json", result)
        return 0
    except Exception as error:
        atomic_json(root / "worker_failure.json", {"error": f"{type(error).__name__}: {error}",
                                                    "traceback": traceback.format_exc()})
        return 1


@contextmanager
def child_environment(source_root=None):
    previous_launcher = os.environ.get("__PYVENV_LAUNCHER__")
    previous_cwd = Path.cwd()
    command = [sys.executable, "-m", "experiments.conditional_runtime", "--worker"]
    if os.name == "nt":
        os.environ["__PYVENV_LAUNCHER__"] = sys.executable
        command[0] = sys._base_executable
    try:
        if source_root is not None:
            os.chdir(source_root)
        yield command
    finally:
        os.chdir(previous_cwd)
        if previous_launcher is None:
            os.environ.pop("__PYVENV_LAUNCHER__", None)
        else:
            os.environ["__PYVENV_LAUNCHER__"] = previous_launcher


def run_runtime(input_path, case, output, cfg, *, initial=None, source_root=None):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    with child_environment(source_root) as command:
        result = run_guarded(input_path, case, output, {"seed": cfg["seed"], "config": cfg},
                             total_ms=cfg["total_ms"], initial_schedule=initial,
                             worker_command=command, **cfg["guard"])
    observed = (perf_counter() - result["started_perf_counter"]) * 1000
    result["caller_observed_ms"] = observed
    result["delivered_on_time"] &= observed <= cfg["total_ms"]
    result["credited_makespan"] = result["schedule"]["makespan"] if result["delivered_on_time"] else None
    return result


def prepare(root):
    from experiments.resched_qualification import configure_cpu
    cfg = cost.read(CONFIG)
    cost.check(cost.digest(cfg["case_config"]) == cfg["case_config_sha256"], "case source changed")
    case = next(item for item in cost.read(cfg["case_config"])["cases"] if item["instance"] == cfg["instance"])
    cost.check(cost.digest(case["input"]) == case["sha256"], "input changed")
    environment = {"python": platform.python_version(), "ortools": version("ortools")}
    cost.check(environment == {"python": "3.14.10", "ortools": "9.15.6755"}, "local env mismatch")
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCES:
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
        hashes[name] = cost.digest(path)
    shutil.copyfile(case["input"], root / "input.json")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "case": case,
            "source_sha256": hashes, "environment": environment, "affinity": configure_cpu(),
            "training": False, "scope": cfg["scope"]}
    cost.save(root / "plan.json", plan)
    return plan


def audit(root):
    from fjsp import parse_instance, validate_schedule
    plan, result = cost.read(root / "plan.json"), cost.read(root / "run/result.json")
    for name, expected in plan["source_sha256"].items():
        cost.check(cost.digest(name) == cost.digest(root / "source" / name) == expected, "source changed")
    instance = parse_instance(cost.pinned(root / "input.json", plan["case"]["sha256"]))
    initial_value = validate_schedule(instance, result["initial"])
    previous, validated = initial_value + 1, 0
    for receipt in result["receipts"]:
        path = root / "run" / receipt["file"]
        cost.check(cost.digest(path) == receipt["sha256"], "receipt changed")
        schedule = cost.read(path)
        value = validate_schedule(instance, schedule)
        cost.check(value == receipt["makespan"], "receipt objective changed")
        if receipt["eligible"]:
            cost.check(receipt["delivery_written_ms"] <= plan["config"]["total_ms"] and value < previous,
                       "ineligible or nonimproving receipt")
            if receipt["source"].startswith("random64:"):
                _, call_id, sequence = receipt["source"].split(":")
                directory = root / "run/calls" / f"call-{int(call_id):06d}"
                frame = cost.read(directory / f"candidate-{int(sequence):06d}.json")
                cost.check(frame["schedule"] == schedule, "delivered candidate mapping differs")
                cost.verify_candidate(instance, cost.read(directory / "anchor.json"), frame["released"], schedule)
            previous = value
            validated += 1
    final = cost.read(root / "run/delivery.json")
    cost.check(final == result["schedule"] and validate_schedule(instance, final) == previous, "final differs from best receipt")
    ready = cost.read(root / "run/worker_ready.json")
    cost.check(ready["pid"] == result["worker"]["pid"] and not result["worker"]["still_running"], "ownership/cleanup failure")
    events = [json.loads(line) for line in (root / "run/calls.jsonl").read_text().splitlines()]
    starts = [event for event in events if event["event"] == "CALL_STARTED"]
    complete = [event for event in events if event["event"] == "CALL_COMPLETED"]
    violations = [event for event in result["events"] if event["event"] == "REJECTED"]
    summary = {"initial_Cmax": initial_value, "delivered_Cmax": previous,
               "caller_observed_ms": result["caller_observed_ms"], "total_budget_ms": plan["config"]["total_ms"],
               "delivered_on_time": result["delivered_on_time"], "validated_receipts": validated,
               "repair_calls_started": len(starts), "repair_calls_completed": len(complete),
               "interrupted_calls": len(starts) - len(complete),
               "local_target_exceeded": sum(event["local_target_exceeded"] for event in complete),
               "worker_reaped": not result["worker"]["still_running"], "publication_rejections": len(violations),
               "worker_failure": (root / "run/worker_failure.json").exists(),
               "training": False, "learned_selector": False, "lambda_calibrated": False}
    summary["runtime_passed_this_execution"] = bool(summary["delivered_on_time"] and summary["worker_reaped"]
                                                     and not violations and not summary["worker_failure"])
    if (root / "summary.json").exists():
        cost.check(cost.read(root / "summary.json") == summary, "re-audit differs")
        path = root / f"re-audit-{len(list(root.glob('re-audit-*.json'))):03d}.json"
    else:
        cost.save(root / "summary.json", summary)
        path = root / "audit.json"
    cost.save(path, {"consistent": True, "runtime_passed": summary["runtime_passed_this_execution"],
                     "summary_sha256": cost.digest(root / "summary.json"), "result_sha256": cost.digest(root / "run/result.json")})
    print(json.dumps(summary), flush=True)
    return summary["runtime_passed_this_execution"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    root = args.output.resolve()
    if not args.audit:
        plan = prepare(root)
        result = run_runtime(root / "input.json", plan["case"], root / "run", plan["config"], source_root=root / "source")
        cost.save(root / "run/result.json", result)
    return 0 if audit(root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
