"""Bounded reusable prefix collection with warm CPU worker and soft local target."""

import argparse
from importlib.metadata import version
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
from queue import Queue, Empty
import shutil
import subprocess
import sys
from threading import Thread
from time import perf_counter
import traceback

from experiments import conditional_selection_cost_pilot as cost
from experiments.cp_streaming_delivery_probe import atomic_save, read_handshake, observed_repair


TASK_PLAN = Path("outputs/conditional-selection-first-chunk-plan-20261110")
SOURCES = ["experiments/conditional_selection_collect.py", "experiments/cp_streaming_delivery_probe.py",
           "experiments/conditional_selection_cost_pilot.py", "experiments/resched_qualification.py",
           "fjsp.py", "repair.py", "tests/test_conditional_selection_collect.py"]


def emit(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def worker(plan_path):
    from experiments.resched_qualification import configure_cpu
    from fjsp import parse_instance, validate_schedule
    import repair
    plan_path = Path(plan_path)
    root = plan_path.parent
    plan = cost.read(plan_path)
    case = plan["case"]
    affinity = configure_cpu()
    instance = parse_instance(cost.pinned(case["input"], case["sha256"]))
    initial = cost.pinned(case["anchor"], case["anchor_sha256"])
    cost.check(validate_schedule(instance, initial) == case["before"], "worker anchor changed")
    tasks = {task["id"]: task for task in plan["tasks"]}
    emit({"event": "ready", "pid": os.getpid(), "affinity": affinity,
          "python": platform.python_version(), "ortools": version("ortools")})
    for line in sys.stdin:
        command = json.loads(line)
        if command["event"] == "shutdown":
            emit({"event": "bye"})
            return 0
        task = tasks[command["task_id"]]
        directory = root / "requests" / task["id"]
        sequence = 0
        try:
            def publish(schedule, kind, details):
                nonlocal sequence
                if schedule["makespan"] >= initial["makespan"]:
                    return False
                sequence += 1
                path = directory / f"candidate-{sequence:06d}.json"
                scope = "global" if len(task["released"]) == len(instance.options) else "neighborhood"
                atomic_save(path, {"task_id": task["id"], "sequence": sequence, "kind": kind,
                                   "schedule": schedule, **details, "bound_scope": scope})
                emit({"event": "candidate", "task_id": task["id"], "path": path.name, "sequence": sequence})
                return True

            report = observed_repair(instance, initial, task["released"], command["solver_deadline"],
                                     seed=task["solver_seed"], stream=True, publish=publish)
            report.update(task=task, completed_at=perf_counter(), frames=sequence)
            atomic_save(directory / "worker-report.json", report)
            emit({"event": "done", "task_id": task["id"], "path": "worker-report.json"})
        except Exception as error:
            atomic_save(directory / "worker-error.json", {"error": f"{type(error).__name__}: {error}",
                                                          "traceback": traceback.format_exc()})
            emit({"event": "error", "task_id": task["id"], "path": "worker-error.json"})
            return 1
    return 0


def reader(pipe, mailbox):
    try:
        for line in pipe:
            try:
                mailbox.put(json.loads(line))
            except Exception:
                mailbox.put({"event": "protocol_error", "line": line[:500]})
    finally:
        mailbox.put({"event": "eof"})


def receive(mailbox, deadline):
    try:
        return mailbox.get(timeout=max(.0001, deadline - perf_counter()))
    except Empty:
        raise TimeoutError("worker watchdog or batch deadline")


def measure(root, plan, task, process, mailbox, batch_stop):
    case, cfg = plan["case"], plan["config"]
    directory = root / "requests" / task["id"]
    directory.mkdir(parents=True, exist_ok=False)
    # Reuse parsed parent data across calls; setup cost is charged to batch wall time.
    instance, initial = plan["_instance"], plan["_initial"]
    started = perf_counter()
    solver_deadline = started + cfg["repair_target_seconds"]
    watchdog = min(batch_stop, started + cfg["proposed_abnormal_request_watchdog_seconds"])
    process.stdin.write(json.dumps({"event": "repair", "task_id": task["id"],
                                    "started": started, "solver_deadline": solver_deadline}) + "\n")
    process.stdin.flush()
    best, accepted, last_sequence = initial, [], 0
    while True:
        event = receive(mailbox, watchdog)
        cost.check(event.get("task_id") == task["id"], f"unexpected worker event: {event}")
        cost.check(event["event"] in ["candidate", "done"], f"worker error: {event}")
        if event["event"] == "done":
            report, _ = read_handshake(directory / event["path"], watchdog)
            cost.check(report["task"] == task, "report identity mismatch")
            native = report["result"]
            if native["schedule"] is not None:
                native_value = cost.verify_candidate(instance, initial, task["released"], native["schedule"])
                cost.check(native["status"] in ["FEASIBLE", "OPTIMAL"], "unexpected native schedule status")
                cost.check(native_value >= best["makespan"], "native best improvement was not published")
                if native["lower_bound"] is not None:
                    cost.check(math.isfinite(native["lower_bound"]) and native["lower_bound"] <= best["makespan"] + 1e-6,
                               "native bound invalid")
                if native["status"] == "OPTIMAL":
                    cost.check(abs(native_value - native["lower_bound"]) < 1e-6, "invalid optimality claim")
            cost.check(native["status"] not in ["INFEASIBLE", "MODEL_INVALID"], "anchor unexpectedly infeasible")
            break
        cost.check(event["sequence"] == last_sequence + 1
                   and event["path"] == f"candidate-{event['sequence']:06d}.json", "frame sequence mismatch")
        frame, _ = read_handshake(directory / event["path"], watchdog)
        received = perf_counter()
        cost.check(frame["task_id"] == task["id"] and frame["sequence"] == event["sequence"], "foreign frame")
        value = cost.verify_candidate(instance, initial, task["released"], frame["schedule"])
        expected_scope = "global" if len(task["released"]) == case["operations"] else "neighborhood"
        cost.check(frame["bound_scope"] == expected_scope and math.isfinite(frame["lower_bound"])
                   and frame["lower_bound"] <= value + 1e-6, "callback bound/scope mismatch")
        last_sequence = event["sequence"]
        if value < best["makespan"]:
            verified = perf_counter()
            name = f"accepted-{last_sequence:06d}.json"
            atomic_save(directory / name, frame["schedule"])
            saved = perf_counter()
            accepted.append({"frame": event["path"], "path": name, "value": value,
                             "received_at": received, "verified_at": verified, "saved_at": saved})
            best = frame["schedule"]
    cost.verify_candidate(instance, initial, task["released"], best)
    atomic_save(directory / "best-saved.json", best)
    finished = perf_counter()
    return {"task": task, "state": "COMPLETED", "before": initial["makespan"], "value": best["makespan"],
            "gain": initial["makespan"] - best["makespan"], "started": started, "finished": finished,
            "request_seconds": finished - started,
            "over_local_target_seconds": max(0, finished - solver_deadline),
            "solver_deadline": solver_deadline, "native_status": native["status"],
            "bound_scope": native["bound_scope"], "accepted": accepted,
            "callback_seconds": report["callback_seconds"],
            "artifacts": {path.name: cost.digest(path) for path in sorted(directory.glob("*.json"))}}


def prepare(output, prepared_plan=TASK_PLAN):
    from experiments.resched_qualification import configure_cpu
    from fjsp import parse_instance, validate_schedule
    prepared_plan = Path(prepared_plan)
    config = cost.read(prepared_plan / "source" / "conditional_selection_first_chunk.plan.json")
    review = cost.read(prepared_plan / "review.json")
    for name, expected in cost.read(prepared_plan / "artifact-hashes.json").items():
        cost.check(cost.digest(prepared_plan / name) == expected, "prepared artifact changed")
    cost.check(cost.digest(prepared_plan / "tasks.jsonl") == review["tasks_sha256"], "task list changed")
    tasks = [json.loads(line) for line in (prepared_plan / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    for name in ["input.json", "anchor.json", "permutation.json", "tasks.jsonl"]:
        shutil.copyfile(prepared_plan / name, output / name)
    source_hashes = {}
    for name in SOURCES:
        path = output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
        source_hashes[name] = cost.digest(path)
    instance = parse_instance(cost.read(output / "input.json"))
    initial = cost.read(output / "anchor.json")
    cost.check(validate_schedule(instance, initial) == review["initial_Cmax"], "initial changed")
    case = {"instance": instance.name, "operations": len(instance.options), "before": initial["makespan"],
            "input": str((output / "input.json").resolve()), "sha256": cost.digest(output / "input.json"),
            "anchor": str((output / "anchor.json").resolve()), "anchor_sha256": cost.digest(output / "anchor.json")}
    environment = {"python": platform.python_version(), "ortools": version("ortools")}
    cost.check(environment == {"python": "3.14.10", "ortools": "9.15.6755"}, "unqualified local environment")
    plan = {"config": config, "case": case, "tasks": tasks, "source_sha256": source_hashes,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_status": "frozen-for-first-unit-collection; collector-qualified-by-tests",
            "environment": environment, "affinity": configure_cpu(), "target": "local Windows CPU; not server timing",
            "prepared_plan": str(prepared_plan.resolve()), "prepared_task_sha256": review["tasks_sha256"],
            "timing": "One warm worker; shared input/anchor setup and imports count in batch wall, separate from per-repair clock. Per-repair clock includes IPC, CP build/search/callback/serialization, parent validation/persistence and final best file. No graph/conditional-decoder time, no full online60s claim.",
            "training": False}
    cost.save(output / "plan.json", plan)
    return plan


def collect(output, plan, worker_command=None):
    from fjsp import parse_instance
    root = Path(output)
    cfg = plan["config"]
    started = perf_counter()
    end = started + cfg["proposed_batch_wall_limit_seconds"]
    stop = end - cfg["proposed_cleanup_grace_seconds"]
    plan["_instance"] = parse_instance(cost.read(plan["case"]["input"]))
    plan["_initial"] = cost.read(plan["case"]["anchor"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({key: "1" for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    command = worker_command or [sys.executable, "-m", "experiments.conditional_selection_collect", "--worker", str(root / "plan.json")]
    command = list(command)
    if os.name == "nt" and Path(command[0]).resolve() == Path(sys.executable).resolve():
        env["__PYVENV_LAUNCHER__"] = sys.executable
        command[0] = sys._base_executable
    mailbox, stopped, errors, completed = Queue(), None, 0, 0
    with (root / "worker.stderr.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root / "source", env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        thread = Thread(target=reader, args=(process.stdout, mailbox), daemon=True)
        thread.start()
        try:
            ready = receive(mailbox, min(stop, perf_counter() + 10))
            cost.check(ready["event"] == "ready" and ready["pid"] == process.pid, "worker not directly owned")
            cost.check({key: ready[key] for key in ["python", "ortools"]} == plan["environment"], "worker env mismatch")
            cost.save(root / "ready.json", ready)
            with (root / "runs.jsonl").open("x", encoding="utf-8", buffering=1) as file:
                for index, task in enumerate(plan["tasks"], 1):
                    if perf_counter() >= stop:
                        stopped = stopped or "batch wall limit"
                    if stopped:
                        row = {"task": task, "state": "NOT_STARTED", "reason": stopped}
                    else:
                        try:
                            row = measure(root, plan, task, process, mailbox, stop)
                            completed += 1
                        except (Exception, KeyboardInterrupt) as error:
                            stopped = f"{type(error).__name__}: {error}"
                            errors += 1
                            row = {"task": task, "state": "ERROR", "error": stopped}
                    file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    if index % 25 == 0 or index == len(plan["tasks"]):
                        emit({"finished_slots": index, "planned": len(plan["tasks"]), "completed": completed,
                              "errors": errors, "wall_seconds": round(perf_counter() - started, 2)})
                    if index % 25 == 0 and not stopped:
                        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
                        if used >= cfg["proposed_output_limit_decimal_GB"] * 10**9 - 64 * 10**6:
                            stopped = "storage warning threshold"
        except (Exception, KeyboardInterrupt) as error:
            stopped = f"{type(error).__name__}: {error}"
            errors += 1
            if not (root / "runs.jsonl").exists():
                with (root / "runs.jsonl").open("x", encoding="utf-8") as file:
                    for task in plan["tasks"]:
                        file.write(json.dumps({"task": task, "state": "NOT_STARTED", "reason": stopped}) + "\n")
        finally:
            if process.poll() is None and not stopped:
                try:
                    process.stdin.write('{"event":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=min(5, max(.01, end - perf_counter())))
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
            elif process.poll() is None:
                process.kill()
            process.wait(timeout=cfg["proposed_cleanup_grace_seconds"])
            thread.join(timeout=1)
            process.stdin.close()
            process.stdout.close()
    cost.save(root / "runtime.json", {"wall_seconds": perf_counter() - started, "worker_pid": process.pid,
                                       "process_reaped": process.poll() is not None, "exit_code": process.returncode,
                                       "completed": completed, "errors": errors, "stopped": stopped})


def audit(root):
    from collections import Counter
    from fjsp import parse_instance
    plan = cost.read(root / "plan.json")
    if (root / "audit.json").exists():
        frozen = cost.read(root / "audit.json")
        cost.check(cost.digest(root / "runs.jsonl") == frozen["runs_sha256"]
                   and cost.digest(root / "summary.json") == frozen["summary_sha256"]
                   and cost.digest(root / "plan.json") == frozen["plan_sha256"], "frozen run evidence changed")
    for name, expected in plan["source_sha256"].items():
        cost.check(cost.digest(name) == cost.digest(root / "source" / name) == expected, "source changed")
    case = plan["case"]
    instance = parse_instance(cost.pinned(case["input"], case["sha256"]))
    initial = cost.pinned(case["anchor"], case["anchor_sha256"])
    rows = [json.loads(line) for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    cost.check([row["task"] for row in rows] == plan["tasks"], "missing or reassigned task slots")
    if "prepared_task_sha256" in plan:
        cost.check(cost.digest(root / "tasks.jsonl") == plan["prepared_task_sha256"], "prepared tasks changed")
        expected_tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        cost.check(plan["tasks"] == expected_tasks, "plan does not match prepared tasks")
        permutation = cost.read(root / "permutation.json")["operations"]
        cost.check(len(permutation) == len(instance.options) and set(permutation) == instance.options.keys(),
                   "complete selection order changed")
        for task in expected_tasks:
            cost.check(task["released"] == sorted(permutation[:task["k"]])
                       and task["solver_seed"] in plan["config"]["solver_seeds"], "task no longer matches prefix")
    groups, issues, validated = {}, [], 0
    for row in rows:
        task = row["task"]
        if row["state"] != "COMPLETED":
            issues.append({"task": task["id"], "state": row["state"]})
            continue
        try:
            directory = root / "requests" / task["id"]
            for name, expected in row["artifacts"].items():
                cost.check(cost.digest(directory / name) == expected, "request artifact changed")
            value = cost.verify_candidate(instance, initial, task["released"], cost.read(directory / "best-saved.json"))
            cost.check(value == row["value"] and row["gain"] == case["before"] - value, "objective changed")
            cost.check(abs(row["request_seconds"] - row["finished"] + row["started"]) < 1e-7, "clock changed")
            cost.check(abs(row["over_local_target_seconds"] - max(0, row["request_seconds"] - plan["config"]["repair_target_seconds"])) < 1e-7,
                       "local overrun changed")
            scope = "global" if len(task["released"]) == case["operations"] else "neighborhood"
            cost.check(row["bound_scope"] == scope, "scope changed")
            for path in directory.glob("candidate-*.json"):
                frame = cost.read(path)
                candidate_value = cost.verify_candidate(instance, initial, task["released"], frame["schedule"])
                cost.check(frame["task_id"] == task["id"] and frame["bound_scope"] == scope,
                           "candidate identity/scope changed")
                cost.check(math.isfinite(frame["lower_bound"]) and frame["lower_bound"] <= candidate_value + 1e-6,
                           "candidate bound changed")
            report = cost.read(directory / "worker-report.json")
            cost.check(report["task"] == task and report["result"]["status"] == row["native_status"], "report changed")
            previous = case["before"]
            for receipt in row["accepted"]:
                frame = cost.read(directory / receipt["frame"])
                schedule = cost.read(directory / receipt["path"])
                cost.check(frame["schedule"] == schedule and frame["task_id"] == task["id"], "accepted frame differs")
                candidate = cost.verify_candidate(instance, initial, task["released"], schedule)
                cost.check(candidate == receipt["value"] < previous and frame["bound_scope"] == scope, "acceptance differs")
                cost.check(row["started"] <= receipt["received_at"] <= receipt["verified_at"] <= receipt["saved_at"] <= row["finished"],
                           "acceptance clock differs")
                previous = candidate
            cost.check(previous == value, "best candidate not delivered")
            validated += 1
            groups.setdefault(task["k"], []).append(row)
        except Exception as error:
            issues.append({"task": task["id"], "error": str(error)})
    runtime = cost.read(root / "runtime.json")
    cost.check(runtime["process_reaped"], "worker remains running")
    if plan.get("affinity") and (root / "ready.json").exists():
        ready = cost.read(root / "ready.json")
        cost.check(ready["pid"] == runtime["worker_pid"]
                   and ready["affinity"]["logical_processor"] == plan["affinity"]["logical_processor"],
                   "worker PID or CPU differs")
    expected_seeds = set(plan["config"]["solver_seeds"])
    consistent = [k for k, repeated in groups.items() if len(repeated) == len(expected_seeds)
                  and {row["task"]["solver_seed"] for row in repeated} == expected_seeds
                  and all(row["gain"] > 0 for row in repeated)]
    times = sorted(row["request_seconds"] for row in rows if row["state"] == "COMPLETED")
    summary = {"planned": len(rows), "validated": validated, "issues": issues, "audit_clean": not issues,
               "improving_requests": sum(row.get("gain", 0) > 0 for row in rows),
               "consistent_improving_checkpoints": consistent,
               "best_saved_Cmax": min((row["value"] for row in rows if row["state"] == "COMPLETED"), default=case["before"]),
               "native_statuses": dict(Counter(row.get("native_status", row["state"]) for row in rows)),
               "over_local_target": sum(row.get("over_local_target_seconds", 0) > 0 for row in rows),
               "median_request_seconds": cost.median(times) if times else None,
               "max_request_seconds": max(times) if times else None, "runtime": runtime,
               "training": False, "STOP_labels_generated": False, "server_timing": False}
    if (root / "summary.json").exists():
        cost.check(cost.read(root / "summary.json") == summary, "re-audit changed")
        name = f"re-audit-{len(list(root.glob('re-audit-*.json'))):03d}.json"
    else:
        cost.save(root / "summary.json", summary)
        name = "audit.json"
    cost.save(root / name, {"clean": not issues, "issues": issues, "plan_sha256": cost.digest(root / "plan.json"),
                            "runs_sha256": cost.digest(root / "runs.jsonl"),
                            "summary_sha256": cost.digest(root / "summary.json")})
    emit({key: summary[key] for key in ["planned", "validated", "audit_clean", "improving_requests",
                                      "best_saved_Cmax", "over_local_target", "max_request_seconds"]})
    return not issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.output is None:
        parser.error("output required")
    root = args.output.resolve()
    if not args.audit:
        plan = prepare(root)
        emit({"prepared": str(root), "requests": len(plan["tasks"]), "training": False})
        collect(root, plan)
    return 0 if audit(root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
