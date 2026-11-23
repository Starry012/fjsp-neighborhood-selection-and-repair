"""Isolated incumbent publication and deadline delivery; no policy training."""

import argparse
from collections import Counter, defaultdict
import copy
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
import subprocess
import sys
from time import perf_counter, sleep
import traceback
from unittest.mock import patch
from uuid import uuid4

from experiments import conditional_selection_cost_pilot as cost


CONFIG = Path("experiments/cp_streaming_delivery_probe.json")
SOURCES = [str(CONFIG), "experiments/cp_streaming_delivery_probe.py",
           "tests/test_cp_streaming_delivery_probe.py"]


def atomic_save(path, value):
    """Publish complete immutable frames; charge flush and rename to the caller."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, separators=(",", ":"), allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_handshake(path, deadline, poll_seconds=.001):
    retries = 0
    while True:
        try:
            return cost.read(path), retries
        except (FileNotFoundError, PermissionError):
            if perf_counter() >= deadline:
                raise TimeoutError(f"handshake file not readable: {Path(path).name}")
            retries += 1
            sleep(poll_seconds)


def build_tasks(source_tasks, cfg):
    selected = [task for task in source_tasks if task["k"] in cfg["release_sizes"]]
    cost.check(len(selected) == cfg["expected_source_requests"], "source task count changed")
    cost.check(len({task["id"] for task in selected}) == len(selected), "duplicate source task")
    tasks = []
    for original in selected:
        for condition in cfg["conditions"]:
            task = copy.deepcopy(original)
            task.update(source_task_id=original["id"], condition=copy.deepcopy(condition))
            tasks.append(task)
    Random(cfg["order_seed"]).shuffle(tasks)
    for number, task in enumerate(tasks, 1):
        task["id"] = f"stream-{number:04d}"
    cost.check(len(tasks) == cfg["planned_requests"], "planned task count changed")
    return tasks


def observed_repair(instance, initial, released, deadline, *, seed, stream, publish):
    """Intercept solve only inside an isolated worker; shared model stays intact."""
    from ortools.sat.python import cp_model
    from repair import repair_schedule
    original_solve = cp_model.CpSolver.solve
    initial_rows = {row["operation"]: row for row in initial["assignments"]}
    released = frozenset(released)
    captured, stats = {}, {"callbacks": 0, "callback_seconds": 0.0, "published": 0}
    best = initial["makespan"]

    class Progress(cp_model.CpSolverSolutionCallback):
        def __init__(self, model):
            super().__init__()
            self.model, self.variables = model, None

        def on_solution_callback(self):
            nonlocal best
            entered = perf_counter()
            stats["callbacks"] += 1
            try:
                value = round(self.objective_value)
                if value >= best:
                    return
                if self.variables is None:
                    self.variables = {var.name: self.model.get_int_var_from_proto_index(index)
                                      for index, var in enumerate(self.model.proto.variables)}
                rows = []
                for op, options in instance.options.items():
                    machine = initial_rows[op]["machine"]
                    if op in released:
                        machine = next(machine for machine in options
                                       if self.boolean_value(self.variables[f"use_{op}_{machine}"]))
                    rows.append({"operation": op, "machine": machine,
                                 "start": self.value(self.variables[f"start_{op}"]),
                                 "end": self.value(self.variables[f"end_{op}"])})
                schedule = {"instance": instance.name, "assignments": rows, "makespan": value}
                cost.check(cost.verify_candidate(instance, initial, released, schedule) == value,
                           "callback objective mismatch")
                lower_bound = self.best_objective_bound
                cost.check(math.isfinite(lower_bound) and lower_bound <= value + 1e-6,
                           "invalid callback bound")
                verified = perf_counter()
                published = publish(schedule, "incumbent", {"found_at": entered, "worker_verified_at": verified,
                                                           "worker_validation_seconds": verified - entered,
                                                           "lower_bound": lower_bound, "bound_scope": "neighborhood"})
                best = value
                stats["published"] += published is not False
            finally:
                stats["callback_seconds"] += perf_counter() - entered

    def intercepted(solver, model, solution_callback=None):
        cost.check(solution_callback is None, "unexpected original callback")
        captured["model"] = model
        return original_solve(solver, model, Progress(model) if stream else None)

    with patch.object(cp_model.CpSolver, "solve", intercepted):
        result = repair_schedule(instance, initial, released, deadline, workers=1, seed=seed, presolve=False)
    # Final extraction and independent validation finish before publishing a final frame.
    if result.schedule is not None:
        cost.verify_candidate(instance, initial, released, result.schedule)
        publish(result.schedule, "final", {"worker_verified_at": perf_counter(),
                                         "lower_bound": result.lower_bound,
                                         "bound_scope": result.bound_scope, "status": result.status})
    cost.check(result.status not in ("INFEASIBLE", "MODEL_INVALID"), "validated anchor became infeasible")
    model_hash = sha256(str(captured["model"].proto).encode()).hexdigest() if captured else None
    return {"result": asdict(result), "model_sha256": model_hash, **stats}


def worker(request_path):
    directory = Path(request_path).parent
    try:
        from experiments.resched_qualification import configure_cpu
        affinity = configure_cpu()
        from fjsp import parse_instance, validate_schedule
        import repair  # Import before ready; instance loading is inside the common request clock.
        spec = cost.read(request_path)
        atomic_save(directory / "ready.json", {"pid": os.getpid(), "affinity": affinity,
                                               "python": platform.python_version(), "ortools": version("ortools"),
                                               "prefix": sys.prefix, "executable": sys.executable})
        boot_deadline = perf_counter() + spec["config"]["warmup_watchdog_seconds"]
        go, go_read_retries = read_handshake(directory / "go.json", boot_deadline)
        case, task = spec["case"], spec["task"]
        instance = parse_instance(cost.pinned(case["input"], case["sha256"]))
        initial = cost.pinned(case["anchor"], case["anchor_sha256"])
        cost.check(validate_schedule(instance, initial) == case["before"], "anchor mismatch")
        cost.check(instance.name == task["instance"] == case["instance"], "instance mismatch")
        cost.check(len(instance.options) == case["operations"], "size mismatch")
        cost.check(len(task["released"]) == task["k"] == len(set(task["released"]))
                   and set(task["released"]) <= instance.options.keys(), "release mismatch")
        sequence = 0

        def publish(schedule, kind, details):
            nonlocal sequence
            if schedule["makespan"] >= initial["makespan"] or perf_counter() >= go["publish_until"]:
                return False
            sequence += 1
            atomic_save(directory / f"candidate-{sequence:06d}.json",
                        {"sequence": sequence, "task_id": task["id"], "kind": kind,
                         "schedule": schedule, **details})
            return True

        report = observed_repair(instance, initial, task["released"], go["solver_deadline"],
                                 seed=task["solver_seed"], stream=task["condition"]["stream"], publish=publish)
        report.update(task_id=task["id"], request_sha256=cost.digest(request_path),
                      completed_at=perf_counter(), frames=sequence, go_read_retries=go_read_retries)
        atomic_save(directory / "worker-report.json", report)
        return 0
    except Exception as error:
        atomic_save(directory / "worker-error.json", {"error": f"{type(error).__name__}: {error}",
                                                       "traceback": traceback.format_exc()})
        return 1


class CandidateStore:
    """Only parent-verified, persisted, timely candidates can replace the fallback."""

    def __init__(self, directory, spec, started, publish_until):
        from fjsp import parse_instance
        self.directory, self.spec = Path(directory), spec
        self.started, self.publish_until = started, publish_until
        case = spec["case"]
        self.instance = parse_instance(cost.pinned(case["input"], case["sha256"]))
        self.initial = cost.pinned(case["anchor"], case["anchor_sha256"])
        cost.check(cost.verify_candidate(self.instance, self.initial, spec["task"]["released"], self.initial)
                   == case["before"], "fallback mismatch")
        atomic_save(self.directory / "accepted-000000.json", self.initial)
        self.initial_saved_at = perf_counter()
        self.current, self.current_path = self.initial, "accepted-000000.json"
        self.accepted, self.events, self.sequence = [], [], 0

    def consume(self):
        while perf_counter() < self.publish_until:
            sequence = self.sequence + 1
            relative = f"candidate-{sequence:06d}.json"
            path = self.directory / relative
            if not path.exists():
                return
            try:
                try:
                    raw = path.read_bytes()
                except PermissionError as error:
                    self.events.append({"event": "READ_RETRY", "sequence": sequence, "error": str(error)})
                    return  # Retry this sequence; never silently discard a locked frame.
                received = perf_counter()
                frame = json.loads(raw)
                cost.check(frame["sequence"] == sequence and frame["task_id"] == self.spec["task"]["id"],
                           "frame identity mismatch")
                value = cost.verify_candidate(self.instance, self.initial, self.spec["task"]["released"],
                                              frame["schedule"])
                verified = perf_counter()
                if verified >= self.publish_until:
                    self.events.append({"event": "VALIDATION_CUTOFF", "sequence": sequence})
                    return
                self.sequence = sequence
                if value >= self.current["makespan"]:
                    continue
                accepted_path = f"accepted-{sequence:06d}.json"
                atomic_save(self.directory / accepted_path, frame["schedule"])
                saved = perf_counter()
                receipt = {"sequence": sequence, "frame": relative, "frame_sha256": sha256(raw).hexdigest(),
                           "path": accepted_path, "sha256": cost.digest(self.directory / accepted_path),
                           "value": value, "kind": frame["kind"], "received_at": received,
                           "verified_at": verified, "saved_at": saved}
                if saved >= self.publish_until:
                    self.events.append({"event": "SAVE_CUTOFF", **receipt})
                    return
                self.accepted.append(receipt)
                self.current, self.current_path = frame["schedule"], accepted_path
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.sequence = sequence
                self.events.append({"event": "REJECTED", "sequence": sequence,
                                    "error": f"{type(error).__name__}: {error}"})


def execute_request(directory, spec, source_root, worker_command=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cost.save(directory / "request.json", spec)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    command = worker_command or [sys.executable, "-m", "experiments.cp_streaming_delivery_probe",
                                 "--worker", str(directory / "request.json")]
    command = list(command)
    if os.name == "nt" and Path(command[0]).resolve() == Path(sys.executable).resolve():
        # The Windows venv redirector can spawn a second PID. Launch the base
        # interpreter directly, retaining venv detection, so Popen owns the solver.
        env["__PYVENV_LAUNCHER__"] = sys.executable
        command[0] = sys._base_executable
    launched = perf_counter()
    started, store, forced, error = None, None, False, None
    cfg = spec["config"]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            warmup_deadline = launched + cfg["warmup_watchdog_seconds"]
            while not (directory / "ready.json").exists():
                if process.poll() is not None or perf_counter() >= warmup_deadline:
                    raise RuntimeError("worker failed or timed out before ready")
                sleep(cfg["poll_seconds"])
            ready, ready_read_retries = read_handshake(directory / "ready.json", warmup_deadline)
            cost.check(ready["pid"] == process.pid, "wrong ready PID")
            started = perf_counter()
            deadline = started + cfg["request_wall_budget_ms"] / 1000
            publish_until = deadline - cfg["publication_reserve_ms"] / 1000
            stop_at = deadline - cfg["stop_reserve_ms"] / 1000
            solver_deadline = deadline - spec["task"]["condition"]["return_reserve_ms"] / 1000
            atomic_save(directory / "go.json", {"started": started, "deadline": deadline,
                                                "publish_until": publish_until, "stop_at": stop_at,
                                                "solver_deadline": solver_deadline})
            store = CandidateStore(directory, spec, started, publish_until)
            while True:
                store.consume()
                if process.poll() is not None:
                    store.consume()
                    break
                if perf_counter() >= stop_at:
                    forced = True
                    process.terminate()
                    break
                sleep(min(cfg["poll_seconds"], max(0, stop_at - perf_counter())))
            try:
                process.wait(timeout=max(.0001, deadline - perf_counter() - .003))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=max(.0001, deadline - perf_counter() - .001))
            reaped_at = perf_counter()
            atomic_save(directory / "delivered.json", store.current)
            delivered_at = perf_counter()
            row = {"task": spec["task"], "before": spec["case"]["before"],
                   "started": started, "deadline": deadline, "publish_until": publish_until,
                   "stop_at": stop_at, "solver_deadline": solver_deadline,
                   "warmup_seconds": started - launched, "request_seconds": delivered_at - started,
                   "reaped_at": reaped_at, "delivered_at": delivered_at,
                   "process_reaped": process.poll() is not None, "exit_code": process.returncode,
                   "worker_pid": process.pid,
                   "forced_stop": forced, "initial_saved_at": store.initial_saved_at,
                   "ready_read_retries": ready_read_retries,
                   "delivered_value": store.current["makespan"], "selected_path": store.current_path,
                   "accepted": store.accepted, "events": store.events,
                   "delivered_on_time": delivered_at <= deadline and reaped_at <= deadline,
                   "ready": ready}
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            row = {"task": spec["task"], "error": error, "delivered_on_time": False}
        finally:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=cfg["cleanup_grace_seconds"])
                except subprocess.TimeoutExpired:
                    error = "owned child could not be reaped"
            row["process_reaped"] = process.poll() is not None
            row["exit_code"] = process.returncode
    if error:
        row["error"] = error
    for name in ["worker-error.json", "worker-report.json"]:
        if (directory / name).exists():
            row[name.removesuffix(".json")] = cost.read(directory / name)
    if "worker-error" in row or (not forced and row["exit_code"] != 0):
        row["error"] = row.get("error", "unexpected worker failure")
    row["artifacts"] = {path.name: cost.digest(path) for path in sorted(directory.glob("*.json"))}
    row["total_seconds_before_diagnostics"] = perf_counter() - launched
    return row


def prepare(output):
    from experiments.resched_qualification import configure_cpu
    cfg = cost.read(CONFIG)
    environment = {"python": platform.python_version(), "ortools": version("ortools")}
    cost.check(environment == cfg["expected_environment"], "environment mismatch")
    affinity = configure_cpu()
    source = Path(cfg["source_run"])
    for name, expected in cfg["source_evidence_sha256"].items():
        cost.check(cost.digest(source / name) == expected, f"source evidence changed: {name}")
    old = cost.read(source / "plan.json")
    cost.check(cost.read(source / "audit.json")["clean"], "source audit not clean")
    cost.check(old["environment"] == environment, "source environment changed")
    for name, expected in old["source_sha256"].items():
        cost.check(cost.digest(name) == cost.digest(source / "source" / name) == expected,
                   f"original source changed: {name}")
    tasks = build_tasks(old["tasks"], cfg)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in dict.fromkeys([*old["source_sha256"], *SOURCES]):
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        source_hashes[name] = cost.digest(target)
    cases = []
    for original in old["cases"]:
        case = copy.deepcopy(original)
        for kind, key, expected in [("inputs", "input", case["sha256"]),
                                    ("anchors", "anchor", case["anchor_sha256"])]:
            cost.check(cost.digest(case[key]) == expected, f"source {kind} changed")
            path = output / kind / f"{case['instance']}.json"
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(case[key], path)
            case[key] = str(path.resolve())
        cases.append(case)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
            "environment": environment, "platform": platform.platform(), "affinity": affinity,
            "source_sha256": source_hashes, "source_tasks": old["tasks"], "cases": cases, "tasks": tasks}
    cost.save(output / "plan.json", plan)
    cost.save(output / "qualification.json", {"passed": True, "environment": environment,
                                               "requests": len(tasks), "ECT_regenerated": False})
    return plan


def audit(output):
    from fjsp import parse_instance
    plan = cost.read(output / "plan.json")
    cfg = plan["config"]
    for name, expected in plan["source_sha256"].items():
        cost.check(cost.digest(name) == cost.digest(output / "source" / name) == expected,
                   f"source changed: {name}")
    for name, expected in cfg["source_evidence_sha256"].items():
        cost.check(cost.digest(Path(cfg["source_run"]) / name) == expected, "old evidence changed")
    old = cost.read(Path(cfg["source_run"]) / "plan.json")
    cost.check(old["tasks"] == plan["source_tasks"], "source tasks changed")
    cost.check(plan["tasks"] == build_tasks(old["tasks"], cfg), "planned order changed")
    rows = [json.loads(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    cost.check([row["task"] for row in rows] == plan["tasks"], "missing or reassigned rows")
    cases, instances, anchors = {}, {}, {}
    for case in plan["cases"]:
        name = case["instance"]
        original = next(item for item in old["cases"] if item["instance"] == name)
        for key in ["before", "sha256", "anchor_sha256", "operation_ids", "operations"]:
            cost.check(case[key] == original[key], "case identity changed")
        cases[name] = case
        instances[name] = parse_instance(cost.pinned(case["input"], case["sha256"]))
        anchors[name] = cost.pinned(case["anchor"], case["anchor_sha256"])
        cost.verify_candidate(instances[name], anchors[name], [], anchors[name])
    issues, observations, validated, callback_frames = [], [], 0, 0
    model_hashes = defaultdict(set)
    for row in rows:
        task = row["task"]
        directory = output / "requests" / task["id"]
        name, before = task["instance"], cases[task["instance"]]["before"]
        row_valid = True
        try:
            cost.check(not row.get("error") and row.get("process_reaped"), row.get("error", "child not reaped"))
            for relative, expected in row["artifacts"].items():
                cost.check(cost.digest(directory / relative) == expected, "request artifact changed")
            request = cost.read(directory / "request.json")
            cost.check(request == {"task": task, "case": cases[name], "config": cfg}, "request reassigned")
            go = cost.read(directory / "go.json")
            for key in ["started", "deadline", "publish_until", "stop_at", "solver_deadline"]:
                cost.check(go[key] == row[key], "clock disagreement")
            cost.check(abs(row["deadline"] - row["started"] - cfg["request_wall_budget_ms"] / 1000) < 1e-7,
                       "request budget changed")
            for key, reserve in [("publish_until", cfg["publication_reserve_ms"]),
                                 ("stop_at", cfg["stop_reserve_ms"]),
                                 ("solver_deadline", task["condition"]["return_reserve_ms"])]:
                cost.check(abs(row[key] - row["deadline"] + reserve / 1000) < 1e-7, "cutoff changed")
            cost.check(row["ready"]["pid"] == row["worker_pid"], "solver process is not directly owned")
            cost.check({key: row["ready"][key] for key in ["python", "ortools"]} == plan["environment"],
                       "worker environment differs")
            cost.check(row["ready"]["affinity"]["logical_processor"] == plan["affinity"]["logical_processor"],
                       "worker CPU differs")
            cost.check(row["delivered_on_time"] == (row["delivered_at"] <= row["deadline"]
                                                    and row["reaped_at"] <= row["deadline"]), "timing flag changed")
            cost.check(row["initial_saved_at"] <= row["delivered_at"], "fallback timestamp invalid")
            current, selected = before, "accepted-000000.json"
            for receipt in row["accepted"]:
                frame = cost.read(directory / receipt["frame"])
                schedule = cost.read(directory / receipt["path"])
                cost.check(frame["task_id"] == task["id"] and frame["sequence"] == receipt["sequence"],
                           "frame identity mismatch")
                cost.check(cost.digest(directory / receipt["frame"]) == receipt["frame_sha256"]
                           and cost.digest(directory / receipt["path"]) == receipt["sha256"], "receipt hash changed")
                cost.check(schedule == frame["schedule"], "accepted schedule differs from received frame")
                value = cost.verify_candidate(instances[name], anchors[name], task["released"], schedule)
                cost.check(value == receipt["value"] < current, "accepted values not strictly improving")
                cost.check(row["started"] <= receipt["received_at"] <= receipt["verified_at"]
                           <= receipt["saved_at"] < row["publish_until"], "late or backdated acceptance")
                cost.check(frame["kind"] == receipt["kind"] and (task["condition"]["stream"]
                           or frame["kind"] == "final"), "unexpected callback in final-only control")
                cost.check(row["started"] <= frame["worker_verified_at"] <= receipt["received_at"],
                           "worker/parent monotonic clocks disagree")
                current, selected = value, receipt["path"]
            cost.check(row["selected_path"] == selected, "did not deliver best accepted candidate")
            delivered = cost.read(directory / "delivered.json")
            cost.check(delivered == cost.read(directory / selected), "delivered bytes differ from selected schedule")
            value = cost.verify_candidate(instances[name], anchors[name], task["released"], delivered)
            cost.check(value == current == row["delivered_value"], "delivery objective mismatch")
            validated += 1
            # Audit all complete callback frames, including ones not accepted before cutoff.
            for path in sorted(directory.glob("candidate-*.json")):
                frame = cost.read(path)
                cost.check(frame["task_id"] == task["id"], "foreign candidate")
                candidate_value = cost.verify_candidate(instances[name], anchors[name], task["released"], frame["schedule"])
                cost.check(frame["bound_scope"] == "neighborhood" and math.isfinite(frame["lower_bound"])
                           and frame["lower_bound"] <= candidate_value + 1e-6, "invalid frame bound/scope")
                callback_frames += int(frame["kind"] == "incumbent")
            report = row.get("worker-report")
            if report:
                cost.check(report["task_id"] == task["id"] and report["request_sha256"] == row["artifacts"]["request.json"],
                           "native report identity changed")
                native = report["result"]
                if report["model_sha256"] is not None:
                    model_hashes[task["source_task_id"]].add(report["model_sha256"])
                if native["schedule"] is not None:
                    native_value = cost.verify_candidate(instances[name], anchors[name], task["released"], native["schedule"])
                    cost.check(native["status"] in ["FEASIBLE", "OPTIMAL"], "invalid native status")
                    if native["status"] == "OPTIMAL":
                        cost.check(abs(native_value - native["lower_bound"]) < 1e-6, "invalid optimality proof")
            rejected = [event for event in row["events"] if event["event"] == "REJECTED"]
            cost.check(not rejected, f"rejected candidate frames: {rejected}")
        except Exception as error:
            row_valid = False
            issues.append({"task": task["id"], "error": f"{type(error).__name__}: {error}"})
        timely = bool(row_valid and row.get("delivered_on_time") and not row.get("error") and row.get("process_reaped"))
        observations.append({"task_id": task["id"], "source_task_id": task["source_task_id"],
                             "condition": task["condition"]["name"], "instance": name, "k": task["k"],
                             "construction_seed": task["construction_seed"], "solver_seed": task["solver_seed"],
                             "before": before, "value": row.get("delivered_value"),
                             "credited_value": row["delivered_value"] if timely else before,
                             "timely": timely, "request_seconds": row.get("request_seconds"),
                             "forced_stop": row.get("forced_stop", False), "accepted": len(row.get("accepted", [])),
                             "native_report": "worker-report" in row})
    for source_id, hashes in model_hashes.items():
        if len(hashes) > 1:
            issues.append({"source_task_id": source_id, "error": "paired model protos differ"})
    conditions = []
    for condition in cfg["conditions"]:
        selected = [row for row in observations if row["condition"] == condition["name"]]
        groups = defaultdict(list)
        for row in selected:
            groups[(row["instance"], row["construction_seed"], row["k"])].append(row)
        times = sorted(row["request_seconds"] for row in selected if row["request_seconds"] is not None)
        conditions.append({"name": condition["name"], "planned": len(selected),
                           "timely": sum(row["timely"] for row in selected),
                           "improved": sum(row["credited_value"] < row["before"] for row in selected),
                           "stable_improving_groups": sum(len(group) == 3 and all(row["credited_value"] < row["before"]
                                                                                   for row in group) for group in groups.values()),
                           "forced_stops": sum(row["forced_stop"] for row in selected),
                           "native_reports": sum(row["native_report"] for row in selected),
                           "median_ms": cost.median(times) * 1000 if times else None,
                           "p90_ms": times[math.ceil(.9 * len(times)) - 1] * 1000 if times else None,
                           "max_ms": max(times) * 1000 if times else None,
                           "by_scale": [{"operations": size,
                                         "planned": sum(cases[row["instance"]]["operations"] == size for row in selected),
                                         "improved": sum(cases[row["instance"]]["operations"] == size
                                                         and row["credited_value"] < row["before"] for row in selected)}
                                        for size in [250, 500]]})
    paired = []
    for reference in ["final-20", "final-200"]:
        pairs = defaultdict(dict)
        for row in observations:
            if row["condition"] in [reference, "stream-20"]:
                pairs[row["source_task_id"]][row["condition"]] = row
        deltas = []
        for source_id, pair in pairs.items():
            cost.check(set(pair) == {reference, "stream-20"}, "missing paired condition")
            first, second = pair[reference], pair["stream-20"]
            for key in ["instance", "k", "construction_seed", "solver_seed", "before"]:
                cost.check(first[key] == second[key], "paired identity mismatch")
            delta = second["credited_value"] - first["credited_value"]
            deltas.append({"source_task_id": source_id, "instance": first["instance"], "k": first["k"],
                           "construction_seed": first["construction_seed"], "solver_seed": first["solver_seed"],
                           "reference_value": first["credited_value"], "stream_value": second["credited_value"], "delta": delta})
        paired.append({"reference": reference, "counts": dict(Counter("better" if row["delta"] < 0
                                                                 else "worse" if row["delta"] > 0 else "tie" for row in deltas)),
                       "deltas": deltas})
    summary = {"protocol": cfg["protocol"], "planned": len(rows), "validated_deliveries": validated,
               "callback_frames": callback_frames, "matching_model_groups": len(model_hashes),
               "issues": issues, "conditions": conditions,
               "paired": paired, "observations": observations}
    if (output / "summary.json").exists():
        cost.check(cost.read(output / "summary.json") == summary, "re-audit summary changed")
    else:
        cost.save(output / "summary.json", summary)
    result = {"clean": not issues, "issues": issues, "validated_deliveries": validated,
              "plan_sha256": cost.digest(output / "plan.json"), "runs_sha256": cost.digest(output / "runs.jsonl"),
              "summary_sha256": cost.digest(output / "summary.json")}
    cost.save(output / ("audit.json" if not (output / "audit.json").exists() else f"re-audit-{uuid4().hex}.json"), result)
    if not (output / "REPORT.md").exists():
        lines = ["# Streaming incumbent delivery diagnostic", "", "216 fresh requests; shared 1s outer guard.", "",
                 "| Mode | Timely legal deliveries | Timely improved | All-3-seed improving groups | Forced stops | Max ms |",
                 "|---|---:|---:|---:|---:|---:|"]
        for item in conditions:
            lines.append(f"| {item['name']} | {item['timely']}/{item['planned']} | {item['improved']} | "
                         f"{item['stable_improving_groups']} | {item['forced_stops']} | {item['max_ms']} |")
        lines += ["", "Pair counts compare stream-20 against each fresh reference (lower Cmax is better):", "",
                  "```json", json.dumps([{key: item[key] for key in ["reference", "counts"]} for item in paired], indent=2),
                  "```", "", "Publication cutoff 960ms, forced-stop target 970ms; reaping and final file must finish by 1000ms.",
                  "Cold imports and diagnostic serialization are separate. Prior batches used different timing boundaries and are not rescored.",
                  "Interrupted native results are unknown; a saved incumbent proves feasibility only. No training or generalization claim.", ""]
        (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit_clean": not issues, "conditions": conditions,
                      "pairs": [{key: item[key] for key in ["reference", "counts"]} for item in paired],
                      "issues": len(issues)}), flush=True)
    return not issues


def run(output):
    plan = prepare(output)
    cfg = plan["config"]
    cases = {case["instance"]: case for case in plan["cases"]}
    source_root = output / "source"
    started, interrupted = perf_counter(), False
    print(json.dumps({"prepared": str(output), "requests": len(plan["tasks"])}), flush=True)
    with (output / "runs.jsonl").open("x", encoding="utf-8", buffering=1) as file:
        for number, task in enumerate(plan["tasks"], 1):
            if interrupted or perf_counter() - started >= cfg["batch_wall_limit_seconds"]:
                row = {"task": task, "error": "NOT_STARTED: interrupted or batch limit", "process_reaped": True}
            else:
                spec = {"task": task, "case": cases[task["instance"]], "config": cfg}
                row = execute_request(output / "requests" / task["id"], spec, source_root)
                interrupted = row.get("error", "").startswith("KeyboardInterrupt")
            file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            if number % 12 == 0 or number == len(plan["tasks"]):
                print(json.dumps({"finished": number, "planned": len(plan["tasks"]),
                                  "wall_seconds": round(perf_counter() - started, 2),
                                  "last_error": row.get("error")}), flush=True)
    cost.save(output / "runtime.json", {"main_wall_seconds": perf_counter() - started})
    return audit(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if args.output is None:
        parser.error("output is required")
    return 0 if (audit(args.output.resolve()) if args.audit else run(args.output.resolve())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
