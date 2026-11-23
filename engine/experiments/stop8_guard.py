"""Publish validated incumbents and bound a separate search process's lifetime."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

from experiments.stop8_monitor import append_receipt
from baseline import load_pinned_json, solve_ect
from fjsp import parse_instance, validate_schedule


def atomic_json(path, value):
    """Only complete JSON replaces a published file, on the same filesystem."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CandidatePublisher:
    """Append-only worker mailbox; a published frame is never replaced."""

    protocol = "immutable-sequence-v2"

    def __init__(self, directory, started):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = started
        self.sequence = 0
        self.best = math.inf

    def __call__(self, schedule, available_ms, source):
        value = schedule["makespan"]
        if value >= self.best:
            return
        self.sequence += 1
        name = f"candidate-{self.sequence:06d}.json"
        atomic_json(self.directory / name, {"sequence": self.sequence, "schedule": schedule,
                    "available_ms": available_ms, "source": source})
        self.best = value


def stop_worker(process, deadline, return_reserve_ms):
    """Terminate only our direct child; never wait unboundedly for search."""
    terminated = process.poll() is None
    if terminated:
        process.terminate()
    remaining = max(0, deadline - perf_counter() - return_reserve_ms / 1000)
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=max(0, deadline - perf_counter()))
        except subprocess.TimeoutExpired:
            pass
    return {"terminated": terminated, "exit_code": process.poll(), "still_running": process.poll() is None}


def run_guarded(input_path, case, output, worker_request, *, total_ms=60000,
                publish_reserve_ms=500, stop_reserve_ms=250, return_reserve_ms=50,
                poll_ms=10, initial_schedule=None, worker_command=None, worker_pass_fds=(),
                common_started=None, initial_source=None):
    """Count read/ECT/startup/IPC/verification/delivery/child cleanup in one budget.

    An early-published incumbent survives a hung or crashed worker. Actual return
    time is still measured; OS or filesystem stalls must not be called timely.
    initial_schedule is for correctness fixtures; production runs construct ECT.
    """
    if not (math.isfinite(total_ms) and total_ms > publish_reserve_ms > stop_reserve_ms > return_reserve_ms >= 0
            and math.isfinite(poll_ms) and poll_ms > 0):
        raise ValueError("use finite positive budgets and descending delivery reserves")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "worker").mkdir()
    (output / "published").mkdir()
    started = perf_counter() if common_started is None else common_started
    if not math.isfinite(started) or started > perf_counter():
        raise ValueError("invalid shared wall clock")
    deadline = started + total_ms / 1000
    publish_until = deadline - publish_reserve_ms / 1000
    stop_at = deadline - stop_reserve_ms / 1000
    elapsed = lambda: (perf_counter() - started) * 1000
    events, receipts = [], []
    current = instance = process = None
    initial = None
    last_sequence = 0
    state = "ERROR"
    error = None
    cleanup = {"terminated": False, "exit_code": None, "still_running": False}

    def commit(schedule, source, available_ms=None):
        nonlocal current
        validated_ms = elapsed()
        number = len(receipts) + 1
        name = f"published/incumbent-{number:06d}.json"
        atomic_json(output / name, schedule)
        written_ms = elapsed()
        eligible = written_ms <= total_ms
        receipts.append({"file": name, "source": source, "makespan": schedule["makespan"],
                         "worker_available_ms": available_ms, "validated_ms": validated_ms,
                         "written_ms": written_ms, "eligible": False,
                         "sha256": sha256((output / name).read_bytes()).hexdigest()})
        if eligible:
            atomic_json(output / "delivery.json", schedule)
            delivered_ms = elapsed()
            receipts[-1]["delivery_written_ms"] = delivered_ms
            if delivered_ms <= total_ms:
                receipts[-1]["eligible"] = True
                append_receipt(output, delivered_ms, source, schedule["makespan"])
                current = schedule
                return True
        return False

    def consume():
        nonlocal last_sequence
        while True:
            sequence = last_sequence + 1
            candidate_path = output / "worker" / f"candidate-{sequence:06d}.json"
            if not candidate_path.exists():
                return
            if perf_counter() >= publish_until:
                events.append({"event": "PUBLICATION_CUTOFF", "ms": elapsed(), "sequence": sequence})
                return
            try:
                try:
                    contents = candidate_path.read_bytes()
                except PermissionError as exc:
                    # Windows may expose a renamed frame before releasing its
                    # sharing lock. Retry this same sequence on the next poll.
                    events.append({"event": "READ_RETRY", "ms": elapsed(),
                                   "sequence": sequence, "error": str(exc)})
                    return
                last_sequence = sequence
                candidate = json.loads(contents)
                if candidate["sequence"] != sequence:
                    raise ValueError("candidate sequence differs from filename")
                value = validate_schedule(instance, candidate["schedule"])
                if perf_counter() >= publish_until:
                    events.append({"event": "VALIDATION_CUTOFF", "ms": elapsed(), "sequence": sequence})
                    return
                schedule = {**candidate["schedule"], "makespan": value}
                if value < current["makespan"]:
                    if not commit(schedule, candidate.get("source", "worker"), candidate.get("available_ms")):
                        raise TimeoutError("publication did not complete by delivery deadline")
                    events.append({"event": "ACCEPTED", "ms": elapsed(), "sequence": sequence, "makespan": value})
                else:
                    events.append({"event": "NOT_BETTER", "ms": elapsed(), "sequence": sequence, "makespan": value})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                last_sequence = sequence
                events.append({"event": "REJECTED", "ms": elapsed(), "error": f"{type(exc).__name__}: {exc}", "sequence": sequence})

    try:
        instance = parse_instance(load_pinned_json(Path(input_path), case["sha256"]))
        if len(instance.options) != case["operations"]:
            raise ValueError("instance size differs from frozen case")
        input_finish_ms = elapsed()
        initial = initial_schedule if initial_schedule is not None else solve_ect(instance, publish_until).schedule
        if initial is None:
            raise TimeoutError("ECT did not produce an initial schedule")
        initial = {**initial, "makespan": validate_schedule(instance, initial)}
        if not commit(initial, "ECT" if initial_schedule is None else (initial_source or "fixture-initial")):
            raise TimeoutError("initial publication exceeded the delivery deadline")
        if perf_counter() >= publish_until:
            raise TimeoutError("initial publication consumed the search budget")
        request = {"input_path": str(Path(input_path).resolve()), "case": case, "initial": initial,
                   "started": started, "worker_deadline": publish_until, "outer_deadline": deadline,
                   "output": str(output.resolve()), "spec": worker_request}
        request["mailbox_protocol"] = CandidatePublisher.protocol
        atomic_json(output / "request.json", request)
        command = worker_command or [sys.executable, "-m", "experiments.v4_delivery_worker"]
        with (output / "worker.stdout.txt").open("wb") as stdout, (output / "worker.stderr.txt").open("wb") as stderr:
            process = subprocess.Popen([*command, str((output / "request.json").resolve())],
                                       stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                       **({"pass_fds": tuple(worker_pass_fds)} if os.name != "nt" else {}),
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            worker_started_ms = elapsed()
            while process.poll() is None and perf_counter() < stop_at:
                consume()
                sleep(min(poll_ms / 1000, max(0, stop_at - perf_counter())))
            consume()
            state = "WORKER_DONE" if process.poll() == 0 else ("WORKER_ERROR" if process.poll() is not None else "DEADLINE")
            cleanup = stop_worker(process, deadline, return_reserve_ms)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process is not None:
            cleanup = stop_worker(process, deadline, return_reserve_ms)
    finished_ms = elapsed()
    result = {"state": state, "error": error, "instance": case["instance"], "initial": initial,
              "mailbox_protocol": CandidatePublisher.protocol,
              "started_perf_counter": started,
              "schedule": current, "elapsed_ms": finished_ms, "budget_ms": total_ms,
              "input_finish_ms": locals().get("input_finish_ms"),
              "worker_started_ms": locals().get("worker_started_ms"),
              "publication_cutoff_ms": total_ms - publish_reserve_ms,
              "stop_at_ms": total_ms - stop_reserve_ms, "events": events, "receipts": receipts,
              "worker": {"pid": None if process is None else process.pid, **cleanup},
              "delivered_on_time": current is not None and finished_ms <= total_ms and not cleanup["still_running"]}
    result["credited_makespan"] = current["makespan"] if result["delivered_on_time"] else None
    # The caller may persist diagnostics after receiving this return value.
    return result
