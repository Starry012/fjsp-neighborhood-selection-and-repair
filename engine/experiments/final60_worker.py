"""Timed entry points for full-model solvers and the frozen active CP control."""

from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


class Delivery:
    def __init__(self, request, folder):
        self.folder = folder
        self.started = request["started"]
        self.deadline = self.started + request["seconds"]
        self.best = None
        self.trace = []
        self.native_count = 0

    def publish(self, instance, schedule, source):
        from fjsp import validate_schedule

        score = validate_schedule(instance, schedule)
        if source != "ECT":
            self.native_count += 1
        if self.best is not None and score >= self.best:
            return
        if perf_counter() >= self.deadline:
            return
        raw = json.dumps(schedule, separators=(",", ":")).encode()
        digest = sha256(raw).hexdigest()
        path = self.folder / f"frame-{len(self.trace):04d}.json"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        at = (perf_counter() - self.started) * 1000
        if perf_counter() <= self.deadline:
            record = {"ms": at, "Cmax": score, "source": source, "file": path.name, "sha256": digest}
            with (self.folder / "frames.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            self.best = score
            self.trace.append(record)


def chuffed(exe, instance, initial, folder, delivery, seed):
    from experiments.chuffed_development_probe import TIGHT_MODEL, decode, model_input, tighten_input

    data, modes, operation_number = model_input(instance, initial)
    tighten_input(data, instance, operation_number, True)
    encoded = folder / "model_input.json"
    encoded.write_text(json.dumps(data), encoding="utf-8")
    remaining = int((delivery.deadline - perf_counter()) * 1000) - 2000
    check(remaining > 0, "Chuffed construction used all search time")
    command = [str(exe), "--solver", "org.chuffed.chuffed", "--random-seed", str(seed), "--free-search",
               "--json-stream", "--intermediate-solutions", "--statistics", "--time-limit", str(remaining),
               "--cleanup-time-limit", "300", str(TIGHT_MODEL), str(encoded)]
    status = None
    with (folder / "solver.jsonl").open("w", encoding="utf-8") as log, (folder / "solver.stderr").open("wb") as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, encoding="utf-8") as process:
            for line in process.stdout:
                log.write(line)
                event = json.loads(line)
                if event["type"] == "error":
                    raise ValueError(event.get("message", "Chuffed output error"))
                if event["type"] == "solution":
                    schedule = decode(instance, json.loads(event["output"]["default"]), modes, operation_number)
                    delivery.publish(instance, schedule, "Chuffed")
                if event["type"] == "status":
                    status = event["status"]
            check(process.wait() == 0, "Chuffed exited with an error")
    check(delivery.native_count and status not in ("UNSATISFIABLE", "ERROR", "UNBOUNDED"),
          f"Chuffed returned no usable native schedule or invalid status: {status}")
    return status, delivery.best if status == "OPTIMAL_SOLUTION" else None


def portable_legacy_ranker(cfg):
    """Load the same frozen tensor state on the final server's CPU PyTorch build."""
    import torch
    from baseline import load_pinned_json
    from v3_ranker import TabularRanker

    rank = load_pinned_json(Path("experiments/v3_ranker_dev.json"), cfg["ranker_config_sha256"])
    root = Path("outputs/v3-ranker-dev")
    manifest = load_pinned_json(root / "artifacts.json", cfg["checkpoint_manifest_sha256"])
    trained = load_pinned_json(root / "run.json", manifest["run.json"])
    check(trained["config"] == rank and sha256(Path("v3_ranker.py").read_bytes()).hexdigest() ==
          trained["code_sha256"]["v3_ranker.py"], "legacy ranker training identity differs")
    name = "tabular-42.pt"
    path = root / name
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == manifest[name], "legacy ranker weight differs")
    from io import BytesIO
    state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    model = TabularRanker(rank["width"], state["center"][None, :])
    model.load_state_dict(state, strict=True)
    return model.eval(), {name: manifest[name]}


def active(request, delivery):
    # Isolate the archived V4 modules from the frozen current warm-start modules.
    legacy = Path("legacy").resolve()
    sys.path.insert(0, str(legacy))
    os.chdir(legacy)
    from baseline import load_pinned_json
    from experiments.v4_active_baseline import configure_cpu, execute
    from fjsp import parse_instance

    configure_cpu()
    cfg = read(Path("experiments/v4_active_baseline.json"))
    cfg["total_ms"] = request["seconds"] * 1000
    base = load_pinned_json(Path(cfg["source_config"]), cfg["source_config_sha256"])
    case = request["case"]
    path = Path(request["input"])
    instance = parse_instance(load_pinned_json(path, case["sha256"]))
    result = execute(path, case, "active-strict", request["seed"], cfg, base, delivery.folder / "legacy.json",
                     trial_started=delivery.started, trial_deadline=delivery.deadline - .65,
                     on_incumbent=lambda schedule, _, source: delivery.publish(instance, schedule, source),
                     ranker_loader=portable_legacy_ranker)
    (delivery.folder / "legacy-result.json").write_text(json.dumps(result), encoding="utf-8")
    check(result["stop"] != "ERROR" and result["schedule"] is not None and
          result["elapsed_ms"] <= request["seconds"] * 1000, str(result.get("error", result["stop"])))
    check(delivery.best is not None and delivery.best <= result["schedule"]["makespan"], "legacy final schedule was not published")
    return result["stop"], None


def solve(request, delivery):
    from baseline import load_pinned_json, solve_ect
    from fjsp import parse_instance

    case, method = request["case"], request["method"]
    instance = parse_instance(load_pinned_json(Path(request["input"]), case["sha256"]))
    check(len(instance.options) == case["operations"], "input size differs")
    initial = solve_ect(instance, delivery.deadline - .9).schedule
    delivery.publish(instance, initial, "ECT")
    if method == "Chuffed":
        return chuffed(Path(request["minizinc"]), instance, initial, delivery.folder, delivery, request["seed"])

    from repair import repair_schedule

    result = repair_schedule(instance, initial, instance.options, delivery.deadline - .85, workers=1,
                             seed=request["seed"], presolve=request["presolve"],
                             on_incumbent=lambda schedule: delivery.publish(instance, schedule, "CP-SAT"))
    check(result.bound_scope == "global" and result.status in ("FEASIBLE", "OPTIMAL")
          and result.schedule is not None, f"CP-SAT native status: {result.status}")
    delivery.publish(instance, result.schedule, "CP-SAT final")
    check(result.lower_bound is not None and result.lower_bound <= delivery.best + 1e-6,
          "global CP-SAT lower bound exceeds a verified schedule")
    return result.status, result.lower_bound


def main(path):
    import ctypes
    from experiments.offline_job_state import durable

    request = read(path)
    check(ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == request["parent"],
          "worker parent ownership differs")
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    check(os.sched_getaffinity(0) == {request["cpu"]}, "worker CPU affinity differs")
    delivery = Delivery(request, path.parent)
    legacy = request["method"] == "active-strict"
    status, bound = (active if legacy else solve)(request, delivery)
    result_path = path.parent / "worker-result.json"
    result = {
        "status": status, "lower_bound": bound, "native_schedules_validated": delivery.native_count,
        "last_ms": (perf_counter() - delivery.started) * 1000, "cpu": request["cpu"],
    }
    if legacy:
        # This isolated worker owns no solver subprocess. All delivery and diagnostic
        # writers are closed. Avoid unbounded interpreter/native-library finalizers.
        durable(result_path, result)
        os._exit(0)
    result_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
