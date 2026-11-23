"""Bounded, serial qualification of a pinned original ReSched checkpoint."""

import argparse
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

from fjsp import parse_instance, validate_schedule


CONFIG = Path("experiments/resched_qualification.json")
SOURCE_FILES = ("experiments/resched_qualification.py", "experiments/resched_qualification.json",
                "experiments/resched_adapter.py", "experiments/resched_fetch.py", "fjsp.py")
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configure_cpu():
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    if os.name != "nt":
        allowed = os.sched_getaffinity(0)
        selected = min(allowed)
        os.sched_setaffinity(0, {selected})
        return {"logical_processor": selected}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = kernel.GetCurrentProcess()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
        raise OSError("cannot inspect CPU affinity")
    selected = process.value & -process.value
    if not kernel.SetProcessAffinityMask(handle, selected):
        raise OSError("cannot select one logical CPU")
    return {"original_mask": process.value, "selected_mask": selected,
            "logical_processor": selected.bit_length() - 1}


def pinned_input(case):
    raw = Path(case["input"]).read_bytes()
    if sha256(raw).hexdigest() != case["sha256"]:
        raise ValueError("input fingerprint changed")
    instance = parse_instance(json.loads(raw))
    if instance.name != case["instance"] or len(instance.options) != case["operations"]:
        raise ValueError("case identity or operation count changed")
    return instance


def worker(request_path):
    request = json.loads(Path(request_path).read_bytes())
    out = Path(request["output"])
    try:
        configure_cpu()
        case, cfg = request["case"], request["config"]
        instance = pinned_input(case)
        from experiments.resched_adapter import construct, load_runtime
        runtime = load_runtime(cfg["snapshot"], cfg["acquisition_sha256"], cfg["seed"])
        save(out / "load.json", runtime["load_info"])
        def progress(value):
            save(out / "progress.json", value)
        schedule, details = construct(runtime, instance, cfg["seed"], request["work_deadline"], progress)
        save(out / "candidate.json", schedule)
        save(out / "inference.json", details)
        return 0
    except Exception as error:
        save(out / "worker-error.json", {"type": type(error).__name__, "message": str(error),
                                          "traceback": traceback.format_exc()})
        return 1


def trial(case, cfg, out):
    out.mkdir()
    result = {"instance": case["instance"], "operations": case["operations"], "state": "ERROR",
              "makespan": None, "delivered_on_time": False, "process_reaped": True}
    started = perf_counter()
    deadline = started + cfg["total_ms"] / 1000
    process = None
    try:
        instance = pinned_input(case)
        result["input_finish_ms"] = (perf_counter() - started) * 1000
        save(out / "request.json", {"case": case, "config": cfg, "output": str(out.resolve()),
                                     "work_deadline": deadline - cfg["worker_reserve_ms"] / 1000})
        command = [sys.executable, "-m", "experiments.resched_qualification", "--worker", str((out / "request.json").resolve())]
        with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                process.wait(timeout=max(0, deadline - perf_counter() - cfg["stop_reserve_ms"] / 1000))
            except subprocess.TimeoutExpired:
                result["state"] = "TIMEOUT"
                process.terminate()
                process.wait(timeout=max(0.001, deadline - perf_counter()))
        result["exit_code"] = process.returncode
        if process.returncode == 0:
            schedule = json.loads((out / "candidate.json").read_bytes())
            value = validate_schedule(instance, schedule)
            save(out / "delivery.json", schedule)
            result["delivery_ms"] = (perf_counter() - started) * 1000
            result["delivered_on_time"] = result["delivery_ms"] <= cfg["total_ms"]
            result["native_makespan"] = value
            if result["delivered_on_time"]:
                result.update(state="SUCCESS", makespan=value)
            else:
                result["state"] = "LATE"
        elif (out / "worker-error.json").exists():
            result["worker_error"] = json.loads((out / "worker-error.json").read_bytes())
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=max(0.001, deadline - perf_counter()))
            except subprocess.TimeoutExpired:
                pass
            result["process_reaped"] = process.poll() is not None
        result["elapsed_ms"] = (perf_counter() - started) * 1000
        if result["elapsed_ms"] > cfg["total_ms"] and result["state"] == "SUCCESS":
            result.update(state="LATE", delivered_on_time=False, makespan=None)
        save(out / "result.json", result)
    return result


def audit(out):
    plan = json.loads((out / "plan.json").read_bytes())
    rows = json.loads((out / "summary.json").read_bytes())["results"]
    checked = []
    for case, row in zip(plan["config"]["cases"], rows, strict=True):
        instance = pinned_input(case)
        if row["state"] == "SUCCESS":
            delivery = json.loads((out / case["instance"] / "delivery.json").read_bytes())
            assert validate_schedule(instance, delivery) == row["makespan"]
            assert row["delivered_on_time"] and row["process_reaped"]
            checked.append({"instance": instance.name, "makespan": row["makespan"],
                            "delivery_sha256": sha256((out / case["instance"] / "delivery.json").read_bytes()).hexdigest()})
    from experiments.resched_adapter import verify_snapshot
    verify_snapshot(plan["config"]["snapshot"], plan["config"]["acquisition_sha256"])
    for name, expected in plan["code_sha256"].items():
        assert sha256(Path(name).read_bytes()).hexdigest() == expected
    report = {"audited_planned_cases": len(rows), "validated_deliveries": checked,
              "code_and_upstream_unchanged": True, "holdout_used": False}
    save(out / "audit.json", report)
    return report


def run(out):
    cfg = json.loads(CONFIG.read_bytes())
    out.mkdir(parents=True, exist_ok=False)
    affinity = configure_cpu()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "affinity": affinity, "dependencies": {k: version(k) for k in ("torch", "numpy")},
                            "threads": {k: os.environ[k] for k in THREAD_KEYS}},
            "code_sha256": {name: sha256(Path(name).read_bytes()).hexdigest() for name in SOURCE_FILES}}
    save(out / "plan.json", plan)
    (out / "source").mkdir()
    for name in SOURCE_FILES:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    rows, tiny_passed = [], True
    for index, case in enumerate(cfg["cases"]):
        if not tiny_passed:
            row = {"instance": case["instance"], "state": "SKIPPED_DEPENDENCY", "makespan": None}
        else:
            row = trial(case, cfg, out / case["instance"])
        rows.append(row)
        if index == 0:
            tiny_passed = row["state"] == "SUCCESS"
        summary = {"protocol": cfg["protocol"], "results": rows,
                   "qualification_passed": len(rows) == len(cfg["cases"]) and all(r["state"] == "SUCCESS" for r in rows),
                   "holdout_used": False, "new_training": False, "cp_used": False}
        save(out / "summary.json", summary)
        print(json.dumps(row), flush=True)
    audit(out)
    return 0 if summary["qualification_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    if args.output is None:
        parser.error("provide a fresh output directory")
    if args.audit:
        print(json.dumps(audit(args.output)))
    else:
        raise SystemExit(run(args.output))
