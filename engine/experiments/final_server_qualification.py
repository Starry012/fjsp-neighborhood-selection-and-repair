"""Linux-only environment and delivery verification on frozen development cases."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
from time import perf_counter, sleep
import traceback

from experiments import glnsa_runtime, server_resource_snapshot, server_storage_snapshot
from experiments.offline_job_state import JobState, durable as atomic_json, effective_status
from fjsp import parse_instance, validate_schedule


ROOT = Path(__file__).resolve().parents[1]
CP_VERSION = "9.15.6755"
TINY = {"instance": "tiny", "jobs": [{"job": 5, "operations": [11, 22]}, {"job": 19, "operations": [50]}],
        "operations": [{"operation": op, "machine": m, "duration": p}
                       for op, m, p in [(11, 2, 2), (11, 17, 9), (22, 17, 3), (50, 17, 2)]]}


def read(path):
    return json.loads(path.read_bytes())


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def snapshots(directory, phase):
    project = directory.parents[2]
    for name, capture in (
        ("resource", lambda: server_resource_snapshot.snapshot(phase=phase)),
        ("storage", lambda: server_storage_snapshot.snapshot(phase=phase, account=project.parent, project=project)),
    ):
        try:
            result = capture()
        except Exception as error:
            result = {"status": "UNKNOWN", "reason": f"{type(error).__name__}: {error}"}
        atomic_json(directory / f"{phase}-{name}.json", result)


def choose_cpu(allowed):
    stat = Path("/proc/stat")
    first = server_resource_snapshot.cpu_counters(stat.read_text())
    sleep(0.25)
    sample = server_resource_snapshot.cpu_delta(first, server_resource_snapshot.cpu_counters(stat.read_text()))
    busy = {cpu: sample.get(f"cpu{cpu}", {}).get("busy_percent") for cpu in sorted(allowed)}
    measured = {cpu: value for cpu, value in busy.items() if value is not None}
    check(measured, "no usable CPU sample in the current affinity")
    return min(measured, key=lambda cpu: (measured[cpu], cpu)), busy


def capture_chuffed(exe, instance, initial, output, end):
    from experiments.chuffed_development_probe import TIGHT_MODEL, decode, model_input, tighten_input

    data, modes, operation_number = model_input(instance, initial)
    tighten_input(data, instance, operation_number, True)
    encoded = output / "model_input.json"
    atomic_json(encoded, data)
    limit = int((end - perf_counter()) * 1000) - 2000
    check(limit > 0, "Chuffed construction used all search time")
    command = [str(exe), "--solver", "org.chuffed.chuffed", "--random-seed", "42", "--free-search",
               "--json-stream", "--intermediate-solutions", "--statistics", "--time-limit", str(limit),
               "--cleanup-time-limit", "300", str(TIGHT_MODEL), str(encoded)]
    stdout = output / "solver.jsonl"
    with stdout.open("wb") as log, (output / "solver.stderr").open("wb") as errors:
        subprocess.run(command, stdout=log, stderr=errors, check=True)
    schedules, status = [], None
    with stdout.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["type"] == "error":
                raise ValueError(event.get("message", "Chuffed output error"))
            if event["type"] == "solution":
                schedules.append(decode(instance, json.loads(event["output"]["default"]), modes, operation_number))
            if event["type"] == "status":
                status = event["status"]
    check(schedules and status not in ("UNSATISFIABLE", "ERROR"), f"Chuffed returned no usable schedule: {status}")
    return schedules, status, None


def run_child(request_path):
    from baseline import load_pinned_json, solve_ect
    from repair import repair_schedule

    request = read(request_path)
    output, started = request_path.parent, request["started"]
    end = started + request["seconds"]
    check(os.sched_getaffinity(0) == {request["cpu"]}, "trial CPU affinity differs")
    instance = parse_instance(load_pinned_json(Path(request["input"]), request["sha256"]))
    initial = solve_ect(instance, end - 0.8).schedule
    initial_value = validate_schedule(instance, initial)
    method = request["method"]
    if method == "chuffed":
        schedules, status, bound = capture_chuffed(Path(request["minizinc"]), instance, initial, output, end)
    else:
        solve = repair_schedule(instance, initial, tuple(instance.options), end - 0.85,
                                workers=1, seed=42, presolve=method == "ortools-on")
        check(solve.bound_scope == "global", "CP-SAT returned a local bound")
        check(solve.status in ("OPTIMAL", "FEASIBLE") and solve.schedule is not None,
              f"CP-SAT returned no native feasible schedule: {solve.status}")
        schedules, status, bound = [solve.schedule], solve.status, solve.lower_bound
    best = min([initial, *schedules], key=lambda schedule: validate_schedule(instance, schedule))
    if request["tiny"]:
        check(status in ("OPTIMAL", "OPTIMAL_SOLUTION") and best["makespan"] == 5,
              f"small fixture must return a proven optimum of 5: {status}")
    check(bound is None or bound <= best["makespan"] + 1e-6, "lower bound exceeds a verified schedule")
    atomic_json(output / "schedule.json", best)
    atomic_json(output / "result.json", {"instance": instance.name, "method": method, "status": status,
        "native_schedules": len(schedules), "ect": initial_value, "makespan": best["makespan"],
        "bound": bound, "cpu": request["cpu"], "elapsed_seconds": perf_counter() - started})


def reap_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=0.35)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=1)


def run_trial(directory, case, method, seconds, cpu, exe):
    output = directory / f"{case['id']}-{method}"
    output.mkdir()
    request = {"method": method, "seconds": seconds, "cpu": cpu, "minizinc": str(exe), **case}
    path = output / "request.json"
    started = perf_counter()
    atomic_json(path, {**request, "started": started})
    with (output / "worker.log").open("wb") as log:
        proc = subprocess.Popen([sys.executable, "-B", "-m", "experiments.final_server_qualification", "worker", str(path)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=max(0.001, started + seconds - perf_counter()))
        finally:
            reap_group(proc)
    elapsed = perf_counter() - started
    check(code == 0 and elapsed <= seconds, f"trial failed or returned late: exit={code}, elapsed={elapsed:.3f}")
    result = read(output / "result.json")
    schedule = output / "schedule.json"
    return {**result, "total_seconds": elapsed, "budget_seconds": seconds, "check": "PASS",
            "schedule": str(schedule), "schedule_sha256": glnsa_runtime.digest(schedule)}


def ortools_version():
    try:
        return version("ortools")
    except PackageNotFoundError:
        return None


def environment(exe):
    check(platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"), "requires Linux x86_64")
    check(ortools_version() == CP_VERSION, f"expected OR-Tools {CP_VERSION}; found {ortools_version()}")
    mini = subprocess.check_output([str(exe), "--version"], text=True, timeout=8).splitlines()[0]
    solvers = subprocess.check_output([str(exe), "--solvers"], text=True, timeout=8)
    check("version 2.11.0" in mini and "Chuffed 0.14.0" in solvers, "expected MiniZinc 2.11.0 and Chuffed 0.14.0")
    return {"python": platform.python_version(), "ortools": version("ortools"), "minizinc": mini,
            "chuffed": "0.14.0", "platform": platform.platform(), "binary_sha256": glnsa_runtime.digest(exe)}


def verify_stall80(directory):
    native = directory / "stall80"
    native.mkdir()
    binary, compiler = glnsa_runtime.compile_native(ROOT / "experiments/glnsa.cpp", native)
    instance = parse_instance(TINY)
    body, operations, machines = glnsa_runtime.encode(instance)
    path = native / "input.txt"
    path.write_text(body + "1 0 0 0 1 1\n", encoding="ascii")
    subprocess.run([str(binary), "inspect", str(path), str(native), "42", "1000", str(os.getpid())],
                   check=True, timeout=5, capture_output=True)
    _, schedule = glnsa_runtime.frame(instance, native / "frame-1", operations, machines)
    check(schedule["makespan"] == 5, "Stall80 decoder fixture score differs")
    return {**compiler, "decoder_fixture": 5}


def resume_record(directory, identity):
    path = directory / "report.json"
    if not path.is_file():
        return {}
    old = read(path)
    check(old["identity"] == identity, "qualification inputs, source or executable changed; start a new directory")
    return {row["id"]: row for row in old["rows"] if row["check"] == "PASS"}


def verify_delivery(case, row):
    from baseline import load_pinned_json

    schedule = Path(row["schedule"])
    check(glnsa_runtime.digest(schedule) == row["schedule_sha256"], "saved schedule hash differs")
    instance = parse_instance(load_pinned_json(Path(case["input"]), case["sha256"]))
    check(validate_schedule(instance, read(schedule)) == row["makespan"], "saved schedule score differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "development", "worker", "status"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("minizinc", type=Path, nargs="?")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == "worker":
        run_child(directory)
        return 0
    if args.mode == "status":
        print(json.dumps(effective_status(read(directory / "status.json")), indent=2))
        return 0
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupt)

    manifest = read(ROOT / "manifest.json")
    exe = args.minizinc.resolve()
    selected = [] if args.mode == "smoke" else manifest["cases"]
    marker = directory / "identity.json"
    def digest_if_present(path):
        return glnsa_runtime.digest(path) if path.is_file() else None

    identity = {"mode": args.mode, "manifest": glnsa_runtime.digest(ROOT / "manifest.json"),
                "minizinc": digest_if_present(exe), "backend": digest_if_present(exe.with_name("fzn-chuffed")),
                "python": sys.version, "ortools": ortools_version()}
    if marker.exists():
        check(read(marker) == identity, "resume expects the same mode and software snapshot")
        status_path = directory / "status.json"
        old_status = effective_status(read(status_path)) if status_path.exists() else {"state": "INTERRUPTED_OR_UNCONFIRMED"}
        if old_status["state"] == "COMPLETED":
            print("Qualification is already complete.")
            return 0
    else:
        atomic_json(marker, identity)
    state = JobState(directory, total=4 + 2 * len(selected))
    allowed = os.sched_getaffinity(0)
    done, reason = [], None
    attempt = directory / f"attempt-{state.data['owner']['pid']}-{state.data['owner']['start_ticks']}"
    attempt.mkdir()
    report = {"identity": identity, "rows": done, "scope": "development qualification, not formal comparison"}
    try:
        state.update(phase="snapshot before", state="RUNNING")
        snapshots(attempt, "before")
        saved = resume_record(directory, identity)
        for item in manifest["files"]:
            check(glnsa_runtime.digest(ROOT / item["path"]) == item["sha256"], f"file hash differs: {item['path']}")
        report["environment"] = environment(exe)
        cpu, traffic = choose_cpu(allowed)
        report.update(cpu=cpu, sampled_busy_percent=traffic)
        os.sched_setaffinity(0, {cpu})
        state.update(phase="Stall80 compiler and decoder", current_case="Stall80")
        report["stall80"] = verify_stall80(attempt)
        done.append({"id": "stall80", "check": "PASS"})
        state.update(completed=len(done))
        tiny = attempt / "tiny.json"
        atomic_json(tiny, TINY)
        tiny_case = {"id": "tiny", "input": str(tiny), "sha256": glnsa_runtime.digest(tiny), "tiny": True}
        jobs = [(tiny_case, method, 7) for method in ("ortools-off", "ortools-on", "chuffed")]
        for index, entry in enumerate(selected):
            case = {"id": entry["id"], "input": str(ROOT / entry["path"]), "sha256": entry["sha256"], "tiny": False}
            methods = ("ortools-off", "chuffed") if index % 2 == 0 else ("chuffed", "ortools-off")
            jobs.extend((case, method, 60) for method in methods)
        for case, method, seconds in jobs:
            trial = f"{case['id']}-{method}"
            state.update(phase="verify schedule", current_case=trial)
            try:
                row = saved.get(trial) or run_trial(attempt, case, method, seconds, cpu, exe)
                verify_delivery(case, row)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row = {"check": "FAIL", "error": f"{type(error).__name__}: {error}",
                       "worker_log": str(attempt / trial / "worker.log")}
                print(f"{trial}: {row['error']}", flush=True)
            done.append({**row, "id": trial})
            atomic_json(directory / "report.json", report)
            state.update(completed=len(done))
        if any(row["check"] == "FAIL" for row in done):
            reason = "one or more planned checks failed; see report.json and worker logs"
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        report["error_detail"] = traceback.format_exc()
    finally:
        state.update(state="FINALIZING", phase="snapshot and report", current_case=None)
        os.sched_setaffinity(0, allowed)
        snapshots(attempt, "after")
        report.update(passed=reason is None, reason=reason, attempt=str(attempt))
        atomic_json(directory / "report.json", report)
        temporary = directory / "brief.partial.tar.gz"
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(directory / "report.json", arcname="report.json")
            for path in sorted(attempt.glob("*-resource.json")) + sorted(attempt.glob("*-storage.json")):
                archive.add(path, arcname=path.name)
        brief = directory / "brief.tar.gz"
        os.replace(temporary, brief)
        (directory / "brief.tar.gz.sha256").write_text(glnsa_runtime.digest(brief) + "  brief.tar.gz\n", encoding="ascii", newline="\n")
        state.finish("COMPLETED" if reason is None else "PAUSED" if reason == "interrupted" else "FAILED",
                     0 if reason is None else 130 if reason == "interrupted" else 2, reason=reason)
        print(f"Reports: {directory}", flush=True)
    return state.data["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
