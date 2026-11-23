"""Local, four-instance Chuffed delivery probe on already exposed FJSP inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile
from time import perf_counter

from baseline import solve_ect
from delivery_guard import atomic_json
from experiments.copt_formulations import time_windows
from experiments.v4_delivery_worker import direct_environment
from fjsp import parse_instance, validate_schedule


LOCAL_BENCHMARKS = Path("data/raw/benchmarks-main/flexible-jobshop/instances/json")
ALNS_INPUTS = Path("outputs/alns-online-release-20261117/fjsp-alns-online-20261117/inputs")
WARMSTART = "fjsp-stall80-warmstart-20261120"
WARMSTART_ARCHIVE = Path("outputs/stall80-warmstart-release-20261120") / f"{WARMSTART}.tar.gz"
MODEL = Path(__file__).with_name("chuffed_fjsp.mzn")
TIGHT_MODEL = Path(__file__).with_name("chuffed_fjsp_tight.mzn")
METHODS = ("warm", "free", "tight", "structured")
HINT_ANNOTATION = "seq_search([warm_start(present, ect_present), warm_start(starts, ect_start)])"
COMPLETE_ANNOTATION = """seq_search([warm_start(present, ect_present), warm_start(starts, ect_start),
    bool_search(present, input_order, indomain_max, complete),
    int_search(starts, input_order, indomain_min, complete)])"""
CASES = {
    "behnke12": (LOCAL_BENCHMARKS, "3be957bbd4ec34c8d63627bc617bfc86d97c43c1ff6963644ae05d0084c657cb"),
    "sd2mix-validation-25x10-0000": (ALNS_INPUTS, "cb4f3e021a4bf04a9e6b7a2f792bf1ff94a39f5af205b3edcb03ad8329bce656"),
    "behnke59": (LOCAL_BENCHMARKS, "abbfa6a16beac4a6124be91828d6d32c922c7fa313a3bf12bce082646041d435"),
    "sd2mix-validation-50x10-44000": (WARMSTART_ARCHIVE, "0faf28fc2f7ee63ad28111281ffec5d62a58422ad57e26eb75a89ee2ddf367e5"),
}


def case_bytes(name):
    source, expected = CASES[name]
    if source == WARMSTART_ARCHIVE:
        with tarfile.open(source) as archive:
            with archive.extractfile(f"{WARMSTART}/inputs/{name}.json") as stream:
                contents = stream.read()
    else:
        contents = (source / f"{name}.json").read_bytes()
    if sha256(contents).hexdigest() != expected:
        raise ValueError(f"input hash differs for {name}")
    return contents


def load_case(name):
    return parse_instance(json.loads(case_bytes(name)))


def model_input(instance, initial):
    operations = sorted(instance.options)
    modes = [(op, machine, duration) for op in operations
             for machine, duration in sorted(instance.options[op].items())]
    machine_number = {m: i for i, m in enumerate(sorted({m for _, m, _ in modes}), 1)}
    operation_number = {op: i for i, op in enumerate(operations, 1)}
    arcs = [(a, b) for chain in instance.jobs.values() for a, b in zip(chain, chain[1:])]
    assigned = {row["operation"]: row for row in initial["assignments"]}
    data = {
        "operations": len(operations), "machines": len(machine_number), "modes": len(modes),
        "horizon": validate_schedule(instance, initial), "precedences": len(arcs),
        "mode_operation": [operation_number[op] for op, _, _ in modes],
        "mode_machine": [machine_number[machine] for _, machine, _ in modes],
        "mode_duration": [duration for _, _, duration in modes],
        "predecessor": [operation_number[a] for a, _ in arcs],
        "successor": [operation_number[b] for _, b in arcs],
        "ect_start": [assigned[op]["start"] for op in operations],
        "ect_present": [assigned[op]["machine"] == machine for op, machine, _ in modes],
    }
    return data, modes, operation_number


def decode(instance, output, modes, operation_number):
    if len(output["starts"]) != len(operation_number):
        raise ValueError("Chuffed output start array differs from operation count")
    assignments = []
    for choice in output["modes"]:
        if type(choice) is not int or not 1 <= choice <= len(modes):
            raise ValueError("Chuffed output mode index out of range")
        op, machine, duration = modes[choice - 1]
        start = output["starts"][operation_number[op] - 1]
        assignments.append({"operation": op, "machine": machine, "start": start, "end": start + duration})
    schedule = {"instance": instance.name, "assignments": assignments, "makespan": output["makespan"]}
    validate_schedule(instance, schedule)
    return schedule


def tighten_input(data, instance, operation_number, structured):
    horizon = data["horizon"]
    windows = time_windows(instance, horizon)
    operations = sorted(operation_number)
    minimum = {o: min(instance.options[o].values()) for o in operations}
    work_lb = (sum(minimum.values()) + data["machines"] - 1) // data["machines"]
    job_lb = max(sum(minimum[o] for o in chain) for chain in instance.jobs.values())
    data.update(earliest=[windows[o]["start_lb"] for o in operations],
                latest=[windows[o]["start_ub"] for o in operations],
                tail=[horizon - windows[o]["end_ub"] for o in operations],
                job_last=[operation_number[chain[-1]] for chain in instance.jobs.values()],
                objective_lb=max(work_lb, job_lb), structured_search=structured)


def trial(executable, name, directory, *, seconds=60, method="warm", completed_model=None):
    directory.mkdir()
    deadline = perf_counter() + seconds
    instance = load_case(name)
    initial = solve_ect(instance, deadline).schedule
    data, modes, operation_number = model_input(instance, initial)
    model = MODEL
    if method == "free":
        model = completed_model
    if method in ("tight", "structured"):
        model = TIGHT_MODEL
        tighten_input(data, instance, operation_number, method == "structured")
    atomic_json(directory / "initial.json", initial)
    delivery = directory / "delivery.json"
    atomic_json(delivery, initial)
    data_path = directory / "model_input.json"
    atomic_json(data_path, data)
    remaining_ms = int((deadline - perf_counter()) * 1000 - min(3500, seconds * 180))
    if remaining_ms <= 0:
        raise TimeoutError("no time after input and ECT")
    command = [str(executable), "--solver", "org.chuffed.chuffed", "--json-stream",
               "--intermediate-solutions", "--statistics", "--random-seed", "42",
               "--time-limit", str(remaining_ms), "--cleanup-time-limit", "300"]
    if method != "warm":
        command.append("--free-search")
    command.extend((str(model), str(data_path)))
    stdout_path = directory / "solver.jsonl"
    with stdout_path.open("wb") as stdout, (directory / "solver.stderr").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            exit_code = process.wait(timeout=max(0, deadline - perf_counter() - 0.35))
        except subprocess.TimeoutExpired:
            # MiniZinc owns a descendant on Windows. Terminate this trial's tree only.
            killed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    capture_output=True, check=False, timeout=5)
            if killed.returncode != 0 and process.poll() is None:
                process.kill()
            exit_code = process.wait(timeout=5)
    solver_end = perf_counter()
    best, valid, status, statistics = initial, 0, None, {}
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] == "solution":
            schedule = decode(instance, json.loads(event["output"]["default"]), modes, operation_number)
            valid += 1
            if schedule["makespan"] < best["makespan"]:
                best = schedule
        if event["type"] == "status":
            status = event["status"]
        if event["type"] == "statistics":
            statistics.update(event["statistics"])
    if perf_counter() + 0.1 < deadline:
        atomic_json(delivery, best)
    done = perf_counter()
    return {"instance": name, "method": method, "operations": data["operations"], "seed": 42, "initial": initial["makespan"],
            "best_verified": best["makespan"], "delivered": json.loads(delivery.read_bytes())["makespan"],
            "solver_status": status, "exit_code": exit_code, "validated_solver_frames": valid,
            "solver_return_seconds": solver_end - deadline + seconds, "total_seconds": done - deadline + seconds,
            "on_time": done <= deadline, "statistics": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("minizinc", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=60, choices=range(3, 61), metavar="3..60")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["warm"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = direct_environment()
    executable = args.minizinc.resolve(strict=True)
    models = [MODEL, TIGHT_MODEL]
    completed_model = None
    if "free" in args.methods:
        completed_model = args.output / "complete_search.mzn"
        original_model = MODEL.read_text(encoding="utf-8")
        if original_model.count(HINT_ANNOTATION) != 1:
            raise ValueError("cannot locate frozen warm-start annotation")
        completed_model.write_text(original_model.replace(HINT_ANNOTATION, COMPLETE_ANNOTATION), encoding="utf-8")
        models.append(completed_model)
    plan = {"environment": environment, "cases": args.cases, "budget_seconds": args.seconds, "methods": args.methods,
            "seed": 42, "models": {p.name: sha256(p.read_bytes()).hexdigest() for p in models}}
    atomic_json(args.output / "plan.json", plan)
    rows = []
    for i, name in enumerate(args.cases):
        offset = i % len(args.methods)
        for method in args.methods[offset:] + args.methods[:offset]:
            directory = args.output / (name if len(args.methods) == 1 else f"{name}-{method}")
            try:
                row = trial(executable, name, directory, seconds=args.seconds, method=method,
                            completed_model=completed_model)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                row = {"instance": name, "method": method, "error": f"{type(error).__name__}: {error}"}
            atomic_json(directory / "result.json", row)
            rows.append(row)
            atomic_json(args.output / "summary.json", rows)
            print(json.dumps({k: v for k, v in row.items() if k != "statistics"}), flush=True)


if __name__ == "__main__":
    main()
