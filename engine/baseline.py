"""Run and independently verify the two V1 scheduling baselines."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

from ortools.sat.python import cp_model

from fjsp import Instance, parse_instance, validate_schedule


@dataclass
class SolveResult:
    status: str
    schedule: dict | None = None
    lower_bound: float | None = None
    search_seconds: float | None = None
    detail: str | None = None


def solve_cp_sat(instance: Instance, deadline: float, *, workers: int, seed: int) -> SolveResult:
    model = cp_model.CpModel()
    # A serial schedule is feasible within this horizon for any machine choices.
    horizon = sum(max(options.values()) for options in instance.options.values())
    starts = {}
    ends = {}
    choices = {}
    by_machine = defaultdict(list)
    for operation, options in instance.options.items():
        start = starts[operation] = model.new_int_var(0, horizon, f"start_{operation}")
        end = ends[operation] = model.new_int_var(0, horizon, f"end_{operation}")
        choices[operation] = {}
        for machine, duration in options.items():
            selected = model.new_bool_var(f"use_{operation}_{machine}")
            choices[operation][machine] = selected
            interval = model.new_optional_interval_var(
                start, duration, end, selected, f"interval_{operation}_{machine}"
            )
            by_machine[machine].append(interval)
        model.add_exactly_one(choices[operation].values())

    for chain in instance.jobs.values():
        for before, after in zip(chain, chain[1:]):
            model.add(ends[before] <= starts[after])
    for intervals in by_machine.values():
        model.add_no_overlap(intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends.values())
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return SolveResult("TIME_LIMIT", detail="budget exhausted while building the model")
    solver.parameters.max_time_in_seconds = remaining
    status = solver.solve(model)
    result = SolveResult(solver.status_name(status), search_seconds=solver.wall_time)
    if status == cp_model.MODEL_INVALID:
        result.detail = solver.response_proto.solution_info
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result.lower_bound = solver.best_objective_bound
        result.schedule = {
            "instance": instance.name,
            "assignments": [
                {"operation": op, "machine": machine, "start": solver.value(starts[op]),
                 "end": solver.value(ends[op])}
                for op, machines in choices.items()
                for machine, selected in machines.items()
                if solver.boolean_value(selected)
            ],
            "makespan": solver.value(makespan),
        }
    elif status == cp_model.UNKNOWN and solver.best_objective_bound > 0:
        result.lower_bound = solver.best_objective_bound
    return result


def solve_ect(instance: Instance, deadline: float) -> SolveResult:
    """Append the earliest finishing operation and machine pair at each step."""
    positions = dict.fromkeys(instance.jobs, 0)
    job_end = dict.fromkeys(instance.jobs, 0)
    machine_end = defaultdict(int)
    assignments = []
    for _ in instance.options:
        if perf_counter() >= deadline:
            return SolveResult("TIME_LIMIT", detail="budget exhausted during construction")
        candidates = []
        for job, chain in instance.jobs.items():
            if positions[job] == len(chain):
                continue
            operation = chain[positions[job]]
            for machine, duration in instance.options[operation].items():
                start = max(job_end[job], machine_end[machine])
                candidates.append((start + duration, job, machine, start))
        end, job, machine, start = min(candidates)
        operation = instance.jobs[job][positions[job]]
        assignments.append({"operation": operation, "machine": machine, "start": start, "end": end})
        positions[job] += 1
        job_end[job] = machine_end[machine] = end
    schedule = {"instance": instance.name, "assignments": assignments, "makespan": max(job_end.values())}
    return SolveResult("FEASIBLE", schedule)


def load_pinned_json(path: Path, expected_sha256: str):
    contents = path.read_bytes()
    if sha256(contents).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return json.loads(contents)


def run_case(method, solve, instance, group, reference, budget, output) -> dict:
    started = perf_counter()
    try:
        result = solve(instance, started + budget)
    except Exception as error:
        # Record one failed run and continue with the rest of the experiment.
        result = SolveResult("ERROR", detail=f"{type(error).__name__}: {error}")
    elapsed = perf_counter() - started
    row = {
        "instance": instance.name, "group": group, "method": method, "status": result.status,
        "seconds": elapsed, "search_seconds": result.search_seconds,
        "budget_exceeded": elapsed > budget or result.status == "TIME_LIMIT",
        "lower_bound": result.lower_bound, "detail": result.detail,
        "schedule_file": None, "validation": "NOT_RUN", "validation_error": None,
        "verified_makespan": None, "gap_to_solver_bound": None, "deviation_from_reference_upper": None,
        "reference": reference,
    }
    if result.schedule is not None:
        file_name = f"{instance.name}.{method}.schedule.json"
        (output / file_name).write_text(json.dumps(result.schedule, indent=2) + "\n", encoding="utf-8")
        row["schedule_file"] = file_name
        try:
            row["verified_makespan"] = validate_schedule(instance, result.schedule)
            row["validation"] = "PASS"
        except ValueError as error:
            row["validation"] = "FAIL"
            row["validation_error"] = str(error)
    row["usable"] = row["validation"] == "PASS" and not row["budget_exceeded"]
    if row["usable"]:
        upper = row["verified_makespan"]
        if result.lower_bound is not None:
            row["gap_to_solver_bound"] = (upper - result.lower_bound) / upper
        if reference is not None:
            reference_upper = reference["upper_bound"]
            row["deviation_from_reference_upper"] = (upper - reference_upper) / reference_upper
    return row


def run_experiment(config_path: Path, output: Path, pilot: bool) -> int:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    requirements = Path("requirements.txt")
    packages = dict(line.split("==") for line in requirements.read_text().splitlines())
    for name, expected in packages.items():
        if version(name) != expected:
            raise ValueError(f"{name} must be {expected} for this protocol")
    if platform.python_version() != config["python"]:
        raise ValueError(f"Python must be {config['python']} for this protocol")

    methods = {
        "cp-sat": partial(solve_cp_sat, workers=config["workers"], seed=config["seed"]),
        "ect": solve_ect,
    }
    if pilot:
        case = config["pilot"]
        instance = parse_instance(load_pinned_json(Path(case["path"]), case["sha256"]))
        cases = [(instance, "hand-check", None)]
    else:
        reference_source = config["reference"]
        reference_records = load_pinned_json(Path(reference_source["path"]), reference_source["sha256"])
        by_name = {record["instance"]: record for record in reference_records}
        cases = []
        for case in config["cases"]:
            name = case["instance"]
            instance_path = Path(config["instance_dir"]) / f"{name}.json"
            instance = parse_instance(load_pinned_json(instance_path, case["sha256"]))
            if instance.name != name:
                raise ValueError(f"instance name mismatch: {instance_path}")
            reference = {key: by_name[name][key] for key in ("lower_bound", "upper_bound", "status")}
            cases.append((instance, case["group"], reference))

    output.mkdir(parents=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "pilot" if pilot else "formal",
        "config": config, "config_sha256": sha256(config_bytes).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        "packages": packages},
        "code_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("fjsp.py"), requirements)
        },
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as log:
        for instance, group, reference in cases:
            for method in config["methods"]:
                row = run_case(method, methods[method], instance, group, reference,
                               config["budget_seconds"], output)
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows.append(row)
                print(f"{instance.name} {method}: {row['status']}, "
                      f"check={row['validation']}, Cmax={row['verified_makespan']}")

    if pilot:
        cp_sat = next(row for row in rows if row["method"] == "cp-sat")
        expected = config["pilot"]["expected_makespan"]
        return int(not (cp_sat["status"] == "OPTIMAL"
                        and cp_sat["lower_bound"] == cp_sat["verified_makespan"] == expected
                        and all(row["usable"] for row in rows)))
    return int(any(row["status"] in {"ERROR", "MODEL_INVALID"} or row["validation"] == "FAIL" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for this run")
    parser.add_argument("--config", type=Path, default=Path("experiments/v1_baseline.json"))
    parser.add_argument("--pilot", action="store_true", help="run only the hand-check example")
    args = parser.parse_args()
    try:
        return run_experiment(args.config, args.output, args.pilot)
    except (OSError, ValueError) as error:
        print(f"Experiment setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
