"""Read a chain FJSP instance and verify a schedule independently of its solver."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Instance:
    name: str
    jobs: dict[int, tuple[int, ...]]
    options: dict[int, dict[int, int]]


def _integer(value, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _array(value, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _object(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def parse_instance(data) -> Instance:
    """Parse explicit job chains or infer them from FJSPLib precedence records."""
    data = _object(data, "instance input")
    name = data.get("instance")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("instance must be a nonempty string")

    options = {}
    for row in _array(data.get("operations"), "operations"):
        row = _object(row, "machine option")
        operation = _integer(row.get("operation"), "operation")
        machine = _integer(row.get("machine"), "machine")
        duration = _integer(row.get("duration"), "duration", minimum=1)
        machines = options.setdefault(operation, {})
        if machine in machines:
            raise ValueError(f"duplicate machine {machine} for operation {operation}")
        machines[machine] = duration
    if not options:
        raise ValueError("an instance must contain at least one operation")

    if "jobs" in data:
        if "precedences" in data:
            raise ValueError("provide either jobs or precedences")
        jobs = {}
        assigned = set()
        for row in _array(data["jobs"], "jobs"):
            row = _object(row, "job")
            job = _integer(row.get("job"), "job")
            if job in jobs:
                raise ValueError(f"duplicate job {job}")
            chain = tuple(_integer(op, "operation") for op in _array(row.get("operations"), "job operations"))
            if not chain:
                raise ValueError(f"job {job} must contain an operation")
            for operation in chain:
                if operation not in options or operation in assigned:
                    raise ValueError(f"unknown or repeated operation {operation} in jobs")
                assigned.add(operation)
            jobs[job] = chain
        if assigned != options.keys():
            raise ValueError(f"missing job membership for operations {sorted(options.keys() - assigned)}")
    else:
        jobs = _jobs_from_precedences(_array(data.get("precedences"), "precedences"), options)
    return Instance(name, jobs, options)


def _jobs_from_precedences(rows, options) -> dict[int, tuple[int, ...]]:
    owners = {}
    job_nodes = defaultdict(set)
    successors = {}
    predecessors = set()
    for row in rows:
        row = _object(row, "precedence")
        before = _integer(row.get("before"), "before")
        after = _integer(row.get("after"), "after")
        job = _integer(row.get("job"), "job")
        if before in successors or after in predecessors:
            raise ValueError("precedences contain repeated edges, splits or merges")
        for operation in (before, after):
            if operation not in options:
                raise ValueError(f"unknown operation {operation} in precedences")
            if owners.setdefault(operation, job) != job:
                raise ValueError(f"operation {operation} belongs to more than one job")
            job_nodes[job].add(operation)
        successors[before] = after
        predecessors.add(after)

    if owners.keys() != options.keys():
        missing = sorted(options.keys() - owners.keys())
        raise ValueError(f"operations {missing} lack job membership in precedences; provide explicit jobs")

    jobs = {}
    for job, nodes in job_nodes.items():
        heads = nodes - predecessors
        if len(heads) != 1:
            raise ValueError(f"job {job} must form one acyclic chain")
        operation = heads.pop()
        chain = []
        # With at most one predecessor, a chain from a head cannot enter a cycle.
        while operation is not None:
            chain.append(operation)
            operation = successors.get(operation)
        if len(chain) != len(nodes):
            raise ValueError(f"job {job} must form one acyclic chain")
        jobs[job] = tuple(chain)
    return jobs


def validate_schedule(instance: Instance, schedule) -> int:
    """Return the independently computed makespan, or raise ValueError."""
    schedule = _object(schedule, "schedule")
    if schedule.get("instance") != instance.name:
        raise ValueError(f"schedule instance must match {instance.name!r}")
    assignments = {}
    by_machine = defaultdict(list)
    for row in _array(schedule.get("assignments"), "assignments"):
        row = _object(row, "assignment")
        operation = _integer(row.get("operation"), "operation")
        if operation not in instance.options:
            raise ValueError(f"unknown operation {operation}")
        if operation in assignments:
            raise ValueError(f"duplicate assignment for operation {operation}")
        machine = _integer(row.get("machine"), "machine")
        if machine not in instance.options[operation]:
            raise ValueError(f"machine {machine} is not eligible for operation {operation}")
        start = _integer(row.get("start"), "start")
        end = _integer(row.get("end"), "end")
        if end <= start:
            raise ValueError(f"end must be after start for operation {operation}")
        if end - start != instance.options[operation][machine]:
            raise ValueError(f"incorrect duration for operation {operation} on machine {machine}")
        assignments[operation] = (start, end)
        by_machine[machine].append((start, end, operation))

    missing = instance.options.keys() - assignments.keys()
    if missing:
        raise ValueError(f"missing assignments for operations {sorted(missing)}")

    for job, chain in instance.jobs.items():
        for before, after in zip(chain, chain[1:]):
            if assignments[before][1] > assignments[after][0]:
                raise ValueError(f"job {job} precedence violated: operation {before} before {after}")
    for machine, intervals in by_machine.items():
        intervals.sort()
        for (_, previous_end, previous), (start, _, current) in zip(intervals, intervals[1:]):
            if previous_end > start:
                raise ValueError(f"machine {machine} overlap: operations {previous} and {current}")

    makespan = max(end for _, end in assignments.values())
    if "makespan" in schedule and _integer(schedule["makespan"], "makespan") != makespan:
        raise ValueError(f"reported makespan {schedule['makespan']} does not match computed {makespan}")
    return makespan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    try:
        instance = parse_instance(json.loads(args.instance.read_text(encoding="utf-8")))
        schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
        makespan = validate_schedule(instance, schedule)
    except (OSError, ValueError) as error:
        print(f"Invalid: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"instance": instance.name, "valid": True, "makespan": makespan}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
