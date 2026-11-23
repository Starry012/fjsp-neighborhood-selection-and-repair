"""Evaluate one specified move to another machine without choosing or applying it."""

from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

from fjsp import Instance, validate_schedule


@dataclass(frozen=True)
class RelocationEvaluation:
    candidate: dict
    original_makespan: int

    @property
    def makespan_change(self) -> int:
        """Candidate makespan minus original makespan; negative means improvement."""
        return self.candidate["makespan"] - self.original_makespan


def relocate_order(
    instance: Instance, schedule: dict, operation: int, target_machine: int, position: int
) -> dict[int, tuple[int, ...]] | None:
    """Return candidate machine queues, or None if job and machine orders conflict.

    Position 0 inserts before the first operation; len(queue) appends.
    Invalid input raises ValueError. The candidate has no start or end times yet.
    """
    validate_schedule(instance, schedule)
    queues = _moved_queues(instance, schedule, operation, target_machine, position)
    try:
        TopologicalSorter(_predecessors(instance, queues)).prepare()
    except CycleError:
        return None
    return queues


def evaluate_relocation(
    instance: Instance, schedule: dict, operation: int, target_machine: int, position: int
) -> RelocationEvaluation | None:
    """Build and verify the earliest schedule for the specified candidate queues.

    Return None for an order conflict. A feasible candidate is returned even if
    it ties or worsens the objective, so the caller can make the policy decision.
    """
    original_makespan = validate_schedule(instance, schedule)
    queues = _moved_queues(instance, schedule, operation, target_machine, position)
    predecessors = _predecessors(instance, queues)
    try:
        operations = tuple(TopologicalSorter(predecessors).static_order())
    except CycleError:
        return None

    machine_of = {op: machine for machine, queue in queues.items() for op in queue}
    assignments = []
    ends = {}
    for op in operations:
        machine = machine_of[op]
        start = max((ends[before] for before in predecessors[op]), default=0)
        end = start + instance.options[op][machine]
        ends[op] = end
        assignments.append({"operation": op, "machine": machine, "start": start, "end": end})

    candidate = {"instance": instance.name, "assignments": assignments}
    candidate["makespan"] = validate_schedule(instance, candidate)
    return RelocationEvaluation(candidate, original_makespan)


def _moved_queues(
    instance: Instance, schedule: dict, operation: int, target_machine: int, position: int
) -> dict[int, tuple[int, ...]]:
    if type(operation) is not int or operation not in instance.options:
        raise ValueError(f"unknown operation {operation}")
    if type(target_machine) is not int or target_machine not in instance.options[operation]:
        raise ValueError(f"machine {target_machine} is not eligible for operation {operation}")

    queues = defaultdict(list)
    for row in sorted(schedule["assignments"], key=lambda row: row["start"]):
        if row["operation"] == operation:
            if row["machine"] == target_machine:
                raise ValueError("target machine must differ from the current machine")
        else:
            queues[row["machine"]].append(row["operation"])

    target = queues[target_machine]
    if type(position) is not int or not 0 <= position <= len(target):
        raise ValueError(f"position must be an integer between 0 and {len(target)}")
    target.insert(position, operation)
    return {machine: tuple(queue) for machine, queue in queues.items()}


def _predecessors(instance: Instance, queues: dict[int, tuple[int, ...]]) -> dict[int, set[int]]:
    predecessors = {op: set() for op in instance.options}
    # Both kinds of edges mean that the predecessor must finish first.
    for sequence in (*instance.jobs.values(), *queues.values()):
        for before, after in zip(sequence, sequence[1:]):
            predecessors[after].add(before)
    return predecessors
