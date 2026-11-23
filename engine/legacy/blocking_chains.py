"""Grow a release set along a bounded chain of machine blockers."""

from collections import defaultdict
from dataclasses import dataclass

from fjsp import Instance, validate_schedule


@dataclass(frozen=True)
class BlockerLink:
    operation: int
    blocker: int
    target_machine: int
    overlap: int


def grow_blocking_chain(
    instance: Instance, initial: dict, anchor: int, *, max_size: int
) -> tuple[BlockerLink, ...]:
    """Follow the highest-overlap unvisited flexible blocker in the initial schedule."""
    if type(max_size) is not int or max_size < 1 or anchor not in instance.options:
        raise ValueError("use a known anchor and a positive group size")
    validate_schedule(instance, initial)
    rows = {row["operation"]: row for row in initial["assignments"]}
    queues = defaultdict(list)
    for row in rows.values():
        queues[row["machine"]].append(row)

    seen = {anchor}
    links = []
    current = anchor
    while len(seen) < max_size:
        start = rows[current]["start"]
        candidates = []
        for machine, duration in instance.options[current].items():
            if machine == rows[current]["machine"]:
                continue
            for row in queues.get(machine, ()):
                op = row["operation"]
                if op in seen or len(instance.options[op]) < 2:
                    continue
                overlap = min(start + duration, row["end"]) - max(start, row["start"])
                if overlap > 0:
                    candidates.append((-overlap, machine, op))
        if not candidates:
            break
        negative_overlap, machine, blocker = min(candidates)
        links.append(BlockerLink(current, blocker, machine, -negative_overlap))
        seen.add(blocker)
        current = blocker
    return tuple(links)
