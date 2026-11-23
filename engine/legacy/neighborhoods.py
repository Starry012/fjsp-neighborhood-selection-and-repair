"""Paired neighborhoods for a pilot of two-operation machine repair."""

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from fjsp import Instance, validate_schedule
from move_scan import _critical_operations


@dataclass(frozen=True)
class ReleasePair:
    anchor: int
    blocker: int
    random_partner: int
    target_machine: int
    overlap: int


@dataclass(frozen=True)
class PairSelection:
    pairs: tuple[ReleasePair, ...]
    flexible_critical: int
    no_blocker: int
    no_control: int
    eligible: int


def select_release_pairs(instance: Instance, initial: dict, *, max_pairs: int, seed: int) -> PairSelection:
    """Pair a flexible critical operation with a blocker and a random non-blocker.

    A blocker is flexible and overlaps the anchor's hypothetical interval on an
    alternative machine, using the anchor's original start time. The interval
    only scores a pairing; the repair solver can choose other machines or times.
    """
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    makespan = validate_schedule(instance, initial)
    rows = {row["operation"]: row for row in initial["assignments"]}
    queues = defaultdict(list)
    for op in sorted(rows, key=lambda op: rows[op]["start"]):
        queues[rows[op]["machine"]].append(op)
    critical = _critical_operations(instance, rows, queues, makespan)
    flexible = {op for op, machines in instance.options.items() if len(machines) > 1}
    anchors = sorted(critical & flexible)
    no_blocker = no_control = 0
    eligible = []
    for anchor in anchors:
        start = rows[anchor]["start"]
        blockers = []
        for machine, duration in instance.options[anchor].items():
            if machine == rows[anchor]["machine"]:
                continue
            for op in queues.get(machine, ()):
                row = rows[op]
                overlap = min(start + duration, row["end"]) - max(start, row["start"])
                if overlap > 0 and op in flexible:
                    blockers.append((-overlap, machine, op))
        if not blockers:
            no_blocker += 1
            continue
        controls = sorted(flexible - {anchor} - {op for _, _, op in blockers})
        if not controls:
            no_control += 1
            continue
        neg_overlap, machine, blocker = min(blockers)
        eligible.append((anchor, blocker, machine, -neg_overlap, controls))

    rng = Random(f"{seed}:{instance.name}")
    sampled = sorted(rng.sample(eligible, max_pairs)) if len(eligible) > max_pairs else eligible
    pairs = tuple(ReleasePair(a, b, rng.choice(controls), m, overlap) for a, b, m, overlap, controls in sampled)
    return PairSelection(pairs, len(anchors), no_blocker, no_control, len(eligible))
