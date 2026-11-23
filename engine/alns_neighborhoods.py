"""Three current-schedule constraint-extraction operators from ALNS-CP."""

from collections import deque
from hashlib import sha256
import json
from math import ceil, isfinite

from fjsp import validate_schedule


OPERATORS = ("time-window", "parallel-block", "machine-removal")
METHODS = (*OPERATORS, "random-operations", "uniform-mixture", "full-cp")


def anchor_hash(schedule):
    value = {**schedule, "assignments": sorted(schedule["assignments"], key=lambda r: r["operation"])}
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_neighborhood(instance, schedule, method, rho, rng):
    """Use inclusive time intersections, whole expansion batches and ceil sizes.

    A block stops when its queue empties, even below target. Machine sampling
    includes every eligible machine, including currently unused machines; no
    resampling or truncation changes the paper's whole-machine rule.
    """
    if method not in METHODS or not isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("unknown method or invalid neighborhood fraction")
    makespan = validate_schedule(instance, schedule)
    rows = {r["operation"]: r for r in schedule["assignments"]}
    operations = sorted(instance.options)
    kind = rng.choice(OPERATORS) if method == "uniform-mixture" else method
    target = max(1, ceil(rho * len(operations)))
    details = {"method": method, "operator": kind, "rho": rho, "target_operations": target}
    if kind == "full-cp":
        released = operations
    elif kind == "random-operations":
        released = rng.sample(operations, target)
    elif kind == "time-window":
        width = rho * makespan
        left = rng.uniform(0, makespan - width)
        right = left + width
        released = [op for op in operations if rows[op]["start"] <= right and rows[op]["end"] >= left]
        details.update(window_start=left, window_end=right, window_width=width)
    elif kind == "parallel-block":
        seed = rng.choice(operations)
        selected, queue = {seed}, deque([seed])
        while len(selected) < target and queue:
            row = rows[queue.popleft()]
            neighbors = [op for op in operations if op not in selected and
                         rows[op]["start"] <= row["end"] and rows[op]["end"] >= row["start"]]
            rng.shuffle(neighbors)
            selected.update(neighbors)
            queue.extend(neighbors)
        released = selected
        details.update(seed_operation=seed, queue_exhausted=not queue,
                       target_reached=len(selected) >= target, overshoot=max(0, len(selected) - target))
    else:
        machines = sorted({m for options in instance.options.values() for m in options})
        count = max(1, ceil(rho * len(machines)))
        chosen = sorted(rng.sample(machines, count))
        released = [op for op in operations if rows[op]["machine"] in chosen]
        details.update(selected_machines=chosen, machine_count=len(machines), target_machines=count)
    released = sorted(released)
    details.update(released=released, actual_operations=len(released), empty=not released)
    return details
