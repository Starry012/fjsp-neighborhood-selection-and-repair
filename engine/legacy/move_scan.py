"""Scan one pool of machine moves from a fixed initial schedule."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from time import perf_counter

from fjsp import Instance, validate_schedule
from relocate import evaluate_relocation


Move = tuple[int, int, int]  # operation, target machine, insertion position


@dataclass(frozen=True)
class Trial:
    move: Move
    candidate_makespan: int | None
    best_makespan: int
    scan_seconds: float


@dataclass(frozen=True)
class ScanResult:
    schedule: dict
    trials: tuple[Trial, ...]
    pool_size: int
    critical_operations: int | None
    stop_reason: str


def _critical_operations(instance, rows, queues, makespan):
    """Follow all tight job and machine arcs backwards from Cmax."""
    tight = defaultdict(set)
    for sequence in (*instance.jobs.values(), *queues.values()):
        for before, after in zip(sequence, sequence[1:]):
            if rows[before]["end"] == rows[after]["start"]:
                tight[after].add(before)
    stack = [op for op, row in rows.items() if row["end"] == makespan]
    critical = set(stack)
    while stack:
        for before in tight[stack.pop()]:
            if before not in critical:
                stack.append(before)
                critical.add(before)
    return critical


def scan_relocations(
    instance: Instance, initial: dict, *, order: str, attempt_limit: int,
    deadline: float = float("inf"), result_deadline: float = float("inf")
) -> ScanResult:
    """Compare moves from the same initial schedule, retaining the best seen.

    The fixed order is lexicographic. The critical order moves all candidates
    for critical operations to the front, preserving the other sort keys.
    Infeasible candidates count as attempts. No candidate becomes a new base.
    With a result deadline, a move finishing after it is recorded but cannot improve the incumbent.
    """
    started = perf_counter()
    if order not in {"fixed", "critical-first"}:
        raise ValueError(f"unknown candidate order {order}")
    if type(attempt_limit) is not int or attempt_limit < 1:
        raise ValueError("attempt_limit must be a positive integer")
    best_makespan = validate_schedule(instance, initial)
    best_schedule = initial
    rows = {row["operation"]: row for row in initial["assignments"]}
    queues = defaultdict(list)
    for op in sorted(rows, key=lambda op: rows[op]["start"]):
        queues[rows[op]["machine"]].append(op)
    critical = _critical_operations(instance, rows, queues, best_makespan) if order == "critical-first" else set()
    operations = sorted(rows, key=lambda op: (op not in critical, op))
    moves = [
        (op, machine, position)
        for op in operations
        for machine in sorted(instance.options[op])
        if machine != rows[op]["machine"]
        for position in range(len(queues.get(machine, ())) + 1)
    ]

    trials = []
    stop_reason = "POOL_EXHAUSTED" if len(moves) <= attempt_limit else "ATTEMPT_LIMIT"
    for move in islice(moves, attempt_limit):
        if perf_counter() >= deadline:
            stop_reason = "TIME_LIMIT"
            break
        evaluation = evaluate_relocation(instance, initial, *move)
        candidate_makespan = None if evaluation is None else evaluation.candidate["makespan"]
        observed_at = perf_counter()
        if observed_at <= result_deadline and candidate_makespan is not None and candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_schedule = evaluation.candidate
        trials.append(Trial(move, candidate_makespan, best_makespan, observed_at - started))
        if observed_at > result_deadline:
            stop_reason = "TIME_LIMIT"
            break
    return ScanResult(
        best_schedule, tuple(trials), len(moves), len(critical) if order == "critical-first" else None, stop_reason
    )
