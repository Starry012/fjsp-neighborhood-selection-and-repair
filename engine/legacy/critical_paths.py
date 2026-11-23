"""Find frozen critical-path witnesses and minimum-cardinality release covers."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from fjsp import Instance, validate_schedule


def _known_operations(instance, operations, name):
    operations = tuple(operations)
    if any(type(op) is not int or op not in instance.options for op in operations):
        raise ValueError(f"{name} operations must be known integer operation IDs")
    return frozenset(operations)


def _tight_graph(instance, schedule):
    makespan = validate_schedule(instance, schedule)
    rows = {row["operation"]: row for row in schedule["assignments"]}
    queues = defaultdict(list)
    order = sorted(rows, key=lambda op: (rows[op]["start"], op))
    for op in order:
        queues[rows[op]["machine"]].append(op)
    predecessors = defaultdict(set)
    for chain in (*instance.jobs.values(), *queues.values()):
        for before, after in zip(chain, chain[1:]):
            if rows[before]["end"] == rows[after]["start"]:
                predecessors[after].add(before)

    return makespan, rows, order, predecessors


def _surviving_path(graph, released):
    makespan, rows, order, predecessors = graph
    paths = {}
    for op in order:
        if op in released:
            continue
        if rows[op]["start"] == 0:
            paths[op] = (op,)
        else:
            parent = next((p for p in sorted(predecessors[op]) if p in paths), None)
            if parent is not None:
                paths[op] = (*paths[parent], op)
        if rows[op]["end"] == makespan and op in paths:
            return paths[op]
    return None


def frozen_critical_path(
    instance: Instance, schedule: dict, released: Iterable[int]
) -> tuple[int, ...] | None:
    """Return one frozen tight path of length Cmax, or None if none remains.

    All operations on a returned path keep their assigned processing times.
    Its job or machine-order edges are retained by the existing repair rules.
    Therefore its processing-time sum is a neighborhood lower bound of Cmax.
    None only means this witness cannot rule out strict improvement; it does
    not establish that the neighborhood contains a better schedule.
    """
    graph = _tight_graph(instance, schedule)
    return _surviving_path(graph, _known_operations(instance, released, "released"))


@dataclass(frozen=True)
class CriticalCover:
    status: str
    released: tuple[int, ...]
    minimum_size: int | None
    witness: tuple[int, ...] | None = None


def minimum_critical_cover(
    instance: Instance, schedule: dict, *, allowed: Iterable[int] | None = None
) -> CriticalCover:
    """Cover every original zero-to-Cmax tight path using fewest allowed ops.

    Return one minimum cover; no preference among tied covers is asserted.
    NO_ALLOWED_COVER concerns this auxiliary selection problem, not FJSP
    feasibility. NO_CRITICAL_PATH means the empty set already covers all
    original paths. Neither that status nor a cover implies repair gains.
    """
    graph = _tight_graph(instance, schedule)
    allowed = _known_operations(instance, instance.options if allowed is None else allowed, "allowed")
    original = _surviving_path(graph, frozenset())
    if original is None:
        return CriticalCover("NO_CRITICAL_PATH", (), 0)

    # OR-Tools is needed only for construction; witness checking is standalone.
    from ortools.graph.python import max_flow

    makespan, rows, _, predecessors = graph
    positions = {op: i for i, op in enumerate(sorted(rows))}
    source, sink = 2 * len(rows), 2 * len(rows) + 1
    forbidden = len(rows) + 1  # More than the cost of releasing every operation.
    flow = max_flow.SimpleMaxFlow()
    for op, index in positions.items():
        inlet, outlet = 2 * index, 2 * index + 1
        flow.add_arc_with_capacity(inlet, outlet, 1 if op in allowed else forbidden)
        if rows[op]["start"] == 0:
            flow.add_arc_with_capacity(source, inlet, forbidden)
        if rows[op]["end"] == makespan:
            flow.add_arc_with_capacity(outlet, sink, forbidden)
        for before in sorted(predecessors[op]):
            flow.add_arc_with_capacity(2 * positions[before] + 1, inlet, forbidden)
    if flow.solve(source, sink) != flow.OPTIMAL:
        raise RuntimeError("minimum critical-cover max-flow solver failed")
    size = flow.optimal_flow()
    if size >= forbidden:
        witness = _surviving_path(graph, allowed)
        if witness is None:
            raise RuntimeError("uncuttable result lacks a fully protected path")
        return CriticalCover("NO_ALLOWED_COVER", (), None, witness)

    reachable = set(flow.get_source_side_min_cut())
    released = tuple(op for op, i in positions.items() if 2 * i in reachable and 2 * i + 1 not in reachable)
    if len(released) != size or not set(released) <= allowed or _surviving_path(graph, released) is not None:
        raise RuntimeError("minimum cut does not define a valid release cover")
    return CriticalCover("COVER", released, size)
