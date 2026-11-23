"""Lower-bound witnesses and exhaustive one-operation extensions of a witness."""

from collections import defaultdict
from dataclasses import dataclass

from fjsp import validate_schedule
from repair import _frozen_queues


@dataclass(frozen=True)
class PathWitness:
    lower_bound: int
    operations: tuple[int, ...]
    durations: tuple[int, ...]
    relations: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class WitnessExtension:
    operation: int
    released: tuple[int, ...]
    shorter_duration: bool
    removable_machine_relations: tuple[tuple[int, int], ...]
    bound_after: PathWitness


@dataclass(frozen=True)
class ExtensionPool:
    status: str
    target: int
    released: tuple[int, ...]
    witness: PathWitness
    extensions: tuple[WitnessExtension, ...]


def _release_set(instance, released):
    released = tuple(released)
    if any(type(o) is not int or o not in instance.options for o in released):
        raise ValueError("released operations must be known integer operation IDs")
    return frozenset(released)


def retained_path_bound(instance, initial, released=()):
    """Keep all job edges, frozen-machine order, and optimistic durations.

    A bound below the target is necessary but does not ensure repair feasibility
    at that target. In particular, released-operation resource conflicts are
    deliberately relaxed here.
    """
    horizon = validate_schedule(instance, initial)
    released = _release_set(instance, released)
    rows = {r["operation"]: r for r in initial["assignments"]}
    edges = defaultdict(set)
    for kind, chains in (("job", instance.jobs.values()),
                         ("machine", _frozen_queues(initial, released).values())):
        for chain in chains:
            for a, b in zip(chain, chain[1:]):
                edges[a, b].add(kind)
    predecessors = defaultdict(list)
    for a, b in edges:
        predecessors[b].append(a)
    duration = {o: min(options.values()) if o in released else options[rows[o]["machine"]]
                for o, options in instance.options.items()}
    finish, parent = {}, {}
    for o in sorted(rows, key=lambda o: (rows[o]["start"], o)):
        before = max(sorted(predecessors[o]), key=lambda a: finish[a], default=None)
        parent[o] = before
        finish[o] = duration[o] + (finish[before] if before is not None else 0)
    last = max(sorted(finish), key=lambda o: finish[o])
    bound, path = finish[last], []
    while last is not None:
        path.append(last)
        last = parent[last]
    path.reverse()
    if not 0 < bound <= horizon:
        raise ValueError("invalid retained-path bound")
    return PathWitness(bound, tuple(path), tuple(duration[o] for o in path),
                       tuple(tuple(sorted(edges[a, b])) for a, b in zip(path, path[1:])))


def witness_extensions(instance, initial, released, target):
    """List every effective single addition on one over-target witness.

    The operation must shorten a duration lower bound or remove a machine-only
    edge of this witness. Shared job/machine edges remain as job edges. A new
    longest path can keep the overall bound unchanged; such additions are not
    discarded. Numeric order only makes enumeration reproducible.
    """
    if type(target) is not int or target < 0:
        raise ValueError("target must be a nonnegative integer")
    released = _release_set(instance, released)
    witness = retained_path_bound(instance, initial, released)
    if witness.lower_bound <= target:
        return ExtensionPool("BOUND_NOT_EXCLUDING_TARGET", target, tuple(sorted(released)), witness, ())
    rows = {r["operation"]: r for r in initial["assignments"]}
    machine_edges = [(a, b) for (a, b), kinds in zip(
        zip(witness.operations, witness.operations[1:]), witness.relations)
        if "machine" in kinds and "job" not in kinds]
    extensions = []
    for o in sorted(set(witness.operations) - released):
        shorter = min(instance.options[o].values()) < instance.options[o][rows[o]["machine"]]
        removable = tuple((a, b) for a, b in machine_edges if o in (a, b))
        if not shorter and not removable:
            continue
        addition = tuple(sorted(released | {o}))
        extensions.append(WitnessExtension(o, addition, shorter, removable,
                                           retained_path_bound(instance, initial, addition)))
    return ExtensionPool("EXTENSIONS" if extensions else "NO_EFFECTIVE_ADDITION", target,
                         tuple(sorted(released)), witness, tuple(extensions))
