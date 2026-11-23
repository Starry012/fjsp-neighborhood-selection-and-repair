"""Generate distinct singleton covers; exclusions apply only to core selection."""

from dataclasses import dataclass

from critical_paths import minimum_critical_cover
from fjsp import Instance


@dataclass(frozen=True)
class SingletonCores:
    cores: tuple[int, ...]
    stop: str
    next_minimum_size: int | None


def diverse_singleton_cores(instance: Instance, schedule: dict, *, limit: int = 6) -> SingletonCores:
    """Return up to limit distinct one-operation covers without inspecting gains.

    Selected operations are excluded only from subsequent core selection.
    Blocker expansion and CP-SAT must use their own final release sets.
    LIMIT does not claim all singleton covers have been enumerated.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    cores = []
    while len(cores) < limit:
        cover = minimum_critical_cover(instance, schedule, allowed=instance.options.keys() - set(cores))
        if cover.status != "COVER":
            return SingletonCores(tuple(cores), cover.status, cover.minimum_size)
        if cover.minimum_size != 1:
            return SingletonCores(tuple(cores), "NO_SINGLETON_COVER", cover.minimum_size)
        core, = cover.released
        if core in cores:
            raise RuntimeError("core-selection exclusion failed")
        cores.append(core)
    return SingletonCores(tuple(cores), "LIMIT", None)
