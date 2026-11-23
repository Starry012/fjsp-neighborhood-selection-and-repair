"""Choose a search direction from a nonimproving CP repair at one anchor."""

from collections.abc import Iterable
from enum import StrEnum

from fjsp import Instance, validate_schedule
from joint_selection import certify_failure
from repair import RepairResult


class Followup(StrEnum):
    CHANGE_AND_SHRINK = "change_and_shrink"
    EXPAND = "expand"
    GLOBAL_DEADLINE = "global_deadline"


def nonimproving_followup(
    instance: Instance,
    anchor: dict,
    released: Iterable[int],
    result: RepairResult,
    *,
    within_global_budget: bool,
) -> Followup:
    """Return a preference, not a new group or a STOP label.

    Call only if no improving incumbent was accepted from this repair.
    The caller supplies the outer deadline verdict; the local repair
    target can be exceeded without invalidating a certificate.
    """
    if not within_global_budget:
        return Followup.GLOBAL_DEADLINE
    group = tuple(released)
    if (not group or any(type(op) is not int or op not in instance.options for op in group)
            or len(set(group)) != len(group)):
        raise ValueError("repair feedback requires a nonempty group of distinct known operations")

    current = validate_schedule(instance, anchor)
    if result.initial_makespan != current:
        raise ValueError("repair uses a different initial objective")
    if result.schedule is not None and validate_schedule(instance, result.schedule) < current:
        raise ValueError("an improving result must go through incumbent acceptance")
    if result.status == "TIME_LIMIT":
        expected_scope = "global" if len(group) == len(instance.options) else "neighborhood"
        if result.schedule is not None or result.lower_bound is not None or result.bound_scope != expected_scope:
            raise ValueError("inconsistent repair result before search")
        return Followup.CHANGE_AND_SHRINK

    proof = certify_failure(instance, anchor, group, current - 1, result, on_time=within_global_budget)
    return Followup.EXPAND if proof is not None else Followup.CHANGE_AND_SHRINK
