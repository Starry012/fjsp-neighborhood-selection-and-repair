"""Stateful conditional group decoding, independent of the scoring network."""

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from time import perf_counter

from conditional_repair_feedback import Followup, nonimproving_followup
from fjsp import Instance, validate_schedule
from joint_selection import anchor_fingerprint, certify_failure
from repair import RepairResult


@dataclass(frozen=True)
class SelectionState:
    anchor_sha256: str
    selected: tuple[int, ...]
    previous: tuple[int, ...]
    direction: Followup | None


@dataclass(frozen=True)
class ActionScores:
    operations: Mapping[int, float]
    stop: float


class SelectionStatus(StrEnum):
    READY = "ready"
    DEADLINE = "deadline"
    STATE_LIMIT = "state_limit"
    DIRECTION_EXHAUSTED = "direction_exhausted"


@dataclass(frozen=True)
class Proposal:
    status: SelectionStatus
    anchor_sha256: str
    released: tuple[int, ...] | None
    selected_order: tuple[int, ...]
    stop_cause: str | None
    states_scored: int
    seconds: float


@dataclass
class _Frame:
    selected: frozenset[int]
    order: tuple[int, ...]
    actions: Iterator[int | None] | None = None


class ConditionalSelector:
    """One current schedule, a deterministic score callback and bounded alternatives.

    A returned group is reserved immediately, preventing duplicate submissions.
    Scores are recomputed for each selected set, with higher scores preferred.
    Alternative actions are explored in depth-first score order when needed;
    this does not claim to find the globally highest-scoring complete group.
    Neither a search cutoff nor direction exhaustion generates a STOP label.
    """

    def __init__(self, instance: Instance, anchor: dict):
        self.instance = instance
        self.operations = tuple(sorted(instance.options))
        self._set_anchor(anchor)

    def _set_anchor(self, anchor):
        self._anchor = deepcopy(anchor)
        self._anchor["makespan"] = validate_schedule(self.instance, anchor)
        self.anchor_sha256 = anchor_fingerprint(self.instance, self._anchor)
        self._issued: set[frozenset[int]] = set()
        self._completed: set[frozenset[int]] = set()
        self._proved: list[frozenset[int]] = []
        self.direction = None
        self.previous = ()

    @property
    def anchor(self):
        return deepcopy(self._anchor)

    @property
    def issued_groups(self):
        return tuple(sorted(tuple(sorted(group)) for group in self._issued))

    @property
    def proved_groups(self):
        return tuple(sorted(tuple(sorted(group)) for group in self._proved))

    def _group(self, operations):
        group = tuple(operations)
        if (any(type(op) is not int or op not in self.instance.options for op in group)
                or len(set(group)) != len(group)):
            raise ValueError("use distinct known operation IDs")
        return frozenset(group)

    def stop_allowed(self, operations):
        group = self._group(operations)
        return bool(group) and group not in self._issued and not any(group <= failed for failed in self._proved)

    def choose(
        self,
        score: Callable[[SelectionState], ActionScores],
        *,
        deadline: float,
        max_states: int,
        observer=None,
    ) -> Proposal:
        """Decode before an absolute deadline; callback time is charged too.

        The outer runtime still owns process interruption and delivery. A slow
        callback is rejected after it returns, not preempted by this function.
        """
        if type(max_states) is not int or max_states <= 0 or not isfinite(deadline):
            raise ValueError("use a finite deadline and positive integer max_states")
        started, count = perf_counter(), 0

        def result(status, frame=None):
            group = tuple(sorted(frame.selected)) if frame is not None else None
            cause = None
            if frame is not None:
                cause = ("direction_size_limit" if self.direction == Followup.CHANGE_AND_SHRINK
                         and len(group) == limit else
                         "all_operations_selected" if len(group) == len(self.operations) else "scored_stop")
                self._issued.add(frame.selected)
            return Proposal(status, self.anchor_sha256, group, frame.order if frame else (), cause,
                            count, perf_counter() - started)

        if started >= deadline or self.direction == Followup.GLOBAL_DEADLINE:
            return result(SelectionStatus.DEADLINE)
        limit = len(self.operations)
        seed = frozenset(self.previous) if self.direction == Followup.EXPAND else frozenset()
        if self.direction == Followup.CHANGE_AND_SHRINK:
            limit = len(self.previous) - 1
        if limit < 1 or (self.direction == Followup.EXPAND and len(seed) == limit):
            return result(SelectionStatus.DIRECTION_EXHAUSTED)
        stack = [_Frame(seed, tuple(sorted(seed)))]
        visited = set()
        while stack:
            if perf_counter() >= deadline:
                return result(SelectionStatus.DEADLINE)
            frame = stack[-1]
            if frame.actions is None:
                if frame.selected in visited:
                    stack.pop()
                    continue
                if count >= max_states:
                    return result(SelectionStatus.STATE_LIMIT)
                visited.add(frame.selected)
                state = SelectionState(self.anchor_sha256, tuple(sorted(frame.selected)),
                                       self.previous, self.direction)
                scores = score(state)
                count += 1
                if perf_counter() >= deadline:
                    return result(SelectionStatus.DEADLINE)
                if (set(scores.operations) != set(self.operations)
                        or any(type(op) is not int for op in scores.operations)):
                    raise ValueError("score every known operation exactly once")
                values = {op: float(value) for op, value in scores.operations.items()}
                stop = float(scores.stop)
                if not all(isfinite(value) for value in (*values.values(), stop)):
                    raise ValueError("action scores must be finite")
                allowed_stop = self.stop_allowed(frame.selected)
                if observer is not None:
                    observer(state=state, selected_order=frame.order,
                             scores=ActionScores(dict(values), stop), size_limit=limit,
                             stop_allowed=allowed_stop)
                    if perf_counter() >= deadline:
                        return result(SelectionStatus.DEADLINE)
                # On exact ties, operation IDs precede STOP; no random tie-breaking.
                ranked = [(value, 0, op) for op, value in values.items()
                          if op not in frame.selected and len(frame.selected) < limit]
                if allowed_stop:
                    ranked.append((stop, 1, None))
                ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
                frame.actions = iter(item[2] for item in ranked)
            try:
                action = next(frame.actions)
            except StopIteration:
                stack.pop()
                continue
            if action is None:
                if perf_counter() >= deadline:
                    return result(SelectionStatus.DEADLINE)
                return result(SelectionStatus.READY, frame)
            stack.append(_Frame(frame.selected | {action}, (*frame.order, action)))
        return result(SelectionStatus.DIRECTION_EXHAUSTED)

    def _pending(self, proposal):
        if (proposal.status != SelectionStatus.READY or proposal.anchor_sha256 != self.anchor_sha256
                or proposal.released is None or frozenset(proposal.released) not in self._issued):
            raise ValueError("proposal does not belong to this current schedule")
        group = frozenset(proposal.released)
        if group in self._completed:
            raise ValueError("repair feedback already recorded")
        return group

    def record_nonimprovement(self, proposal: Proposal, repair: RepairResult, *, within_global_budget: bool):
        if type(within_global_budget) is not bool:
            raise ValueError("supply a Boolean outer deadline verdict")
        group = self._pending(proposal)
        followup = nonimproving_followup(self.instance, self._anchor, proposal.released, repair,
                                        within_global_budget=within_global_budget)
        self._completed.add(group)
        if followup == Followup.EXPAND:
            self._proved.append(group)
        self.previous, self.direction = proposal.released, followup
        return followup

    def accept_improvement(self, proposal: Proposal, repair: RepairResult, *, within_global_budget: bool):
        if type(within_global_budget) is not bool:
            raise ValueError("supply a Boolean outer deadline verdict")
        self._pending(proposal)
        if not within_global_budget:
            return False
        if repair.status not in ("FEASIBLE", "OPTIMAL"):
            raise ValueError("acceptance requires a solver solution status")
        certify_failure(self.instance, self._anchor, proposal.released, self._anchor["makespan"] - 1,
                        repair, on_time=within_global_budget)
        if repair.schedule is None or validate_schedule(self.instance, repair.schedule) >= self._anchor["makespan"]:
            raise ValueError("acceptance requires a verified strict improvement")
        self._set_anchor(repair.schedule)
        return True
