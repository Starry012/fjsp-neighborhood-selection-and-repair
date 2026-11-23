"""Version2 lambda-conditioned scoring and partial-label comparison losses.

No labels are inferred from missing outcomes. No optimizer, collector or trainer
is defined here; callers must supply already-supported comparisons.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from math import isfinite, log1p

import torch
from torch import nn
from torch.nn import functional as F

from conditional_model import (BoundConditionalPolicy, ConditionalGraphPolicy, CONTEXT_FEATURES, DIRECTIONS,
                               EncodedSchedule, PolicyOutput)
from conditional_repair_feedback import Followup
from conditional_selector import ActionScores, SelectionState


STOP = "STOP"
Action = int | str


def checked_lambda(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError("lambda_value must be a finite nonnegative number")
    return float(value)


class LambdaConditionalPolicy(ConditionalGraphPolicy):
    """Reuse the original encoder; append lambda only to the scoring context."""

    architecture = "conditional-attention-lambda-v2"

    def __init__(self, width=64, layers=2):
        super().__init__(width=width, layers=layers)
        self.operation_head = nn.Sequential(nn.Linear(6 * width + CONTEXT_FEATURES + 1, width),
                                            nn.ReLU(), nn.Linear(width, 1))
        self.stop_head = nn.Sequential(nn.Linear(5 * width + CONTEXT_FEATURES + 1, width),
                                       nn.ReLU(), nn.Linear(width, 1))

    def score_encoded(
        self, encoded: EncodedSchedule, selected: Iterable[int], *, lambda_value,
        previous: Iterable[int] = (), direction: Followup | None = None,
        remaining_seconds=60, repair_seconds=1, return_attention=False,
    ) -> PolicyOutput:
        price = checked_lambda(lambda_value)
        if encoded.parameter_stamp != self._stamp():
            raise ValueError("stale encoding: encode again after changing parameters, mode or device")
        if torch.is_grad_enabled() and not encoded.grad_enabled:
            raise ValueError("encode again with gradients enabled before differentiable scoring")
        if (not isfinite(remaining_seconds) or remaining_seconds < 0 or not isfinite(repair_seconds)
                or repair_seconds < 0 or direction not in DIRECTIONS):
            raise ValueError("use finite nonnegative budgets and a known feedback direction")
        indices = self._indices(encoded, selected)
        previous_indices = self._indices(encoded, previous)
        n = len(encoded.operation_ids)
        selected_read, weights = self.attention.read(
            encoded.queries, encoded.keys[indices], encoded.values[indices], return_attention=return_attention)
        previous_read, _ = self.attention.read(
            encoded.queries[-1:], encoded.keys[previous_indices], encoded.values[previous_indices])
        remaining_mask = torch.ones(n, dtype=torch.bool, device=encoded.operations.device)
        remaining_mask[indices] = False
        remaining_read, _ = self.attention.read(
            encoded.queries[-1:], encoded.keys[remaining_mask], encoded.values[remaining_mask])
        scalars = encoded.operations.new_tensor([
            len(indices) / n, log1p(len(indices)), len(previous_indices) / n,
            remaining_seconds / 60, repair_seconds, float(not indices),
            *(float(direction == option) for option in DIRECTIONS), price,
        ])
        common = torch.cat((encoded.global_state, previous_read[0], scalars))
        features = torch.cat((encoded.operations, selected_read[:-1], encoded.operations * selected_read[:-1],
                              common[None, :].expand(n, -1)), dim=-1)
        stop_features = torch.cat((encoded.global_state, selected_read[-1], previous_read[0], remaining_read[0], scalars))
        return PolicyOutput(self.operation_head(features).squeeze(-1), self.stop_head(stop_features).squeeze(-1), weights)


class BoundLambdaPolicy(BoundConditionalPolicy):
    def __init__(self, model, item, *, lambda_value, remaining_seconds=60, repair_seconds=1):
        self.lambda_value = checked_lambda(lambda_value)
        super().__init__(model, item, remaining_seconds=remaining_seconds, repair_seconds=repair_seconds)

    def __call__(self, state: SelectionState) -> ActionScores:
        if state.anchor_sha256 != self.encoded.anchor_sha256:
            raise ValueError("scoring cache belongs to a different current schedule")
        remaining = self.remaining_seconds() if callable(self.remaining_seconds) else self.remaining_seconds
        with torch.inference_mode():
            result = self.model.score_encoded(self.encoded, state.selected, lambda_value=self.lambda_value,
                                             previous=state.previous, direction=state.direction,
                                             remaining_seconds=remaining, repair_seconds=self.repair_seconds)
            values = torch.cat((result.operations, result.stop.reshape(1))).cpu().tolist()
        return ActionScores(dict(zip(self.encoded.operation_ids, values[:-1])), values[-1])


@dataclass(frozen=True)
class Preference:
    better: Action
    worse: Action
    weight: float = 1.0


@dataclass(frozen=True)
class ApproximateTie:
    left: Action
    right: Action
    weight: float = 1.0


@dataclass(frozen=True)
class ComparisonLoss:
    total: torch.Tensor
    ranking: torch.Tensor | None
    ties: torch.Tensor | None
    preference_count: int
    tie_count: int


def comparison_loss(
    output: PolicyOutput, operation_ids: Iterable[int], preferences: Iterable[Preference] = (),
    ties: Iterable[ApproximateTie] = (), *, selected: Iterable[int],
    legal_actions: Collection[Action] | None = None, tie_coefficient=1.0,
) -> ComparisonLoss | None:
    """One state and one lambda; normalize each label kind within that state.

    Unknown comparisons are omitted, not negative or tied. No supported labels
    returns None so a trainer can exclude the state from its supervised average.
    Supply an extra legal mask for direction, tried-group and proof constraints.
    tie_coefficient=1 is a prototype API default, not a frozen trial setting.
    """
    ids, chosen = tuple(operation_ids), tuple(selected)
    if not ids or any(type(op) is not int for op in ids) or len(set(ids)) != len(ids):
        raise ValueError("operation IDs must be distinct integers")
    if any(type(op) is not int or op not in ids for op in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError("selected operations must be distinct known IDs")
    if output.operations.shape != (len(ids),) or output.stop.shape != torch.Size([]):
        raise ValueError("expected one score per operation and a scalar STOP")
    logits = torch.cat((output.operations, output.stop.reshape(1)))
    if not torch.isfinite(logits).all():
        raise ValueError("comparison scores must be finite")
    coefficient = checked_lambda(tie_coefficient)
    if coefficient == 0:
        raise ValueError("tie_coefficient must be positive")
    positions = {op: i for i, op in enumerate(ids)} | {STOP: len(ids)}
    base_legal = set(ids) - set(chosen)
    if chosen:
        base_legal.add(STOP)
    legal = base_legal if legal_actions is None else set(legal_actions)
    if any(type(a) is not int and a != STOP for a in legal) or not legal <= base_legal:
        raise ValueError("extra action mask includes an unknown or illegal action")

    seen, graph = set(), {}

    def index_pair(a, b, weight):
        if (type(a) is not int and a != STOP) or (type(b) is not int and b != STOP) or a == b:
            raise ValueError("use two different integer operation IDs or STOP")
        if a not in legal or b not in legal:
            raise ValueError("comparison includes a masked or unknown action")
        if not isfinite(weight) or weight <= 0:
            raise ValueError("comparison weights must be finite and positive")
        pair = frozenset((a, b))
        if pair in seen:
            raise ValueError("duplicate, contradictory or both strict-and-tied comparison")
        seen.add(pair)
        return positions[a], positions[b]

    strict_terms, strict_weights, tie_terms, tie_weights = [], [], [], []
    for item in preferences:
        better, worse = index_pair(item.better, item.worse, item.weight)
        graph.setdefault(item.better, set()).add(item.worse)
        strict_terms.append(F.softplus(logits[worse] - logits[better]))
        strict_weights.append(item.weight)
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError("strict preferences contain a cycle") from error
    for item in ties:
        left, right = index_pair(item.left, item.right, item.weight)
        difference = logits[left] - logits[right]
        tie_terms.append(F.smooth_l1_loss(difference, torch.zeros_like(difference)))
        tie_weights.append(item.weight)

    def average(terms, weights):
        return (torch.stack(terms) * logits.new_tensor(weights)).sum() / sum(weights) if terms else None

    ranking, tied = average(strict_terms, strict_weights), average(tie_terms, tie_weights)
    if ranking is None and tied is None:
        return None
    total = (ranking if ranking is not None else logits.sum() * 0) + (coefficient * tied if tied is not None else 0)
    return ComparisonLoss(total, ranking, tied, len(strict_terms), len(tie_terms))
