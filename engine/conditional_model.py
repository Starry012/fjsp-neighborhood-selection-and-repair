"""Full-schedule graph encoding and candidate attention over a selected set."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite, log1p, sqrt

import torch
from torch import Tensor, nn

from conditional_repair_feedback import Followup
from conditional_selector import ActionScores, SelectionState
from fjsp import Instance
from joint_selection import anchor_fingerprint
from v3_ranker import Graph, RelationLayer, graph_input


DIRECTIONS = (None, Followup.CHANGE_AND_SHRINK, Followup.EXPAND, Followup.GLOBAL_DEADLINE)
CONTEXT_FEATURES = 6 + len(DIRECTIONS)


@dataclass(frozen=True)
class PolicyGraph:
    graph: Graph
    operation_ids: tuple[int, ...]
    anchor_sha256: str


@dataclass(frozen=True)
class EncodedSchedule:
    operation_ids: tuple[int, ...]
    positions: dict[int, int]
    anchor_sha256: str
    operations: Tensor
    global_state: Tensor
    queries: Tensor
    keys: Tensor
    values: Tensor
    parameter_stamp: tuple
    grad_enabled: bool


@dataclass(frozen=True)
class PolicyOutput:
    operations: Tensor
    stop: Tensor
    selected_attention: Tensor | None = None


def build_policy_graph(instance: Instance, schedule: dict) -> PolicyGraph:
    """Use the old verified graph features, without a selection flag."""
    return PolicyGraph(graph_input(instance, schedule, []), tuple(instance.options),
                       anchor_fingerprint(instance, schedule))


class SelectedSetAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.scale = sqrt(width)

    def read(self, queries: Tensor, keys: Tensor, values: Tensor, *, return_attention=False):
        if len(keys) == 0:
            read = queries.new_zeros((len(queries), queries.shape[-1]))
            weights = queries.new_zeros((len(queries), 0))
        else:
            weights = (queries @ keys.T / self.scale).softmax(dim=-1)
            read = weights @ values
        return read, weights if return_attention else None


class ConditionalGraphPolicy(nn.Module):
    def __init__(self, width: int = 64, layers: int = 2):
        super().__init__()
        if type(width) is not int or width <= 0 or type(layers) is not int or layers <= 0:
            raise ValueError("use positive integer model dimensions")
        self.width = width
        self.node_input = nn.Linear(4, width)
        self.steps = nn.ModuleList(RelationLayer(width) for _ in range(layers))
        self.attention = SelectedSetAttention(width)
        self.stop_query = nn.Parameter(torch.randn(width) / sqrt(width))
        self.operation_head = nn.Sequential(nn.Linear(6 * width + CONTEXT_FEATURES, width),
                                            nn.ReLU(), nn.Linear(width, 1))
        self.stop_head = nn.Sequential(nn.Linear(5 * width + CONTEXT_FEATURES, width),
                                       nn.ReLU(), nn.Linear(width, 1))

    def _stamp(self):
        # Detect ordinary in-place parameter updates and model/device changes.
        # Callers must not bypass PyTorch version counters with parameter.data.
        return (self.training, tuple((id(p), p._version, p.device, p.dtype) for p in self.parameters()))

    def encode(self, item: PolicyGraph) -> EncodedSchedule:
        device, dtype = self.node_input.weight.device, self.node_input.weight.dtype
        graph = item.graph
        edges = {relation: (source.to(device), target.to(device), attributes.to(device=device, dtype=dtype),
                            degree.to(device=device, dtype=dtype))
                 for relation, (source, target, attributes, degree) in graph.edges.items()}
        moved = Graph(graph.nodes.to(device=device, dtype=dtype), edges, graph.released.to(device))
        state = self.node_input(moved.nodes[:, None, :]).relu()
        for step in self.steps:
            state = step(state, moved)
        nodes = state.squeeze(1)
        operations, machines = nodes[:len(item.operation_ids)], nodes[len(item.operation_ids):]
        global_state = torch.cat((operations.mean(dim=0), machines.mean(dim=0)))
        queries = self.attention.query(torch.cat((operations, self.stop_query[None, :]), dim=0))
        return EncodedSchedule(
            item.operation_ids, {op: i for i, op in enumerate(item.operation_ids)}, item.anchor_sha256,
            operations, global_state, queries, self.attention.key(operations), self.attention.value(operations),
            self._stamp(), torch.is_grad_enabled(),
        )

    @staticmethod
    def _indices(encoded, operations: Iterable[int]):
        group = tuple(operations)
        if (any(type(op) is not int or op not in encoded.positions for op in group)
                or len(set(group)) != len(group)):
            raise ValueError("use a set of distinct known operation IDs")
        return [encoded.positions[op] for op in sorted(group)]

    def score_encoded(
        self, encoded: EncodedSchedule, selected: Iterable[int], *,
        previous: Iterable[int] = (), direction: Followup | None = None,
        remaining_seconds: float = 60, repair_seconds: float = 1, return_attention=False,
    ) -> PolicyOutput:
        if encoded.parameter_stamp != self._stamp():
            raise ValueError("stale encoding: encode again after changing parameters, mode or device")
        if torch.is_grad_enabled() and not encoded.grad_enabled:
            raise ValueError("encode again with gradients enabled before differentiable scoring")
        if (not isfinite(remaining_seconds) or remaining_seconds < 0
                or not isfinite(repair_seconds) or repair_seconds < 0 or direction not in DIRECTIONS):
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
            *(float(direction == option) for option in DIRECTIONS),
        ])
        common = torch.cat((encoded.global_state, previous_read[0], scalars))
        interaction = encoded.operations * selected_read[:-1]
        features = torch.cat((encoded.operations, selected_read[:-1], interaction,
                              common[None, :].expand(n, -1)), dim=-1)
        operation_logits = self.operation_head(features).squeeze(-1)
        stop_features = torch.cat((encoded.global_state, selected_read[-1], previous_read[0],
                                   remaining_read[0], scalars))
        stop_logit = self.stop_head(stop_features).squeeze(-1)
        return PolicyOutput(operation_logits, stop_logit, weights)

    def forward(self, item: PolicyGraph, selected: Iterable[int], **kwargs) -> PolicyOutput:
        """Differentiable path; no detached graph cache is inserted here."""
        return self.score_encoded(self.encode(item), selected, **kwargs)


class BoundConditionalPolicy:
    """Read-only scoring adapter for the existing stateful selector."""

    def __init__(
        self, model: ConditionalGraphPolicy, item: PolicyGraph, *,
        remaining_seconds: float | Callable[[], float] = 60, repair_seconds: float = 1,
    ):
        if model.training:
            raise ValueError("put the model in eval mode before binding an inference cache")
        self.model, self.remaining_seconds, self.repair_seconds = model, remaining_seconds, repair_seconds
        with torch.inference_mode():
            self.encoded = model.encode(item)

    def __call__(self, state: SelectionState) -> ActionScores:
        if state.anchor_sha256 != self.encoded.anchor_sha256:
            raise ValueError("scoring cache belongs to a different current schedule")
        remaining = self.remaining_seconds() if callable(self.remaining_seconds) else self.remaining_seconds
        with torch.inference_mode():
            output = self.model.score_encoded(
                self.encoded, state.selected, previous=state.previous, direction=state.direction,
                remaining_seconds=remaining, repair_seconds=self.repair_seconds)
            values = torch.cat((output.operations, output.stop.reshape(1))).cpu().tolist()
        return ActionScores(dict(zip(self.encoded.operation_ids, values[:-1])), values[-1])
