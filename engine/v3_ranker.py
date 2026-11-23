"""CPU models that score FJSP release sets from one verified incumbent."""

from collections import defaultdict
from dataclasses import dataclass
from math import log1p

import torch
from torch import Tensor, nn

from fjsp import Instance, validate_schedule
from move_scan import _critical_operations


RELATIONS = ("job-next", "job-prev", "machine-next", "machine-prev", "to-machine", "to-operation")


@dataclass
class Graph:
    nodes: Tensor
    edges: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]
    released: Tensor


def _incumbent(instance: Instance, schedule: dict):
    scale = validate_schedule(instance, schedule)
    rows = {row["operation"]: row for row in schedule["assignments"]}
    queues = defaultdict(list)
    for op in sorted(rows, key=lambda op: rows[op]["start"]):
        queues[rows[op]["machine"]].append(op)
    return scale, rows, queues


def _positions(operations, releases):
    positions = {op: i for i, op in enumerate(operations)}
    if any(not r or len(set(r)) != len(r) or not set(r) <= positions.keys() for r in releases):
        raise ValueError("release sets must contain distinct known operations")
    return positions


def graph_input(instance: Instance, schedule: dict, releases: list[tuple[int, ...]]) -> Graph:
    scale, rows, queues = _incumbent(instance, schedule)
    operations = list(instance.options)
    op_index = _positions(operations, releases)
    machines = sorted({m for choices in instance.options.values() for m in choices})
    machine_index = {m: len(operations) + i for i, m in enumerate(machines)}
    nodes = [[1, rows[op]["start"] / scale, rows[op]["end"] / scale, 0] for op in operations]
    nodes.extend([0, 0, 0, sum(instance.options[op][m] for op in queues[m]) / scale] for m in machines)
    by_type = {r: [] for r in RELATIONS}

    for sequences, prefix in ((instance.jobs.values(), "job"), (queues.values(), "machine")):
        for sequence in sequences:
            for a, b in zip(sequence, sequence[1:]):
                by_type[prefix + "-next"].append((op_index[a], op_index[b], 0, 0))
                by_type[prefix + "-prev"].append((op_index[b], op_index[a], 0, 0))
    for op, options in instance.options.items():
        for machine, duration in options.items():
            a, b = op_index[op], machine_index[machine]
            attributes = duration / scale, int(rows[op]["machine"] == machine)
            by_type["to-machine"].append((a, b, *attributes))
            by_type["to-operation"].append((b, a, *attributes))
    edges = {}
    for relation, links in by_type.items():
        source = torch.tensor([link[0] for link in links], dtype=torch.long)
        target = torch.tensor([link[1] for link in links], dtype=torch.long)
        attributes = torch.tensor([link[2:] for link in links], dtype=torch.float32).reshape(-1, 2)
        degree = torch.bincount(target, minlength=len(nodes)).clamp_min(1).float().view(-1, 1, 1)
        edges[relation] = source, target, attributes, degree

    released = torch.zeros((len(operations), len(releases)), dtype=torch.bool)
    for column, release in enumerate(releases):
        released[[op_index[op] for op in release], column] = True
    return Graph(torch.tensor(nodes, dtype=torch.float32), edges, released)


def tabular_input(instance: Instance, schedule: dict, releases: list[tuple[int, ...]]) -> Tensor:
    """A baseline with processing times, criticality and projected machine overlap."""
    scale, rows, queues = _incumbent(instance, schedule)
    _positions(instance.options, releases)
    critical = _critical_operations(instance, rows, queues, scale)
    loads = {m: sum(instance.options[op][m] for op in seq) / scale for m, seq in queues.items()}
    machine_count = len({m for options in instance.options.values() for m in options})
    common = [log1p(len(rows)), log1p(len(instance.jobs)), log1p(machine_count)]
    per_op, overlaps = {}, {}
    for op in sorted({op for release in releases for op in release}):
        row = rows[op]
        machine = row["machine"]
        options = instance.options[op]
        span = options[machine]
        per_op[op] = [row["start"] / scale, row["end"] / scale, span / scale, log1p(len(options)),
                      (span - min(options.values())) / scale, float(op in critical), loads[machine]]
        overlaps[op] = []
        for alternate, duration in options.items():
            if alternate == machine:
                continue
            for other in queues.get(alternate, ()):
                shared = min(row["start"] + duration, rows[other]["end"]) - max(row["start"], rows[other]["start"])
                if shared > 0:
                    overlaps[op].append((other, shared / scale))
    features = []
    for release in releases:
        ops = set(release)
        columns = list(zip(*(per_op[op] for op in release)))
        hits = [hit for op in release for hit in overlaps[op]]
        features.append(common + [log1p(len(ops)), len(ops) / len(rows)] +
                        [sum(col) / len(col) for col in columns] + [max(col) for col in columns] +
                        [sum(length for _, length in hits), sum(length for other, length in hits if other in ops),
                         len({rows[op]["machine"] for op in ops}) / machine_count])
    return torch.tensor(features, dtype=torch.float32).reshape(len(releases), 22)


class RelationLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.own = nn.Linear(width, width)
        self.message = nn.ModuleDict({r: nn.Linear(width, width, bias=False) for r in RELATIONS})
        self.eligibility = nn.ModuleDict({r: nn.Linear(2, width) for r in ("to-machine", "to-operation")})

    def forward(self, states: Tensor, graph: Graph) -> Tensor:
        updated = self.own(states)
        for relation, (source, target, attributes, degree) in graph.edges.items():
            messages = self.message[relation](states)[source]
            if relation in self.eligibility:
                gate = self.eligibility[relation](attributes).sigmoid()
                messages = messages * gate[:, None, :]
            pooled = torch.zeros_like(states).index_add_(0, target, messages)
            updated = updated + pooled / degree
        return states + updated.relu()


class ScoreHead(nn.Module):
    def __init__(self, features: int, width: int, use_budget: bool):
        super().__init__()
        self.use_budget = use_budget
        self.layers = nn.Sequential(nn.Linear(features + 1, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, encoded: Tensor, budget_ms: Tensor) -> Tensor:
        features = encoded[:, None, :].expand(-1, len(budget_ms), -1)
        budget = (budget_ms / 200 if self.use_budget else torch.zeros_like(budget_ms))
        return self.layers(torch.cat((features, budget[None, :, None].expand(len(encoded), -1, -1)), dim=-1)).squeeze(-1)


class GraphRanker(nn.Module):
    def __init__(self, width: int, *, use_budget: bool):
        super().__init__()
        self.input = nn.Linear(5, width)
        self.steps = nn.ModuleList(RelationLayer(width) for _ in range(2))
        self.head = ScoreHead(3 * width + 1, width, use_budget)

    def forward(self, graph: Graph, budget_ms: Tensor) -> Tensor:
        mask = graph.released
        flag = torch.cat((mask, mask.new_zeros((len(graph.nodes) - len(mask), mask.shape[1]))))
        state = self.input(torch.cat((graph.nodes[:, None, :].expand(-1, mask.shape[1], -1),
                                      flag.unsqueeze(-1).float()), dim=-1)).relu()
        for step in self.steps:
            state = step(state, graph)
        operations = state[:len(mask)]
        size = mask.sum(dim=0)
        selected = mask.unsqueeze(-1)
        mean = (operations * selected).sum(dim=0) / size[:, None]
        peak = operations.masked_fill(~selected, -torch.inf).amax(dim=0)
        encoded = torch.cat((mean, peak, operations.mean(dim=0), size.float().log1p()[:, None]), dim=1)
        return self.head(encoded, budget_ms)


class TabularRanker(nn.Module):
    def __init__(self, width: int, train_features: Tensor):
        super().__init__()
        self.register_buffer("center", train_features.mean(dim=0))
        self.register_buffer("scale", train_features.std(dim=0, unbiased=False).clamp_min(0.001))
        self.head = ScoreHead(train_features.shape[1], width, True)

    def forward(self, features: Tensor, budget_ms: Tensor) -> Tensor:
        return self.head((features - self.center) / self.scale, budget_ms)
