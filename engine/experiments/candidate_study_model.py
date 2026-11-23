"""Portable NumPy inference for complete-group tree/GNN scorers (no Torch)."""

from collections import defaultdict, deque
import numpy as np

from experiments import conditional_selection_cost_pilot as cost
from repair import _frozen_queues


def graph_data(instance, anchor, pool):
    ids, machines = sorted(instance.options), sorted({m for o in instance.options.values() for m in o})
    pos = {op: i for i, op in enumerate(ids)}
    mpos = {m: len(ids) + i for i, m in enumerate(machines)}
    rows = {r["operation"]: r for r in anchor["assignments"]}
    queues = _frozen_queues(anchor, frozenset())
    job = [(a, b) for chain in instance.jobs.values() for a, b in zip(chain, chain[1:])]
    machine = [(a, b) for chain in queues.values() for a, b in zip(chain, chain[1:])]
    edges = set(job + machine)
    children, degree = defaultdict(list), dict.fromkeys(ids, 0)
    for a, b in sorted(edges):
        children[a].append(b)
        degree[b] += 1
    order, head = [], dict.fromkeys(ids, 0)
    queue = deque(op for op in ids if degree[op] == 0)
    duration = {op: rows[op]["end"] - rows[op]["start"] for op in ids}
    while queue:
        op = queue.popleft()
        order.append(op)
        for other in children[op]:
            head[other] = max(head[other], head[op] + duration[op])
            degree[other] -= 1
            if degree[other] == 0:
                queue.append(other)
    cost.check(len(order) == len(ids), "cyclic schedule graph")
    tail = {}
    for op in reversed(order):
        tail[op] = max((duration[b] + tail[b] for b in children[op]), default=0)
    span = max(head[o] + duration[o] for o in ids)
    h = anchor["makespan"]
    load = {m: sum(duration[o] for o in queues.get(m, [])) for m in machines}
    node = []
    for op in ids:
        r, opts = rows[op], instance.options[op]
        node.append([1, 0, duration[op] / h, min(opts.values()) / h, len(opts) / len(machines),
            r["start"] / h, r["end"] / h, head[op] / h, tail[op] / h,
            (span - head[op] - duration[op] - tail[op]) / h, load[r["machine"]] / h, 0])
    for m in machines:
        node.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, load[m] / h, len(queues.get(m, [])) / len(ids)])
    relations = []
    for pairs in (job, [(b, a) for a, b in job], machine, [(b, a) for a, b in machine]):
        relations.append([[pos[a], pos[b], 1.] for a, b in pairs])
    eligible = [[pos[o], mpos[m], p / h] for o in ids for m, p in sorted(instance.options[o].items())]
    relations.extend((eligible, [[b, a, w] for a, b, w in eligible]))
    masks, boundaries = [], []
    for item in pool:
        group = set(item["proposal"]["released"])
        boundary = {o for a, b in edges if (a in group) != (b in group) for o in (a, b)}
        masks.append([int(o in group) for o in ids] + [0] * len(machines))
        boundaries.append([int(o in boundary) for o in ids] + [0] * len(machines))
    return {"nodes": node, "relations": relations, "masks": masks, "boundaries": boundaries,
            "operation_count": len(ids)}


def load_model(path, sha):
    model = cost.pinned(path, sha)
    cost.check(model["protocol"] == "complete-group-ranker-v1", "foreign model task")
    cost.check(model["repair_encoding"] == "C" and model["candidate_version"] == "eight-clipped-v1", "model domain differs")
    return model


def tree_predict(trees, x):
    def leaf(tree, row):
        while "feature" in tree:
            tree = tree["left"] if row[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree["value"]
    return np.asarray([sum(leaf(tree, row) for tree in trees) for row in x], dtype=np.float32)


def gnn_predict(weights, data, x):
    w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    nodes = np.asarray(data["nodes"], dtype=np.float32)
    h = np.maximum(nodes @ w["input.weight"].T + w["input.bias"], 0)
    for layer in range(2):
        value = h @ w[f"self_layers.{layer}.weight"].T + w[f"self_layers.{layer}.bias"]
        for rel, rows in enumerate(data["relations"]):
            agg, denom = np.zeros_like(h), np.zeros((len(h), 1), dtype=np.float32)
            if rows:
                arr = np.asarray(rows)
                src, dst, ew = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2:3].astype(np.float32)
                np.add.at(agg, dst, h[src] * ew)
                np.add.at(denom, dst, ew)
            agg /= np.maximum(denom, 1e-12)
            value += agg @ w[f"relations.{layer * 6 + rel}.weight"].T
        h = np.maximum(value, 0)
    opmask = np.arange(len(h)) < data["operation_count"]
    def pooled(mask):
        return (h * mask[:, None]).sum(0) / max(1, mask.sum())
    embedded = []
    for mask, boundary in zip(data["masks"], data["boundaries"], strict=True):
        mask, boundary = np.asarray(mask, dtype=bool), np.asarray(boundary, dtype=bool)
        embedded.append(np.concatenate((pooled(mask), pooled(opmask & ~mask), pooled(boundary),
                                       pooled(~opmask), h.mean(0))))
    readout = np.concatenate((np.asarray(embedded), np.asarray(x, dtype=np.float32)), axis=1)
    hidden = np.maximum(readout @ w["head.0.weight"].T + w["head.0.bias"], 0)
    return (hidden @ w["head.2.weight"].T + w["head.2.bias"]).reshape(-1)


def predict(model, instance, anchor, pool, vectors):
    x = np.asarray(vectors, dtype=np.float32)
    x = (x - np.asarray(model["x_mean"], dtype=np.float32)) / np.asarray(model["x_scale"], dtype=np.float32)
    if model["kind"] == "tree":
        scores = tree_predict(model["trees"], x)
    else:
        scores = gnn_predict(model["weights"], graph_data(instance, anchor, pool), x)
    cost.check(np.isfinite(scores).all(), "nonfinite learned candidate scores")
    return [float(v) for v in scores]
