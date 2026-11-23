"""Eight complete groups: explicit scaled and structural proposals, no CP."""

from collections import defaultdict, deque
from math import ceil
from random import Random

from alns_neighborhoods import select_neighborhood
from experiments.alns_pool_runtime import proposal_rng
from experiments.alns_sparse import ACTIONS, features, graph, semantic
from experiments.alns_layers_runtime import Controller as ScaleController
from experiments import conditional_selection_cost_pilot as cost
from repair import _frozen_queues

SLOTS = (*ACTIONS, "small-window", "small-block", "bottleneck", "migration")
FEATURES = ("released_fraction", "critical_coverage", "near_critical_coverage",
    "current_machine_coverage", "eligible_machine_coverage", "max_load_machine_coverage",
    "mean_eligible_fraction", "processing_reduction_fraction", "time_span_fraction",
    "boundary_edge_fraction", "current_order_span_fraction", "relaxed_gap_fraction",
    "certified_no_strict_improvement")


def clipped_block(instance, anchor, rho, rng):
    rows = {r["operation"]: r for r in anchor["assignments"]}
    ids = sorted(rows)
    target = max(1, ceil(rho * len(ids)))
    seed = rng.choice(ids)
    selected, queue = {seed}, deque([seed])
    while queue and len(selected) < target:
        row = rows[queue.popleft()]
        neighbors = [op for op in ids if op not in selected and rows[op]["start"] <= row["end"]
                     and rows[op]["end"] >= row["start"]]
        rng.shuffle(neighbors)
        added = neighbors[:target - len(selected)]
        selected.update(added)
        queue.extend(added)
    return {"operator": "parallel-block", "released": sorted(selected), "rho": rho,
        "target_operations": target, "seed_operation": seed, "actual_operations": len(selected),
        "queue_exhausted": not queue, "truncated_at_target": len(selected) == target,
        "empty": False, "version": "clipped-block-v1"}


def structural(instance, anchor, rho, rng, kind):
    rows = {r["operation"]: r for r in anchor["assignments"]}
    _, slack, edges = graph(instance, anchor)
    adjacent = {op: set() for op in rows}
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    gains = {op: instance.options[op][r["machine"]] - min(instance.options[op].values()) for op, r in rows.items()}
    migration = kind == "migration" and max(gains.values()) > 0
    eligible = [op for op in sorted(rows) if (gains[op] == max(gains.values()) if migration else slack[op] == min(slack.values()))]
    seed = rng.choice(eligible)
    target_machine = None
    competing = set()
    if migration:
        options = instance.options[seed]
        target_machine = rng.choice([m for m in sorted(options) if options[m] == min(options.values())])
        r = rows[seed]
        competing = {op for op, other in rows.items() if other["machine"] == target_machine
                     and other["start"] <= r["end"] and other["end"] >= r["start"]} - {seed}
    target, selected = max(1, ceil(rho * len(rows))), {seed}
    frontier = set(adjacent[seed])
    while len(selected) < target:
        available = competing - selected
        if not available:
            available = frontier - selected
        if not available:
            # A disconnected retained graph may exhaust the local component.
            available = set(rows) - selected
        least = min(slack[op] for op in available)
        op = rng.choice(sorted(op for op in available if slack[op] == least))
        selected.add(op)
        frontier.update(adjacent[op])
    return {"operator": kind, "released": sorted(selected), "rho": rho,
        "target_operations": target, "actual_operations": len(selected), "empty": False,
        "seed_operation": seed, "target_machine": target_machine,
        "processing_potential": gains[seed], "migration_fallback": kind == "migration" and not migration,
        "version": "structural-group-v1"}


def proposals(instance, anchor, rho, seed, size):
    cost.check(size in (4, 8), "unknown candidate population")
    pool = []
    for slot in SLOTS[:size]:
        rng = proposal_rng(seed, slot)
        if slot in ("parallel-block", "small-block"):
            proposal = clipped_block(instance, anchor, rho / 2 if slot == "small-block" else rho, rng)
        elif slot in ("bottleneck", "migration"):
            proposal = structural(instance, anchor, rho, rng, slot)
        else:
            proposal = select_neighborhood(instance, anchor, "time-window" if slot == "small-window" else slot,
                                           rho / 2 if slot == "small-window" else rho, rng)
        pool.append({"slot": slot, "proposal": proposal})
    return pool


def describe(instance, anchor, pool, context):
    """Identical numerical candidate information is supplied to both learners."""
    horizon = anchor["makespan"]
    n = len(instance.options)
    vectors, details = [], []
    for item in pool:
        group = item["proposal"]["released"]
        f = semantic(features(instance, anchor, group))
        intervals = sum(len(instance.options[op]) for op in group)
        vector = [float(f[k]) for k in FEATURES] + [n / 500, intervals / max(1, n),
            context["rho"], context["remaining_seconds"] / 60, context["tau"] / 3,
            context["window_count"] / 20, context["window_optimal"] / max(1, context["window_count"])]
        vector += [float(item["slot"] == slot) for slot in SLOTS]
        details.append(f)
        vectors.append(vector)
    return vectors, details


class Controller(ScaleController):
    def __init__(self, cfg, method, clock=None):
        super().__init__(cfg, method)
        self.clock = clock or (lambda: 0.)
        self.model = None
        if self.variant["family"] in ("tree", "gnn"):
            from experiments.candidate_study_model import load_model
            self.model = load_model(self.variant["model_path"], self.variant["model_sha256"])

    def select(self, instance, anchor, rng):
        cost.check(self.last_phase is None, "uncompleted repair")
        elapsed = self.clock()
        context = {"rho": self.rho, "remaining_seconds": max(0., self.cfg["total_ms"] / 1000 - elapsed),
                   "tau": self.theta, "window_count": self.window_count, "window_optimal": self.window_optimal}
        proposal_seed, choice_seed = rng.getrandbits(64), rng.getrandbits(64)
        family = self.variant["family"]
        if family == "time-window":
            selected = select_neighborhood(instance, anchor, "time-window", self.rho,
                                           proposal_rng(proposal_seed, "time-window"))
            pool, index, eligible, scores = [], None, [], []
        else:
            pool = proposals(instance, anchor, self.rho, proposal_seed, self.variant["pool_size"])
            if family == "uniform":
                scores = [0.] * len(pool)
            else:
                vectors, details = describe(instance, anchor, pool, context)
                for item, f in zip(pool, details, strict=True):
                    item["features"] = f
                if family == "gap":
                    scores = [-f["relaxed_lower_bound"] for f in details]
                else:
                    from experiments.candidate_study_model import predict
                    scores = predict(self.model, instance, anchor, pool, vectors)
                if all(f["certified_no_strict_improvement"] for f in details):
                    scores = [0.] * len(pool)
            best = max(scores)
            eligible = [i for i, score in enumerate(scores) if score == best]
            index = Random(choice_seed).choice(eligible)
            selected = dict(pool[index]["proposal"])
        self.last_phase = "local"
        selected.update(phase="local", controller=self.snapshot(), context_elapsed_seconds=elapsed,
            context=context, proposal_seed=proposal_seed, choice_seed=choice_seed, pool=pool,
            pool_index=index, eligible_indices=eligible, predicted_scores=scores,
            choice_reason=family, pool_all_zero=None)
        return selected
