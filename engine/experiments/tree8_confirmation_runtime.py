"""Registered half-scale controls and frozen Tree8, with the existing C repair."""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
from unittest.mock import patch

from alns_neighborhoods import select_neighborhood
from experiments import candidate_study_pool as groups
from experiments import candidate_study_runtime as original
from experiments.alns_pool_runtime import proposal_rng
from experiments import conditional_selection_cost_pilot as cost

MODULE = "experiments.tree8_confirmation_runtime"
HALVES = ("small-window", "small-block")


def half_pool(instance, anchor, rho, seed, slots):
    pool = []
    for slot in slots:
        rng = proposal_rng(seed, slot)
        if slot == "small-window":
            group = select_neighborhood(instance, anchor, "time-window", rho / 2, rng)
        else:
            cost.check(slot == "small-block", "unknown half-scale slot")
            group = groups.clipped_block(instance, anchor, rho / 2, rng)
        pool.append({"slot": slot, "proposal": group})
    return pool


class Controller(groups.Controller):
    def __init__(self, cfg, method, clock=None):
        super().__init__(cfg, method, clock)
        self.method = method

    def select(self, instance, anchor, rng):
        if self.method == "Tree8":
            return super().select(instance, anchor, rng)
        cost.check(self.last_phase is None, "uncompleted repair")
        elapsed = self.clock()
        context = {"rho": self.rho, "remaining_seconds": max(0., self.cfg["total_ms"] / 1000 - elapsed),
                   "tau": self.theta, "window_count": self.window_count, "window_optimal": self.window_optimal}
        seed, choice_seed = rng.getrandbits(64), rng.getrandbits(64)
        pool = half_pool(instance, anchor, self.rho, seed, self.variant["slots"])
        scores = [0.] * len(pool)
        if self.method == "Tree2":
            from experiments.candidate_study_model import predict
            vectors, details = groups.describe(instance, anchor, pool, context)
            for item, features in zip(pool, details, strict=True):
                item["features"] = features
            scores = predict(self.model, instance, anchor, pool, vectors)
            if all(f["certified_no_strict_improvement"] for f in details):
                scores = [0.] * len(pool)
        elif self.method == "MinOptions2":
            scores = [-sum(len(instance.options[op]) for op in item["proposal"]["released"]) for item in pool]
        else:
            cost.check(self.method in ("HalfWindow", "HalfBlock", "Uniform2"), "unregistered confirmation policy")
        best = max(scores)
        eligible = [i for i, score in enumerate(scores) if score == best]
        index = Random(choice_seed).choice(eligible)
        self.last_phase = "local"
        return {**pool[index]["proposal"], "phase": "local", "controller": self.snapshot(),
                "context_elapsed_seconds": elapsed, "context": context, "proposal_seed": seed,
                "choice_seed": choice_seed, "pool": pool, "pool_index": index, "eligible_indices": eligible,
                "predicted_scores": scores, "choice_reason": self.method, "pool_all_zero": None}


def run_trial(bundle, output, case, method, seed, cfg, **kwargs):
    return original.base.run_trial(bundle, output, case, method, seed, original.case_config(cfg, case),
                                   worker_module=MODULE, **kwargs)


def audit_trial(output, case, method, seed, cfg):
    with patch.object(groups, "Controller", Controller):
        row = original.audit_trial(output, case, method, seed, cfg)
    # Preserve actual slot names (both window sizes share an operator name).
    starts = {}
    with (Path(output) / "calls.jsonl").open(encoding="utf-8") as log:
        for line in log:
            event = json.loads(line)
            if event["event"] == "CALL_STARTED":
                starts[event["call"]] = event["selection"]
    for call in row["calls"]:
        selection = starts[call["call"]]
        index = selection["pool_index"]
        call["selected_slot"] = selection["pool"][index]["slot"]
        call["eligible_options_by_slot"] = (json.dumps([-s for s in selection["predicted_scores"]])
                                            if method == "MinOptions2" else "")
    row["selected_slot_counts"] = dict(Counter(call["selected_slot"] for call in row["calls"]))
    row["pool_diagnostics"]["all_zero_selection_seconds"] = sum(
        call["selection_seconds"] for call in row["calls"] if call["pool_all_zero"])
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    with patch.object(groups, "Controller", Controller):
        raise SystemExit(original.worker(parser.parse_args().worker))
