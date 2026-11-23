"""Four proposals, one repair: uniform versus retained-constraint bound gap."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from alns_neighborhoods import select_neighborhood
from experiments import alns_cp_runtime as base
from experiments.alns_layers_runtime import Controller as ScaleController
from experiments.alns_sparse import ACTIONS, features, semantic
from experiments import conditional_selection_cost_pilot as cost


def proposal_rng(seed, action):
    return Random(int.from_bytes(sha256(f"pool-v1:{seed}:{action}".encode()).digest(), "big"))


def choose_pool(pool, rule, rng):
    cost.check(rule in ("pool-uniform", "pool-gap"), "unknown pool rule")
    bounds = [p["features"]["relaxed_lower_bound"] for p in pool]
    # The anchor Cmax is common, so max gap is exactly min integer LB.
    eligible = list(range(len(pool))) if rule == "pool-uniform" else [i for i, b in enumerate(bounds) if b == min(bounds)]
    return rng.choice(eligible), eligible


class Controller(ScaleController):
    def select(self, instance, anchor, rng):
        cost.check(self.last_phase is None, "selection without completing the preceding repair")
        rule = self.variant["family"]
        # Two root draws per iteration in every arm. Choice/tie draws cannot
        # perturb later proposals; each operator has its own derived stream.
        proposal_seed, choice_seed = rng.getrandbits(64), rng.getrandbits(64)
        pool, eligible, index = [], [], None
        if rule in ("pool-uniform", "pool-gap"):
            for action in ACTIONS:
                proposal = select_neighborhood(instance, anchor, action, self.rho, proposal_rng(proposal_seed, action))
                pool.append({"proposal": proposal, "features": semantic(features(instance, anchor, proposal["released"]))})
            index, eligible = choose_pool(pool, rule, Random(choice_seed))
            selected = dict(pool[index]["proposal"])
            all_zero = all(p["features"]["certified_no_strict_improvement"] for p in pool)
            reason = "all_zero_uniform" if all_zero else ("pool_uniform" if rule == "pool-uniform" else
                                                        "max_gap_tie" if len(eligible) > 1 else "max_gap")
        else:
            cost.check(rule in ACTIONS, "unknown fixed method")
            selected = select_neighborhood(instance, anchor, rule, self.rho, proposal_rng(proposal_seed, rule))
            all_zero, reason = False, "fixed"
        selected.update(phase="local", controller=self.snapshot(), choice_reason=reason,
                        proposal_seed=proposal_seed, choice_seed=choice_seed, pool=pool,
                        pool_index=index, eligible_indices=eligible, pool_all_zero=all_zero)
        self.last_phase = "local"
        return selected

    def complete(self, status, improved):
        cost.check(status in ("OPTIMAL", "FEASIBLE", "UNKNOWN", "TIME_LIMIT"), "engineering error is not a local observation")
        # Only the one executed repair contributes to the shared scale update.
        super().complete(status, improved)


def run_trial(*args, **kwargs):
    return base.run_trial(*args, worker_module="experiments.alns_pool_runtime", **kwargs)


def audit_trial(output, case, method, seed, cfg):
    row = base.audit_trial(output, case, method, seed, cfg, controller_factory=Controller)
    root = Path(output)
    events = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    starts = {e["call"]: e for e in events if e["event"] == "CALL_STARTED"}
    for call in row["calls"]:
        selection = starts[call["call"]]["selection"]
        pool = selection["pool"]
        selected_features = pool[selection["pool_index"]]["features"] if pool else None
        lower = selected_features["relaxed_lower_bound"] if selected_features else None
        if lower is not None:
            cost.check(call["after_Cmax"] is None or call["after_Cmax"] >= lower,
                       "adopted improvement contradicts the pre-repair certified bound")
            directory = root / "calls" / f"call-{call['call']:06d}"
            # Check all saved frames and native output, including unadopted ones.
            schedules = [cost.read(p)["schedule"] for p in directory.glob("candidate-*.json")]
            if (directory / "report.json").exists():
                native = cost.read(directory / "report.json")["result"]["schedule"]
                if native is not None:
                    schedules.append(native)
            cost.check(all(s["makespan"] >= lower for s in schedules), "CP output contradicts graph lower bound")
        call.update(choice_reason=selection["choice_reason"], pool_size=len(pool),
                    pool_all_zero=selection["pool_all_zero"], pool_unique_groups=len({tuple(p["proposal"]["released"]) for p in pool}),
                    selected_lower_bound=lower,
                    selected_gap_fraction=selected_features["relaxed_gap_fraction"] if selected_features else None,
                    selected_certified_zero=selected_features["certified_no_strict_improvement"] if selected_features else None,
                    eligible_count=len(selection["eligible_indices"]),
                    pool_summary=json.dumps([{"operator": p["proposal"]["operator"], "k": p["features"]["actual_k"],
                        "lower_bound": p["features"]["relaxed_lower_bound"]} for p in pool], separators=(",", ":")))
    calls = row["calls"]
    zero = [c for c in calls if c["pool_all_zero"]]
    selected_zero = [c for c in calls if c["selected_certified_zero"]]
    row["pool_diagnostics"] = {
        "pool_calls": sum(c["pool_size"] == 4 for c in calls), "all_zero_calls": len(zero),
        "all_zero_CP_request_seconds": sum(c["CP_request_seconds"] or 0 for c in zero),
        "all_zero_selection_seconds": sum(c["selection_seconds"] for c in zero),
        "all_zero_native_statuses": dict(Counter(c["native_status"] for c in zero)),
        "selected_certified_zero_calls": len(selected_zero),
        "selected_certified_zero_CP_request_seconds": sum(c["CP_request_seconds"] or 0 for c in selected_zero),
        "all_zero_observed_positive_calls": sum(c["after_Cmax"] is not None and c["after_Cmax"] < c["anchor_Cmax"] for c in zero),
        "operator_counts": dict(Counter(c["operator"] for c in calls)),
        "positive_calls": sum(c["after_Cmax"] is not None and c["after_Cmax"] < c["anchor_Cmax"] for c in calls),
        "mean_pool_unique_groups": sum(c["pool_unique_groups"] for c in calls if c["pool_size"]) /
                                   max(1, sum(bool(c["pool_size"]) for c in calls)),
        "known_gap": "All-zero pools still select uniformly and execute one CP. No skip/resample; measured cost is retained for a later ablation.",
        "counterfactual_CP": False}
    row["checkpoint_Cmax"] = {str(t): min((p["Cmax"] for p in row["anytime_trace"] if p["ms"] <= t * 1000), default=None)
                              for t in (5, 20, 40, 60)}
    row["scope"] = "Six exposed SD2 development instances, fixed graph heuristic; no learned weights, training or held-out evidence."
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    raise SystemExit(base.worker(parser.parse_args().worker, controller_factory=Controller))
