"""Bounded paper-inspired scale/time/intensification ablations, no learning."""

import argparse
from collections import Counter
import json
from pathlib import Path

from alns_neighborhoods import select_neighborhood
from experiments import alns_cp_runtime as runtime
from experiments import conditional_selection_cost_pilot as cost


class Controller:
    """Same controller for uniform random operations and the three-operator mix.

    Update after the iteration's optional full-model intensification, using only
    its local repair's status. Full-model results never inflate the optimal count.
    """

    def __init__(self, cfg, variant):
        self.cfg, self.variant = cfg, cfg["variants"][variant]
        self.rho = cfg["rho"]
        self.theta = self.variant["fixed_seconds"]
        self.local_calls = self.window_count = self.window_optimal = 0
        self.pending = False
        self.staged_status = None
        self.last_phase = None

    @property
    def target_seconds(self):
        return self.theta

    def snapshot(self):
        return {"rho": self.rho, "theta_seconds": self.theta, "local_calls": self.local_calls,
                "window_count": self.window_count, "window_optimal": self.window_optimal,
                "intensification_pending": self.pending, "staged_local_status": self.staged_status}

    def select(self, instance, anchor, rng):
        phase = "intensification" if self.pending else "local"
        method = "full-cp" if self.pending else self.variant["family"]
        selection = select_neighborhood(instance, anchor, method, self.rho, rng)
        selection.update(phase=phase, controller=self.snapshot())
        self.last_phase = phase
        return selection

    def _update(self, status):
        optimal = status == "OPTIMAL"
        self.local_calls += 1
        self.window_count += 1
        self.window_optimal += optimal
        if self.variant["adaptive_time"]:
            self.theta = min(self.cfg["time_max_seconds"], max(self.cfg["time_min_seconds"], self.theta + (1 if optimal else -1)))
        if self.window_count == self.cfg["scale_window"]:
            if self.variant["adaptive_rho"]:
                increment = (2 * self.window_optimal - self.window_count) / self.window_count * self.cfg["scale_delta"]
                self.rho = min(self.cfg["rho_max"], max(self.cfg["rho_min"], self.rho + increment))
            self.window_count = self.window_optimal = 0

    def complete(self, status, improved):
        cost.check(self.last_phase is not None, "completion without a selected request")
        if self.last_phase == "local":
            if self.variant["intensify"] and improved:
                self.pending, self.staged_status = True, status
            else:
                self._update(status)
        else:
            cost.check(self.pending and self.staged_status is not None, "full-model repair without new best")
            self.pending = False
            self._update(self.staged_status)
            self.staged_status = None
        self.last_phase = None


def run_trial(*args, **kwargs):
    return runtime.run_trial(*args, worker_module="experiments.alns_layers_runtime", **kwargs)


def audit_trial(*args, **kwargs):
    row = runtime.audit_trial(*args, controller_factory=Controller, **kwargs)
    calls = row["calls"]
    events = [json.loads(line) for line in (Path(args[0]) / "calls.jsonl").read_text().splitlines()]
    finishes = [event for event in events if event["event"] == "CALL_COMPLETED"]
    row["controller_diagnostics"] = {
        "local_calls": sum(c["phase"] == "local" for c in calls),
        "intensification_calls": sum(c["phase"] == "intensification" for c in calls),
        "local_optimal_calls": sum(c["phase"] == "local" and c["native_status"] == "OPTIMAL" for c in calls),
        "selected_rho_min": min((c["rho"] for c in calls), default=None),
        "selected_rho_max": max((c["rho"] for c in calls), default=None),
        "target_seconds_histogram": dict(Counter(str(c["target_seconds"]) for c in calls)),
        "measured_CP_request_seconds_by_phase": {phase: sum(c["CP_request_seconds"] or 0 for c in calls if c["phase"] == phase)
                                                  for phase in ("local", "intensification")},
        "final": finishes[-1]["controller_after"] if finishes else None}
    row["scope"] = "Exposed development, bounded scale/time/full-model-intensification ablation; no elite pool, roulette, history operators, ML or full-paper reproduction claim."
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    raise SystemExit(runtime.worker(parser.parse_args().worker, controller_factory=Controller))
