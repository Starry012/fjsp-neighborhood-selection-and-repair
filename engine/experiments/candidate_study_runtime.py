"""Closed-loop P1/C adapter; frozen delivery, selection and repair paths reused."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from experiments import alns_cp_runtime as base
from experiments import alns_pool_runtime as pool
from experiments import conditional_selection_cost_pilot as cost
from experiments.cp_rebuild_observer import observed_repair

MODULE = "experiments.candidate_study_runtime"
Controller = pool.Controller


def execute_worker(request, publisher, *, controller_factory=None):
    cfg = request["spec"]["config"]
    variant = cfg["variants"][request["spec"]["method"]]["repair_encoding"]
    cost.check(variant in ("P1", "C"), "unfrozen repair encoding")

    def repair(instance, anchor, released, deadline, *, seed, stream, publish):
        cost.check(stream, "streaming delivery is required")
        return observed_repair(instance, anchor, released, deadline, seed=seed,
            variant=variant, publish=publish, request_started=perf_counter())

    # Keep the original function reference: the worker entrypoint temporarily
    # replaces only dispatch, never the measured loop or its clock boundaries.
    if cfg.get("candidate_stage"):
        from experiments.candidate_study_pool import Controller as GroupController
        controller_factory = lambda c, m: GroupController(c, m, clock=lambda: perf_counter() - request["started"])
    return ORIGINAL_EXECUTE(request, publisher, repair_observer=repair,
                            controller_factory=controller_factory or Controller)


ORIGINAL_EXECUTE = base.execute_worker


def case_config(cfg, case):
    if "time_by_size" not in cfg:
        return cfg
    cfg = deepcopy(cfg)
    for value in cfg["variants"].values():
        value["fixed_seconds"] = cfg["time_by_size"][str(case["operations"])]
    return cfg


def run_trial(bundle, output, case, method, seed, cfg, **kwargs):
    return base.run_trial(bundle, output, case, method, seed, case_config(cfg, case), worker_module=MODULE, **kwargs)


def audit_trial(output, case, method, seed, cfg):
    from fjsp import parse_instance
    from experiments.cp_rebuild_model import preprocess, build_model
    cfg = case_config(cfg, case)
    if cfg.get("candidate_stage"):
        from experiments.candidate_study_pool import Controller as GroupController
        events = [json.loads(line) for line in (Path(output) / "calls.jsonl").read_text().splitlines()]
        times = iter(e["selection"]["context_elapsed_seconds"] for e in events if e["event"] == "CALL_STARTED")
        row = base.audit_trial(output, case, method, seed, cfg,
            controller_factory=lambda c, m: GroupController(c, m, clock=lambda: next(times)))
    else:
        row = pool.audit_trial(output, case, method, seed, cfg)
    encoding = cfg["variants"][method]["repair_encoding"]
    request = cost.read(Path(output) / "request.json")
    instance = parse_instance(cost.pinned(request["input_path"], case["sha256"]))
    starts = {e["call"]: e for e in (json.loads(line) for line in
        (Path(output) / "calls.jsonl").read_text().splitlines()) if e["event"] == "CALL_STARTED"}
    reconstructed = 0
    for report_path in sorted((Path(output) / "calls").glob("*/report.json")):
        report = cost.read(report_path)
        cost.check(report["metrics"]["variant"] == encoding, "repair encoding changed")
        number = int(report_path.parent.name.split("-")[1])
        anchor = cost.read(report_path.parent / "anchor.json")
        released = starts[number]["selection"]["released"]
        certificate, expected = preprocess(instance, anchor, released, encoding)
        metrics = report["metrics"]
        if metrics["preprocessing"] is None:
            cost.check(report["result"]["status"] == "TIME_LIMIT" and report["model_sha256"] is None,
                       "missing preprocessing proof")
        else:
            cost.check(metrics["preprocessing"] == expected, "preprocessing replay differs")
        if report["model_sha256"] is not None:
            model = build_model(instance, anchor, released, encoding, certificate)[0]
            cost.check(sha256(str(model.proto).encode()).hexdigest() == report["model_sha256"],
                       "closed-loop repair model replay differs")
            reconstructed += 1
    row["repair_encoding"] = encoding
    row["repair_models_reconstructed_without_solving"] = reconstructed
    if cfg.get("candidate_stage"):
        from collections import Counter
        from experiments.alns_sparse import features
        zero_count = zero_cost = selected_zero_count = selected_zero_cost = 0
        for call in row["calls"]:
            selection = starts[call["call"]]["selection"]
            proposals = selection["pool"]
            anchor = cost.read(Path(output) / "calls" / f"call-{call['call']:06d}" / "anchor.json")
            # Uniform controls do not pay for scoring features online. Offline
            # audit computes these for monitoring, explicitly outside the60s.
            fs = [item.get("features") or features(instance, anchor, item["proposal"]["released"]) for item in proposals]
            selected = fs[selection["pool_index"]] if fs else None
            all_zero = bool(fs) and all(f["certified_no_strict_improvement"] for f in fs)
            if selected:
                cost.check(call["after_Cmax"] is None or call["after_Cmax"] >= selected["relaxed_lower_bound"], "bound contradiction")
                directory = Path(output) / "calls" / f"call-{call['call']:06d}"
                schedules = [cost.read(p)["schedule"] for p in directory.glob("candidate-*.json")]
                if (directory / "report.json").exists():
                    native = cost.read(directory / "report.json")["result"]["schedule"]
                    if native is not None:
                        schedules.append(native)
                cost.check(all(s["makespan"] >= selected["relaxed_lower_bound"] for s in schedules),
                           "unadopted/native frame contradicts candidate bound")
            zero_count += all_zero
            zero_cost += (call["CP_request_seconds"] or 0) if all_zero else 0
            selected_zero = bool(selected and selected["certified_no_strict_improvement"])
            selected_zero_count += selected_zero
            selected_zero_cost += (call["CP_request_seconds"] or 0) if selected_zero else 0
            call.update(choice_reason=selection["choice_reason"], pool_size=len(proposals), pool_all_zero=all_zero,
                pool_unique_groups=len({tuple(item["proposal"]["released"]) for item in proposals}),
                selected_lower_bound=selected["relaxed_lower_bound"] if selected else None,
                selected_gap_fraction=selected["relaxed_gap_fraction"] if selected else None,
                selected_certified_zero=selected_zero, eligible_count=len(selection["eligible_indices"]),
                pool_summary=json.dumps([{"operator": item["slot"], "k": f["actual_k"], "lower_bound": f["relaxed_lower_bound"]}
                    for item, f in zip(proposals, fs, strict=True)], separators=(",", ":")))
        row["pool_diagnostics"] = {"pool_calls": sum(bool(c["pool_size"]) for c in row["calls"]),
            "all_zero_calls": zero_count, "all_zero_CP_request_seconds": zero_cost,
            "selected_certified_zero_calls": selected_zero_count, "selected_certified_zero_CP_request_seconds": selected_zero_cost,
            "all_zero_observed_positive_calls": 0,
            "operator_counts": dict(Counter(c["operator"] for c in row["calls"])),
            "uniform_feature_monitoring": "computed during offline audit, not charged as online scoring"}
        row["scope"] = "Fresh problem candidate study; C fixed, NumPy CPU inference charged to total60s."
        row["training"] = False
    if not cfg.get("candidate_stage"):
        row["scope"] = "P1/C full60s development confirmation; same selection and delivery instrumentation."
    return row


def worker(path):
    with patch.object(base, "execute_worker", execute_worker):
        return base.worker(path, controller_factory=Controller)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    raise SystemExit(worker(parser.parse_args().worker))
