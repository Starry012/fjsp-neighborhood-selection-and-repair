"""Bounded same-prefix branch acquisition, reusing the frozen CP collector.

No optimizer and no implicit STOP labels. Run from an immutable upload bundle;
the original collector, repair engine and online runtime are not modified.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from random import Random
import shutil
import subprocess
import sys
from time import perf_counter

from experiments import conditional_selection_collect as legacy
from experiments import conditional_selection_cost_pilot as cost


def group_key(case, released, cfg):
    identity = {"input": case["sha256"], "anchor": case["anchor_sha256"],
                "released": sorted(released), "target": cfg["repair_target_seconds"],
                "workers": cfg["workers_per_repair"], "presolve": cfg["cp_model_presolve"]}
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_case(case, ids, cfg):
    """Pure seeded construction: full tails, sparse checkpoints, physical dedup."""
    order = sorted(ids)
    cost.check(len(order) == len(set(order)) > cfg["prefix_size"], "invalid operation domain")
    Random(cfg["prefix_seed"]).shuffle(order)
    prefix = order[:cfg["prefix_size"]]
    remaining = sorted(set(order) - set(prefix))
    candidates = Random(cfg["candidate_seed"]).sample(remaining, cfg["candidate_count"])
    sizes = sorted({len(remaining) if k == "remaining" else k for k in cfg["extension_sizes"]
                    if k == "remaining" or k <= len(remaining)})
    paths, groups = [], {}

    def reference(selected, alias):
        key = group_key(case, selected, cfg)
        record = groups.setdefault(key, {"group_key": key, "released": sorted(selected), "aliases": []})
        record["aliases"].append(alias)
        return key

    root_key = reference(prefix, {"path": "root", "extension": 0, "k": len(prefix)})
    for seed in cfg["suffix_seeds"]:
        common = remaining.copy()
        Random(seed).shuffle(common)
        for candidate in candidates:
            tail = [candidate] + [op for op in common if op != candidate]
            path_id = f"{case['instance']}-a{candidate}-s{seed}"
            checkpoints = [{"extension": 0, "k": len(prefix), "group_key": root_key}]
            for extension in sizes:
                alias = {"path": path_id, "extension": extension, "k": len(prefix) + extension}
                key = reference(prefix + tail[:extension], alias)
                checkpoints.append({"extension": extension, "k": alias["k"], "group_key": key})
            paths.append({"id": path_id, "candidate": candidate, "suffix_seed": seed,
                          "order": prefix + tail, "checkpoints": checkpoints})
    tasks = []
    for key, group in groups.items():
        for seed in cfg["solver_seeds"]:
            tasks.append({"instance": case["instance"], "group_key": key,
                          "k": len(group["released"]), "released": group["released"], "solver_seed": seed})
    Random(cfg["task_order_seed"]).shuffle(tasks)
    for number, task in enumerate(tasks, 1):
        task["id"] = f"branch-{number:04d}"
    return {"case": case, "prefix": prefix, "candidates": candidates, "paths": paths,
            "groups": list(groups.values()), "tasks": tasks,
            "references_before_dedup": (1 + len(paths) * len(sizes)) * len(cfg["solver_seeds"])}


def profile_paths(root, prepared, cfg, checkpoint, deadline):
    """Time CPU forward paths separately from offline checkpoint CP solves."""
    import torch
    from conditional_model import build_policy_graph
    from conditional_training import LambdaConditionalPolicy
    from fjsp import parse_instance
    from experiments.conditional_model_probe import parameter_hash

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model = LambdaConditionalPolicy().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    before = parameter_hash(model)
    cost.check(before == cfg["timing"]["parameter_sha256"], "V2 parameters differ")
    timings, interrupted = [], False
    with torch.inference_mode():
        for unit in prepared:
            case = unit["case"]
            instance = parse_instance(cost.read(case["input"]))
            anchor = cost.read(case["anchor"])
            started = perf_counter()
            graph = build_policy_graph(instance, anchor)
            graph_seconds = perf_counter() - started
            start = perf_counter()
            encoded = model.encode(graph)
            encoding_seconds = perf_counter() - start
            for price in cfg["timing"]["lambda_contexts"]:
                for path in unit["paths"]:
                    selected, selected_set, steps = [], set(), []
                    start = perf_counter()
                    for k in range(len(path["order"]) + 1):
                        if perf_counter() >= deadline:
                            interrupted = True
                            break
                        entered = perf_counter()
                        scores = model.score_encoded(encoded, selected, lambda_value=price,
                                                     remaining_seconds=cfg["timing"]["remaining_seconds_context"],
                                                     repair_seconds=cfg["repair_target_seconds"])
                        values = torch.cat((scores.operations, scores.stop.reshape(1))).tolist()
                        cost.check(all(math.isfinite(v) for v in values), "nonfinite scores")
                        legal = [(op, value) for op, value in zip(encoded.operation_ids, values[:-1])
                                 if op not in selected_set]
                        if selected:
                            legal.append(("STOP", values[-1]))
                        best_scored = max(legal, key=lambda pair: pair[1])[0]
                        forced = path["order"][k] if k < len(path["order"]) else "STOP"
                        cost.check(any(op == forced for op, _ in legal), "forced action is masked")
                        finished = perf_counter()
                        steps.append({"k": k, "entered_seconds": entered - start,
                                      "finished_seconds": finished - start,
                                      "score_seconds": finished - entered,
                                      "forced_action": forced, "highest_scored_action": best_scored})
                        if forced != "STOP":
                            selected.append(forced)
                            selected_set.add(forced)
                    timings.append({"instance": case["instance"], "path": path["id"], "lambda": price,
                                    "steps": steps, "complete": len(steps) == len(path["order"]) + 1,
                                    "graph_seconds": graph_seconds, "encoding_seconds": encoding_seconds})
                    if interrupted:
                        break
                if interrupted:
                    break
            if interrupted:
                break
    cost.check(parameter_hash(model) == before, "weights changed during timing")
    report = {"device": "cpu", "weights_sha256": before, "torch": torch.__version__,
              "cuda_initialized": torch.cuda.is_initialized(), "timings": timings,
              "interrupted": interrupted, "optimizer_steps": 0,
              "interpretation": cfg["timing"]["interpretation"]}
    cost.save(root / "selection-timing.json", report)
    return report


def verify_bundle(bundle):
    manifest = cost.read(bundle / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        cost.check(cost.digest(bundle / name) == expected, f"bundle changed: {name}")
    return manifest


def prepare_case(root, unit, cfg, bundle, manifest, seconds, affinity):
    root.mkdir()
    shutil.copytree(bundle, root / "source")
    case = dict(unit["case"])
    for field, target in [("input", "input.json"), ("anchor", "anchor.json")]:
        shutil.copyfile(case[field], root / target)
        case[field] = str((root / target).resolve())
    legacy_cfg = {"repair_target_seconds": cfg["repair_target_seconds"],
                  "solver_seeds": cfg["solver_seeds"],
                  "proposed_batch_wall_limit_seconds": max(0, seconds),
                  "proposed_cleanup_grace_seconds": cfg["cleanup_grace_seconds"],
                  "proposed_abnormal_request_watchdog_seconds": cfg["request_watchdog_seconds"],
                  "proposed_output_limit_decimal_GB": cfg["output_limit_bytes"] / 1e9}
    plan = {"config": legacy_cfg, "case": case, "tasks": unit["tasks"],
            "source_sha256": manifest["files_sha256"], "affinity": affinity,
            "environment": {"python": platform.python_version(), "ortools": version("ortools")},
            "timing": "Warm CPU CP requests only; selection timing is recorded separately at batch root.",
            "training": False}
    cost.save(root / "branch-aliases.json", {k: v for k, v in unit.items() if k != "case"})
    cost.save(root / "plan.json", plan)
    return plan


def skip_case(root, plan, reason):
    with (root / "runs.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for task in plan["tasks"]:
            file.write(json.dumps({"task": task, "state": "NOT_STARTED", "reason": reason}) + "\n")
    cost.save(root / "runtime.json", {"wall_seconds": 0, "worker_pid": None, "process_reaped": True,
                                     "exit_code": None, "completed": 0, "errors": 0, "stopped": reason})


def directory_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def collect(root, bundle, *, server=True):
    from fjsp import parse_instance, validate_schedule
    from experiments.resched_qualification import configure_cpu
    started = perf_counter()
    manifest = verify_bundle(bundle)
    cfg = cost.read(bundle / "branch-plan.json")
    specification = cost.read(bundle / "branch-tasks.json")
    cost.check(specification["expected"] == cfg["expected"], "task count identity differs")
    root = Path(root).resolve()
    if server:
        approved = Path(".benchmark-work").resolve()
        cost.check(root.is_relative_to(approved / "runs") and bundle.is_relative_to(approved / "code"),
                   "paths outside personal data2 project")
        cost.check(Path(sys.prefix).resolve() == approved / "envs/fjsp-train", "wrong server environment")
        expected = {"python": "3.14.10", "torch": "2.14.1+cu130", "numpy": "2.5.4", "ortools": "9.15.6755"}
        actual = {name: platform.python_version() if name == "python" else version(name) for name in expected}
        cost.check(actual == expected, f"environment differs: {actual}")
        allocated = subprocess.run(["du", "-s", "-B1", str(approved.parent)], capture_output=True,
                                   text=True, check=True, timeout=20)
        personal_bytes = int(allocated.stdout.split()[0])
        cost.check(personal_bytes < cfg["personal_storage_warning_bytes"], "personal data2 reached 250GB warning")
        cost.check(shutil.disk_usage(approved).free > cfg["output_limit_bytes"] + 2_000_000_000,
                   "insufficient data2 free space")
    else:
        personal_bytes = None
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    affinity = configure_cpu()
    finish = started + cfg["batch_wall_limit_seconds"]
    acquire_until = finish - cfg["final_audit_reserve_seconds"]
    prepared = []
    for unit in specification["units"]:
        case = dict(unit["case"])
        for field in ("input", "anchor"):
            case[field] = str((bundle / case[field]).resolve())
        instance = parse_instance(cost.pinned(case["input"], case["sha256"]))
        initial = cost.pinned(case["anchor"], case["anchor_sha256"])
        cost.check(validate_schedule(instance, initial) == case["before"], "anchor changed")
        cost.check(make_case(case, instance.options, cfg)["tasks"] == unit["tasks"], "seeded tasks changed")
        prepared.append({**unit, "case": case})
    cost.save(root / "batch-plan.json", {"config": cfg, "bundle": str(bundle),
              "manifest_sha256": cost.digest(bundle / "manifest.json"),
              "tasks_sha256": cost.digest(bundle / "branch-tasks.json"),
              "created_utc": datetime.now(timezone.utc).isoformat(), "personal_bytes_before": personal_bytes,
              "environment": {"python": platform.python_version(), "torch": version("torch"),
                              "numpy": version("numpy"), "ortools": version("ortools")},
              "server_timing": server, "training": False})
    legacy.emit({"stage": "CPU-selection-timing", "training": False, "GPU_used": False})
    checkpoint = bundle / cfg["timing"]["checkpoint"]
    cost.check(cost.digest(checkpoint) == cfg["timing"]["checkpoint_sha256"], "timing checkpoint changed")
    profile_paths(root, prepared, cfg, checkpoint, acquire_until - cfg["cleanup_grace_seconds"])
    blocked = None
    for unit in prepared:
        name = unit["case"]["instance"]
        remaining = acquire_until - perf_counter()
        blocked = blocked or ("batch wall limit" if remaining < cfg["cleanup_grace_seconds"] + 1 else None)
        blocked = blocked or ("output warning threshold" if directory_bytes(root) >= cfg["output_limit_bytes"] - 100_000_000 else None)
        case_root = root / name
        plan = prepare_case(case_root, unit, cfg, bundle, manifest, remaining, affinity)
        if blocked:
            skip_case(case_root, plan, blocked)
        else:
            legacy.emit({"stage": "CP-collection", "instance": name, "requests": len(unit["tasks"])})
            legacy.collect(case_root, plan)
            runtime = cost.read(case_root / "runtime.json")
            if runtime["errors"]:
                blocked = "earlier worker error; no automatic replacement"
    cost.save(root / "acquisition-runtime.json", {"wall_seconds": perf_counter() - started,
              "limit_seconds": cfg["batch_wall_limit_seconds"], "training": False})
    return audit(root)


def audit(root):
    root = Path(root).resolve()
    plan = cost.read(root / "batch-plan.json")
    bundle = Path(plan["bundle"])
    if (root / "audit.json").exists():
        for name, expected in cost.read(root / "audit.json")["files_sha256"].items():
            cost.check(cost.digest(root / name) == expected, f"frozen acquisition evidence changed: {name}")
    verify_bundle(bundle)
    cost.check(cost.digest(bundle / "manifest.json") == plan["manifest_sha256"], "manifest changed")
    cost.check(cost.digest(bundle / "branch-tasks.json") == plan["tasks_sha256"], "tasks changed")
    specification = cost.read(bundle / "branch-tasks.json")
    all_rows, group_rows, case_results, audit_errors = [], [], [], []
    for unit in specification["units"]:
        name = unit["case"]["instance"]
        case_root = root / name
        recorded = cost.read(case_root / "plan.json")
        cost.check(recorded["tasks"] == unit["tasks"], "physical task slots changed")
        cost.check(cost.read(case_root / "branch-aliases.json") == {k: v for k, v in unit.items() if k != "case"},
                   "branch aliases changed")
        # Legacy audit checks every schedule, frozen restriction, bound and saved receipt.
        # Its by-size consistency statistic is deliberately not used for branch labels.
        previous_cwd = Path.cwd()
        try:
            os.chdir(bundle)
            legacy.audit(case_root)
        finally:
            os.chdir(previous_cwd)
        detail = cost.read(case_root / "summary.json")
        audit_errors.extend({"instance": name, **issue} for issue in detail["issues"]
                            if issue.get("state") != "NOT_STARTED")
        runtime = detail["runtime"]
        if runtime["worker_pid"] is not None and runtime["exit_code"] != 0:
            audit_errors.append({"instance": name, "worker_exit_code": runtime["exit_code"]})
        rows = [json.loads(line) for line in (case_root / "runs.jsonl").read_text().splitlines()]
        all_rows.extend(rows)
        for group in unit["groups"]:
            repeated = [row for row in rows if row["task"]["group_key"] == group["group_key"]]
            complete = (len(repeated) == len(plan["config"]["solver_seeds"])
                        and {row["task"]["solver_seed"] for row in repeated} == set(plan["config"]["solver_seeds"])
                        and all(row["state"] == "COMPLETED" for row in repeated))
            group_rows.append({"instance": name, "group_key": group["group_key"],
                               "aliases": group["aliases"], "released_count": len(group["released"]),
                               "seed_observations": [{"seed": row["task"]["solver_seed"], "state": row["state"],
                                    "value": row.get("value"), "request_seconds": row.get("request_seconds"),
                                    "native_status": row.get("native_status"), "request_id": row["task"]["id"]}
                                    for row in repeated], "three_seeds_completed": complete,
                               "all_three_improve": complete and all(row["gain"] > 0 for row in repeated)})
        case_results.append({"instance": name, "anchor_Cmax": unit["case"]["before"],
                             "planned": len(rows), "validated": detail["validated"],
                             "best_saved_Cmax": detail["best_saved_Cmax"],
                             "worker_reaped": detail["runtime"]["process_reaped"]})
    timing = cost.read(root / "selection-timing.json")
    cost.check(not timing["cuda_initialized"] and timing["optimizer_steps"] == 0, "unexpected compute scope")
    cost.check(timing["device"] == "cpu" and timing["weights_sha256"] == plan["config"]["timing"]["parameter_sha256"],
               "timing device/weight identity changed")
    paths = {(u["case"]["instance"], p["id"]): p for u in specification["units"] for p in u["paths"]}
    seen_profiles = set()
    for profile in timing["timings"]:
        key = (profile["instance"], profile["path"], profile["lambda"])
        cost.check(key not in seen_profiles and key[:2] in paths
                   and key[2] in plan["config"]["timing"]["lambda_contexts"], "unknown or duplicate timing profile")
        seen_profiles.add(key)
        path = paths[key[:2]]
        previous = 0
        for k, step in enumerate(profile["steps"]):
            forced = path["order"][k] if k < len(path["order"]) else "STOP"
            cost.check(step["k"] == k and step["forced_action"] == forced, "timed path changed")
            cost.check(all(math.isfinite(step[field]) for field in
                           ["entered_seconds", "finished_seconds", "score_seconds"]), "nonfinite path clock")
            cost.check(previous <= step["entered_seconds"] <= step["finished_seconds"]
                       and abs(step["score_seconds"] - step["finished_seconds"] + step["entered_seconds"]) < 1e-7,
                       "path timing changed")
            previous = step["finished_seconds"]
        cost.check(profile["complete"] == (len(profile["steps"]) == len(path["order"]) + 1), "path completeness changed")
    counts = dict(Counter(row["state"] for row in all_rows))
    cost.check(len(all_rows) == plan["config"]["expected"]["CP_requests"], "missing planned slots")
    summary = {"planned": len(all_rows), "states": counts, "cases": case_results,
               "unique_groups": len(group_rows), "paths": plan["config"]["expected"]["paths"],
               "three_seed_complete_groups": sum(g["three_seeds_completed"] for g in group_rows),
               "three_seed_improving_groups": sum(g["all_three_improve"] for g in group_rows),
               "all_observations_validated": sum(c["validated"] for c in case_results) == len(all_rows),
               "audit_errors": audit_errors, "selection_profiles": len(timing["timings"]),
               "complete_selection_profiles": sum(t["complete"] for t in timing["timings"]),
               "expected_selection_profiles": len(paths) * len(plan["config"]["timing"]["lambda_contexts"]),
               "acquisition": cost.read(root / "acquisition-runtime.json"),
               "training": False, "numeric_labels_generated": False, "GPU_used": False,
               "complete_online_time_labels": False, "server_timing": plan["server_timing"]}
    summary["passed"] = (summary["all_observations_validated"] and not audit_errors and not timing["interrupted"]
                         and summary["complete_selection_profiles"] == summary["expected_selection_profiles"]
                         and summary["acquisition"]["wall_seconds"] <= summary["acquisition"]["limit_seconds"])
    generated = {"summary.json": summary, "group-observations.json": group_rows}
    for name, value in generated.items():
        if (root / name).exists():
            cost.check(cost.read(root / name) == value, f"frozen {name} differs")
        else:
            cost.save(root / name, value)
    if not (root / "audit.json").exists():
        names = ["batch-plan.json", "acquisition-runtime.json", "selection-timing.json",
                 "summary.json", "group-observations.json"]
        for unit in specification["units"]:
            names.extend(f"{unit['case']['instance']}/{name}" for name in
                         ["plan.json", "branch-aliases.json", "runs.jsonl", "runtime.json", "summary.json", "audit.json"])
        cost.save(root / "audit.json", {"files_sha256": {name: cost.digest(root / name) for name in names},
                                      "training": False})
    legacy.emit(summary)
    print("FJSP_BRANCH_COLLECTION_OK" if summary["passed"] else "FJSP_BRANCH_COLLECTION_INCOMPLETE", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=Path.cwd())
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    summary = audit(args.output) if args.audit else collect(args.output, args.bundle.resolve())
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
