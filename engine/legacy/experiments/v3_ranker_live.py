"""Execute the fixed development ranking comparison against live CP-SAT."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

import torch

from baseline import load_pinned_json, solve_ect
from experiments.v3_dev_replay import _candidates
from experiments.v3_ranker_dev import _save, _sha, METHODS
from fjsp import parse_instance, validate_schedule
from repair import _frozen_queues, repair_schedule
from v3_ranker import GraphRanker, TabularRanker, graph_input, tabular_input


def _load_models(config, rankers):
    rank_config = load_pinned_json(Path("experiments/v3_ranker_dev.json"), config["ranker_config_sha256"])
    if (platform.python_version() != rank_config["python"] or torch.__version__ != rank_config["torch"]
            or version("ortools") != "9.12.4544"):
        raise ValueError("use the pinned Python, PyTorch and OR-Tools builds")
    torch.set_num_threads(rank_config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    manifest_path = rankers / "artifacts.json"
    pinned_manifest = config.get("checkpoint_manifest_sha256")
    manifest = (load_pinned_json(manifest_path, pinned_manifest) if pinned_manifest
                else json.loads(manifest_path.read_bytes()))
    trained = load_pinned_json(rankers / "run.json", manifest["run.json"])
    if trained["config"] != rank_config or trained["code_sha256"]["v3_ranker.py"] != _sha(Path("v3_ranker.py")):
        raise ValueError("ranker configuration or code differs from the trained snapshot")
    models, weight_hashes = {}, {}
    for method, seed in product(METHODS, config["seeds"]):
        path = rankers / f"{method}-{seed}.pt"
        weight_hashes[path.name] = _sha(path)
        if weight_hashes[path.name] != manifest[path.name]:
            raise ValueError(f"checkpoint fingerprint mismatch: {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if method == "tabular":
            model = TabularRanker(rank_config["width"], state["center"][None, :])
        else:
            model = GraphRanker(rank_config["width"], use_budget=method == "graph")
        model.load_state_dict(state)
        models[method, seed] = model.eval()
    return trained, models, weight_hashes


def _load(config, source, rankers):
    label_config = load_pinned_json(Path("experiments/v3_dev_replay.json"), config["label_config_sha256"])
    trained, models, weight_hashes = _load_models(config, rankers)
    plans = {p["instance"]: p for p in load_pinned_json(
        source / "plan.json", trained["source_hashes"]["plan.json"]
    ) if p["split"] == "validation-development"}
    if plans.keys() != set(config["instances"]) or set(trained["validation_cases"]) != plans.keys():
        raise ValueError("unexpected validation cases")
    cases = {p["instance"]: p for p in label_config["cases"]}
    instances = {}
    for name, p in plans.items():
        case = cases[name]
        if case["split"] != p["split"] or case["sha256"] != p["sha256"]:
            raise ValueError(f"{name}: instance differs from the locked development plan")
        instances[name] = parse_instance(load_pinned_json(
            Path(label_config["instance_dir"]) / f"{name}.json", case["sha256"]
        ))
    return label_config, plans, instances, models, weight_hashes


@torch.inference_mode()
def _rank(instance, initial, candidates, method, seed, budget_ms, model, cfg):
    order = list(range(len(candidates)))
    if not order:
        return order
    if method == "structure":
        return sorted(order, key=lambda i: not any(s.startswith("blocker-") for s in candidates[i]["sources"]))
    if method == "random":
        shuffle_id = cfg["seeds"].index(seed)
        Random(f"replay:{cfg['order_seed']}:{instance.name}:{shuffle_id}").shuffle(order)
        return order
    make_input = tabular_input if method == "tabular" else graph_input
    inputs = make_input(instance, initial, [tuple(c["released"]) for c in candidates])
    scores = model(inputs, torch.tensor([budget_ms], dtype=torch.float32)).squeeze(1).tolist()
    return sorted(order, key=lambda i: -scores[i])


def _execute(instance, method, seed, budget_ms, model, cfg, label_cfg):
    start = perf_counter()
    deadline = start + cfg["total_ms"] / 1000
    work_deadline = deadline - cfg.get("global_reserve_ms", 0) / 1000
    reserve = cfg["return_reserve_ms"] / 1000
    budget = budget_ms / 1000
    ect = solve_ect(instance, deadline)
    initial = ect.schedule
    if initial is not None:
        validate_schedule(instance, initial)
    initial_finish = perf_counter()
    if initial is None or initial_finish > deadline:
        return {"stop": "ECT_DEADLINE", "initial": initial, "elapsed_ms": (initial_finish - start) * 1000,
                "best_makespan": None, "calls": []}
    candidates = _candidates(instance, initial, label_cfg)
    prepared = perf_counter()
    order = _rank(instance, initial, candidates, method, seed, budget_ms, model, cfg) if prepared < deadline else []
    ranked = perf_counter()
    best = initial["makespan"]
    calls = []
    stop = "POOL_EXHAUSTED" if ranked <= deadline else "GLOBAL_DEADLINE"
    for idx in order:
        call_start = perf_counter()
        end_by = min(deadline, call_start + budget)
        solver_deadline = min(work_deadline, end_by - reserve)
        if call_start >= solver_deadline:
            stop = "GLOBAL_DEADLINE"
            break
        released = candidates[idx]["released"]
        try:
            result = repair_schedule(instance, initial, released, solver_deadline,
                                     workers=cfg["solver_workers"], seed=cfg["solver_seed"])
        except Exception as exc:
            calls.append({"released": released, "error": f"{type(exc).__name__}: {exc}"})
            stop = "ERROR"
            break
        finish = perf_counter()
        on_time = finish <= end_by
        if on_time and result.schedule is not None:
            best = min(best, result.schedule["makespan"])
        calls.append({"released": released, "effective_ms": (end_by - call_start) * 1000,
                      "truncated": end_by == deadline and call_start + budget > deadline,
                      "start_ms": (call_start - start) * 1000, "finish_ms": (finish - start) * 1000,
                      "solver_deadline_ms": (solver_deadline - start) * 1000,
                      "on_time": on_time, **vars(result)})
        if finish > deadline:
            stop = "GLOBAL_DEADLINE"
            break
    elapsed = (perf_counter() - start) * 1000
    return {"stop": stop, "initial": initial, "candidates": candidates, "order": order, "calls": calls,
            "best_makespan": best, "prep_ms": (prepared - start) * 1000,
            "rank_ms": (ranked - prepared) * 1000, "elapsed_ms": elapsed}


def _review(log, tasks, plans, instances, cfg):
    expected = set(tasks)
    outcomes = {}
    statistics = Counter()
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name, method, seed, budget = key = tuple(row[k] for k in ("instance", "method", "seed", "budget_ms"))
            expected.remove(key)
            statistics["runs"] += 1
            statistics["wall_overruns"] += row["elapsed_ms"] > cfg["total_ms"]
            statistics[f"stop:{row['stop']}"] += 1
            if row.get("error") or row["stop"] == "ECT_DEADLINE":
                continue
            instance = instances[name]
            initial = row["initial"]
            best = validate_schedule(instance, initial)
            if initial != plans[name]["initial"] or row["candidates"] != plans[name]["candidates"]:
                raise ValueError(f"{name}: live inputs differ from development replay")
            if row["rank_ms"] + row["prep_ms"] <= cfg["total_ms"] and sorted(row["order"]) != list(range(len(row["candidates"]))):
                raise ValueError(f"{key}: candidate order is not a permutation")
            calls = row["calls"]
            order = row["order"]
            if len(calls) > len(order):
                raise ValueError(f"{key}: more calls than candidates")
            for call, idx in zip(calls, order):
                statistics["repairs"] += 1
                if call["released"] != row["candidates"][idx]["released"]:
                    raise ValueError(f"{key}: called out of rank order")
                if call.get("error"):
                    statistics["repair_errors"] += 1
                    continue
                statistics[f"solver:{call['status']}"] += 1
                statistics["truncated_calls"] += call["truncated"]
                statistics["late_calls"] += not call["on_time"]
                if call["status"] not in {"OPTIMAL", "FEASIBLE", "UNKNOWN", "TIME_LIMIT"}:
                    statistics["unexpected_status"] += 1
                limit = min(cfg["total_ms"], call["start_ms"] + budget)
                if abs(limit - call["start_ms"] - call["effective_ms"]) > 1e-5:
                    raise ValueError(f"{key}: wrong effective call budget")
                if call["on_time"] != (call["finish_ms"] <= limit) and abs(call["finish_ms"] - limit) > 1e-5:
                    raise ValueError(f"{key}: wrong time classification")
                if call["schedule"] is not None:
                    value = validate_schedule(instance, call["schedule"])
                    frozen = frozenset(call["released"])
                    if value > initial["makespan"] or _frozen_queues(call["schedule"], frozen) != _frozen_queues(initial, frozen):
                        raise ValueError(f"{key}: infeasible repair")
                    statistics["verified_repairs"] += 1
                    if call["on_time"]:
                        best = min(best, value)
            if best != row["best_makespan"]:
                raise ValueError(f"{key}: reported best is not the best on-time result")
            if row["stop"] != "ERROR":
                outcomes[key] = row
    if expected:
        raise ValueError("live log is missing planned cases")

    comparisons = {}
    for budget, opponent in product(cfg["repair_budgets_ms"], (m for m in cfg["methods"] if m != "graph")):
        counts = Counter()
        for name, seed in product(cfg["instances"], cfg["seeds"]):
            a = outcomes.get((name, "graph", seed, budget))
            b = outcomes.get((name, opponent, seed, budget))
            if a is None or b is None:
                counts["missing"] += 1
                continue
            delta = b["best_makespan"] - a["best_makespan"]
            counts["win" if delta > 0 else "loss" if delta < 0 else "tie"] += 1
        comparisons[f"{budget}ms graph vs {opponent}"] = dict(counts)
    timing = {}
    for budget, method in product(cfg["repair_budgets_ms"], cfg["methods"]):
        rows = [r for (_, m, _, b), r in outcomes.items() if (m, b) == (method, budget)]
        if rows:
            timing[f"{budget}ms {method}"] = {
                "runs": len(rows), "rank_median_ms": median(r["rank_ms"] for r in rows),
                "calls_median": median(len(r["calls"]) for r in rows),
                "elapsed_median_ms": median(r["elapsed_ms"] for r in rows),
            }
    return {"statistics": dict(statistics), "graph_comparisons": comparisons, "timing": timing,
            "runs_sha256": _sha(log)}


def run(config_path, source, rankers, output):
    cfg = json.loads(config_path.read_bytes())
    label_cfg, plans, instances, models, weights = _load(cfg, source, rankers)
    tasks = list(product(cfg["instances"], cfg["methods"], cfg["seeds"], cfg["repair_budgets_ms"]))
    Random(cfg["order_seed"]).shuffle(tasks)
    output.mkdir(parents=True)
    files = (config_path, Path(__file__), Path("experiments/v3_dev_replay.py"), Path("v3_ranker.py"),
             Path("repair.py"), Path("fjsp.py"), Path("baseline.py"), Path("blocking_chains.py"),
             Path("neighborhoods.py"), Path("move_scan.py"), Path("v3_chain_pilot.py"))
    _save(output / "run.json", {"config": cfg, "started_utc": datetime.now(timezone.utc).isoformat(),
                               "platform": platform.platform(), "processor": platform.processor(),
                               "python": platform.python_version(), "torch": torch.__version__, "ortools": version("ortools"),
                               "torch_threads": torch.get_num_threads(),
                               "code_sha256": {p.as_posix(): _sha(p) for p in files}, "weight_sha256": weights,
                               "plan_sha256": _sha(source / "plan.json"), "run_order": tasks})
    log = output / "runs.jsonl"
    with log.open("x", encoding="utf-8") as handle:
        for i, (name, method, seed, budget) in enumerate(tasks, 1):
            started = perf_counter()
            try:
                row = _execute(instances[name], method, seed, budget, models.get((method, seed)), cfg, label_cfg)
            except Exception as exc:
                row = {"stop": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_ms": (perf_counter() - started) * 1000}
            record = {"instance": name, "method": method, "seed": seed, "budget_ms": budget, **row}
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            if i % 30 == 0 or row["stop"] == "ERROR":
                print(f"{i}/{len(tasks)} runs, latest stop: {row['stop']}", flush=True)
    summary = _review(log, tasks, plans, instances, cfg)
    _save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(summary["statistics"].get(k, 0) for k in (
        "stop:ERROR", "stop:ECT_DEADLINE", "repair_errors", "unexpected_status"
    )))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("experiments/v3_ranker_live.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/v3-dev-replay"))
    parser.add_argument("--rankers", type=Path, default=Path("outputs/v3-ranker-dev"))
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.source, args.rankers, args.output))
