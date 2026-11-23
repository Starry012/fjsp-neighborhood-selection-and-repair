"""Paired development comparison: old pool versus union, with common search rounds."""

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
from random import Random

import torch

from baseline import load_pinned_json
from blocking_chains import grow_blocking_chain
from diverse_covers import diverse_singleton_cores
from experiments.v3_dev_replay import _candidates
from experiments.v3_ranker_dev import _save, _sha
from experiments.v3_ranker_live import _rank
from fjsp import parse_instance, validate_schedule
from iterated_repair import search_rounds
from repair import _frozen_queues
from v3_ranker import TabularRanker


def _load_v4_ranker(cfg):
    """Reuse frozen weights while explicitly versioning the new solver runtime."""
    if version("ortools") != "9.15.6755":
        raise ValueError("V4 multiround requires the isolated OR-Tools 9.15.6755 environment")
    rank_cfg = load_pinned_json(Path("experiments/v3_ranker_dev.json"), cfg["ranker_config_sha256"])
    if platform.python_version() != rank_cfg["python"] or torch.__version__ != rank_cfg["torch"]:
        raise ValueError("use the pinned Python and PyTorch builds")
    root = Path("outputs/v3-ranker-dev")
    manifest = load_pinned_json(root / "artifacts.json", cfg["checkpoint_manifest_sha256"])
    trained = load_pinned_json(root / "run.json", manifest["run.json"])
    if trained["config"] != rank_cfg or trained["code_sha256"]["v3_ranker.py"] != _sha(Path("v3_ranker.py")):
        raise ValueError("trained ranker metadata or code differs")
    path = root / "tabular-42.pt"
    if _sha(path) != manifest[path.name]:
        raise ValueError("ranker weight differs from the frozen snapshot")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TabularRanker(rank_cfg["width"], state["center"][None, :])
    model.load_state_dict(state)
    return model.eval(), {path.name: manifest[path.name]}


def make_pool(instance, initial, labels, include_cover):
    old = _candidates(instance, initial, labels)
    groups = {tuple(c["released"]): {**c, "origins": []} for c in old}
    if include_cover:
        cores = diverse_singleton_cores(instance, initial, limit=6)
        for core in cores.cores:
            for cap in (1, 2, 4, 8):
                links = grow_blocking_chain(instance, initial, core, max_size=cap)
                release = tuple(sorted({core, *(link.blocker for link in links)}))
                group = groups.setdefault(release, {"released": release, "sources": [], "origins": []})
                group["sources"] = sorted(set(group["sources"]) | {"cover-blocker"})
                group["origins"].append({"core": core, "cap": cap, "links": [asdict(link) for link in links]})
    return list(groups.values())


def audit(instance, result, labels, include_cover, cfg):
    if result["initial"] is None or result["schedule"] is None:
        raise ValueError("missing initial or final feasible schedule")
    current = result["initial"]
    validate_schedule(instance, current)
    calls = returned = 0
    expected_trace = []
    statuses = Counter()
    for round_result in result["rounds"]:
        if round_result["initial"] != current:
            raise ValueError("round did not rebuild from the accepted incumbent")
        expected = json.loads(json.dumps(make_pool(instance, current, labels, include_cover)))
        if expected != round_result["candidates"]:
            raise ValueError("candidate pool cannot be reconstructed")
        if include_cover:
            old = {tuple(c["released"]) for c in make_pool(instance, current, labels, False)}
            if not old <= {tuple(c["released"]) for c in expected}:
                raise ValueError("union dropped an old candidate")
        best = current
        for call in round_result["calls"]:
            calls += 1
            statuses[call["status"]] += 1
            if call["status"] in ("INFEASIBLE", "MODEL_INVALID"):
                raise ValueError("unexpected solver status for a neighborhood with a known feasible initial schedule")
            if call["released"] != round_result["candidates"][call["candidate"]]["released"]:
                raise ValueError("call release differs from planned candidate")
            scope = "global" if set(call["released"]) == instance.options.keys() else "neighborhood"
            if call["initial_makespan"] != current["makespan"] or call["bound_scope"] != scope:
                raise ValueError("call has a stale initial value or invalid bound scope")
            value = None if call["schedule"] is None else validate_schedule(instance, call["schedule"])
            on_time = (call["finish_ms"] <= cfg["total_ms"] and
                       call["finish_ms"] <= call["start_ms"] + call["effective_ms"] + 1e-7)
            if on_time != call["on_time"]:
                raise ValueError("call deadline flag differs from recorded clock")
            if value is not None:
                returned += 1
                if (value > current["makespan"] or
                    _frozen_queues(call["schedule"], frozenset(call["released"])) != _frozen_queues(current, frozenset(call["released"])) or
                    (call["lower_bound"] is not None and call["lower_bound"] > value) or
                    (call["status"] == "OPTIMAL" and call["lower_bound"] != value)):
                    raise ValueError("invalid repair neighborhood or bound")
                if on_time and value < best["makespan"]:
                    best = call["schedule"]
                    expected_trace.append({"available_ms": call["finish_ms"], "makespan": value})
        if "accepted" in round_result and round_result["accepted"] != (best["makespan"] < current["makespan"]):
            raise ValueError("round acceptance is not a strict improvement")
        current = best
        if "best_makespan" in round_result and current["makespan"] != round_result["best_makespan"]:
            raise ValueError("round incumbent cannot be reconstructed")
    if current != result["schedule"]:
        raise ValueError("final incumbent differs from credited repairs")
    if result["trace"][1:] != expected_trace or result["delivered_on_time"] != (result["elapsed_ms"] <= cfg["total_ms"]):
        raise ValueError("incumbent trace or delivery flag cannot be reconstructed")
    validate_schedule(instance, current)
    return {"calls": calls, "returned_and_verified": returned, "statuses": dict(statuses)}


def run(output):
    blind_path = Path("experiments/v3_ranker_blind.json")
    cfg = json.loads(blind_path.read_bytes())
    labels = load_pinned_json(Path("experiments/v3_dev_replay.json"), cfg["label_config_sha256"])
    cfg = {**cfg, "total_ms": 60_000, "seeds": [42], "methods": ["old", "old-plus-cover"]}
    model, weights = _load_v4_ranker(cfg)
    cases = [c for c in cfg["cases"] if c["operations"] in (250, 500)]
    tasks = list(product([c["instance"] for c in cases], cfg["methods"]))
    Random(42).shuffle(tasks)
    instances = {c["instance"]: parse_instance(load_pinned_json(
        Path(labels["instance_dir"]) / f'{c["instance"]}.json', c["sha256"])) for c in cases}
    code_paths = list(json.loads(blind_path.with_suffix(".lock.json").read_bytes())["code_sha256"])
    code_paths += ["iterated_repair.py", "critical_paths.py", "diverse_covers.py", "experiments/v4_multiround.py", "requirements-v4.txt"]
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "plan.json", {"stage": "development-paired-multiround", "cfg": cfg,
                                "cases": cases, "tasks": tasks, "max_cores": 6, "caps": [1, 2, 4, 8],
                                "selection": "existing tabular seed 42; no retraining",
                                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                "dependencies": {name: version(name) for name in
                                                                 ("ortools", "numpy", "pandas", "protobuf", "torch", "typing-extensions")}},
                                "weight_sha256": weights, "code_sha256": {p: _sha(Path(p)) for p in code_paths}})
    results = []
    for number, (name, method) in enumerate(tasks, 1):
        include = method == "old-plus-cover"
        make = lambda instance, initial: make_pool(instance, initial, labels, include)
        rank = lambda instance, initial, candidates: _rank(
            instance, initial, candidates, "tabular", 42, cfg["primary_repair_budget_ms"],
            model, cfg)
        result = search_rounds(instances[name], make, rank, total_ms=cfg["total_ms"],
                               repair_ms=cfg["primary_repair_budget_ms"],
                               global_reserve_ms=cfg["global_reserve_ms"],
                               return_reserve_ms=cfg["return_reserve_ms"], workers=1, seed=42)
        saved = output / f"{name}-{method}.json"
        _save(saved, result)
        review = audit(instances[name], json.loads(saved.read_bytes()), labels, include, cfg)
        row = {"instance": name, "method": method, "initial": result["initial"]["makespan"],
               "best": result["schedule"]["makespan"], "rounds": len(result["rounds"]),
               "accepted_rounds": sum(r.get("accepted", False) for r in result["rounds"]),
               "elapsed_ms": result["elapsed_ms"], "stop": result["stop"],
               "delivered_on_time": result["delivered_on_time"], **review}
        results.append(row)
        print(f"{number}/{len(tasks)} {row}", flush=True)
    _save(output / "summary.json", {"results": results,
                                   "files_sha256": {p.name: _sha(p) for p in sorted(output.glob("*.json"))}})
    return int(any(r["stop"] == "ERROR" for r in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    raise SystemExit(run(parser.parse_args().output))
