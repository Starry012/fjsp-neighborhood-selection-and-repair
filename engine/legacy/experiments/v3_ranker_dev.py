"""Train three CPU rankers and evaluate on the separate development family."""

import argparse
from collections import defaultdict
from hashlib import file_digest, sha256
from itertools import combinations, product
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

import torch

from baseline import load_pinned_json
from experiments.v3_dev_replay import _prefix
from fjsp import parse_instance
from v3_ranker import GraphRanker, TabularRanker, graph_input, tabular_input


METHODS = ("graph", "graph-no-budget", "tabular")


def _sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(source, config):
    manifest = json.loads((source / "run.json").read_bytes())
    upstream = manifest["config"]
    upstream_hash = config["label_config_sha256"]
    if manifest["sha256"].get("experiments/v3_dev_replay.json") != upstream_hash:
        raise ValueError("label run does not use the pinned development configuration")
    if upstream != load_pinned_json(Path("experiments/v3_dev_replay.json"), upstream_hash):
        raise ValueError("label run configuration differs from the pinned development configuration")
    summary = json.loads((source / "summary.json").read_bytes())
    for name in ("plan.json", "labels.jsonl", "replay.json"):
        if _sha(source / name) != summary["output_sha256"][name]:
            raise ValueError(f"label artifact fingerprint mismatch: {name}")

    plans = json.loads((source / "plan.json").read_bytes())
    pinned = {p["instance"]: p for p in upstream["cases"]}
    if {p["instance"] for p in plans} != pinned.keys() or len(plans) != len(pinned):
        raise ValueError("unexpected cases in development plan")
    budgets = upstream["repair_budgets_ms"]
    instances, releases = {}, {}
    expected = set()
    for p in plans:
        name = p["instance"]
        if any(p[k] != value for k, value in pinned[name].items()):
            raise ValueError(f"{name}: case differs from locked development list")
        raw = load_pinned_json(Path(upstream["instance_dir"]) / f"{name}.json", p["sha256"])
        instances[name] = parse_instance(raw)
        releases[name] = [tuple(c["released"]) for c in p["candidates"]]
        expected.update((name, budget, repeat, release) for release, budget, repeat in product(
            releases[name], budgets, range(1, p["repeats"] + 1)))
    results = defaultdict(dict)
    with (source / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["instance"], row["budget_ms"], row["repeat"])
            release = tuple(row["released"])
            expected.remove((*key, release))
            if row["gain"] is None:
                raise ValueError(f"invalid label from {key}: {row['status']}")
            results[key][release] = (row["outer_seconds"] * 1000, row["gain"])
    if expected:
        raise ValueError(f"missing {len(expected)} measured labels")
    return plans, instances, releases, budgets, results, summary["output_sha256"]


def _fit(model, batches, budgets, cfg, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    shuffle = Random(seed)
    indices = list(range(len(batches)))
    history = []
    started = perf_counter()
    for _ in range(cfg["epochs"]):
        shuffle.shuffle(indices)
        losses = []
        for index in indices:
            inputs, target = batches[index]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs, budgets), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(sum(losses) / len(losses))
    model.eval()
    return {"training_seconds": perf_counter() - started, "epoch_mse": history}


@torch.inference_mode()
def _dispatch(model, make_input, instance, initial, releases, budget, repeats):
    elapsed = []
    for _ in range(repeats):
        started = perf_counter()
        inputs = make_input(instance, initial, releases)
        scores = model(inputs, budget).squeeze(1).tolist()
        order = sorted(range(len(releases)), key=lambda i: -scores[i])
        elapsed.append((perf_counter() - started) * 1000)
    return order, scores, elapsed


def _evaluate(model, make_input, valid, instances, releases, budgets, results, cfg):
    rows = []
    for p, budget in product(valid, budgets):
        name = p["instance"]
        order, scores, elapsed = _dispatch(model, make_input, instances[name], p["initial"],
                                          releases[name], torch.tensor([budget], dtype=torch.float32),
                                          cfg["inference_repeats"])
        repeat_data = [results[(name, budget, r)] for r in range(1, p["repeats"] + 1)]
        actual = [median(data[release][1] for data in repeat_data) for release in releases[name]]
        comparable = correct = 0
        for a, b in combinations(range(len(order)), 2):
            if actual[a] == actual[b]:
                continue
            comparable += 1
            product_sign = (actual[a] - actual[b]) * (scores[a] - scores[b])
            correct += 1 if product_sign > 0 else 0.5 if scores[a] == scores[b] else 0
        order = [releases[name][i] for i in order]
        remaining = cfg["replay_total_ms"] - p["prep_ms"] - median(elapsed)
        replay = [_prefix(order, data, remaining) for data in repeat_data]
        rows.append({"instance": name, "budget_ms": budget, "dispatch_ms": elapsed, "order": order,
                     "scores_in_candidate_order_pct": scores, "strict_pairs": comparable,
                     "correct_pair_credits": correct, "prefix": replay})
    return rows


def run(source, config_path, output):
    cfg = json.loads(config_path.read_bytes())
    if platform.python_version() != cfg["python"] or torch.__version__ != cfg["torch"]:
        raise ValueError("use the pinned Python and PyTorch builds")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    plans, instances, releases, budgets, results, source_hashes = _read(source, cfg)
    train = [p for p in plans if p["split"] == "train-development" and releases[p["instance"]]]
    valid = [p for p in plans if p["split"] == "validation-development"]
    budget_input = torch.tensor(budgets, dtype=torch.float32)
    targets = {}
    for p in train:
        name = p["instance"]
        scale = p["initial"]["makespan"]
        targets[name] = torch.tensor([[100 * results[name, b, 1][release][1] / scale for b in budgets]
                                      for release in releases[name]], dtype=torch.float32)
    graph_batches = []
    tabular_batches = []
    for p in train:
        name = p["instance"]
        graph_batches.append((graph_input(instances[name], p["initial"], releases[name]), targets[name]))
        tabular_batches.append((tabular_input(instances[name], p["initial"], releases[name]), targets[name]))
    tabular_features = torch.cat([x for x, _ in tabular_batches])

    output.mkdir(parents=True)
    code = (config_path, Path(__file__), Path("v3_ranker.py"), Path("fjsp.py"), Path("move_scan.py"),
            Path("baseline.py"), Path("experiments/v3_dev_replay.py"))
    _save(output / "run.json", {"config": cfg, "platform": platform.platform(), "processor": platform.processor(),
                               "source_hashes": source_hashes, "code_sha256": {p.as_posix(): _sha(p) for p in code},
                               "train_cases": [p["instance"] for p in train],
                               "validation_cases": [p["instance"] for p in valid],
                               "training_targets": sum(t.numel() for t in targets.values())})
    for seed, method in product(cfg["seeds"], METHODS):
        torch.manual_seed(seed)
        if method == "tabular":
            model = TabularRanker(cfg["width"], tabular_features)
            batches, make_input = tabular_batches, tabular_input
        else:
            model = GraphRanker(cfg["width"], use_budget=method == "graph")
            batches, make_input = graph_batches, graph_input
        training = _fit(model, batches, budget_input, cfg, seed)
        validation = _evaluate(model, make_input, valid, instances, releases, budgets, results, cfg)
        filename = f"{method}-{seed}"
        _save(output / f"{filename}.json", {"method": method, "seed": seed, "training": training,
                                            "validation": validation})
        torch.save(model.state_dict(), output / f"{filename}.pt")
        print(f"{filename}: trained in {training['training_seconds']:.2f} s", flush=True)
    artifacts = [p for p in output.iterdir() if p.is_file()]
    _save(output / "artifacts.json", {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("outputs/v3-dev-replay"))
    parser.add_argument("--config", type=Path, default=Path("experiments/v3_ranker_dev.json"))
    args = parser.parse_args()
    run(args.source, args.config, args.output)
