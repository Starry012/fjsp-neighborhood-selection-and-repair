"""Measure bounded blocker chains against random extensions of the same pair."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
from time import perf_counter

from baseline import load_pinned_json, solve_ect
from blocking_chains import grow_blocking_chain
from fjsp import parse_instance, validate_schedule
from neighborhoods import select_release_pairs
from repair import repair_schedule


def _plans(instance, initial, config):
    seed = config["seed"]
    caps = config["size_caps"]
    pairs = select_release_pairs(instance, initial, max_pairs=config["max_anchors_per_instance"], seed=seed)
    flexible = {op for op, machines in instance.options.items() if len(machines) > 1}
    plans = []
    for pair in pairs.pairs:
        links = grow_blocking_chain(instance, initial, pair.anchor, max_size=max(caps))
        chain = (pair.anchor, *(link.blocker for link in links))
        if not links or (chain[1], links[0].target_machine, links[0].overlap) != (
            pair.blocker, pair.target_machine, pair.overlap
        ):
            raise ValueError("chain must start with the original blocker rule")
        controls = sorted(flexible - set(chain))
        size = min(len(chain), len(controls) + 2)
        rng = Random(f"extension:{seed}:{instance.name}:{pair.anchor}")
        random_chain = (*chain[:2], *rng.sample(controls, size - 2))
        groups = [{"size": k, "chain": sorted(chain[:k]), "random": sorted(random_chain[:k])}
                  for k in sorted({min(cap, size) for cap in caps} - {2})]
        plans.append({"anchor": pair.anchor, "base": sorted(chain[:2]), "links": [asdict(link) for link in links],
                      "chain_length": len(chain), "groups": groups})
    return plans


def _measure(instance, initial, released, budget, repeat, config):
    started = perf_counter()
    outer_deadline = started + budget / 1000
    row = {"instance": instance.name, "released": released, "budget_ms": budget, "repeat": repeat}
    try:
        r = repair_schedule(instance, initial, released, outer_deadline - config["return_reserve_ms"] / 1000,
                            workers=config["workers"], seed=config["seed"])
    except Exception as error:
        return {**row, "status": "ERROR", "detail": f"{type(error).__name__}: {error}",
                "gain": None, "outer_seconds": perf_counter() - started}
    finished = perf_counter()
    late = finished > outer_deadline
    candidate = None if r.schedule is None else r.schedule["makespan"]
    gain = None
    if r.status in {"OPTIMAL", "FEASIBLE", "UNKNOWN", "TIME_LIMIT"}:
        gain = r.initial_makespan - candidate if candidate is not None and not late else 0
    return {**row, **asdict(r), "outer_seconds": finished - started, "outer_budget_exceeded": late,
            "returned_makespan": candidate, "gain": gain}


def run_pilot(config_path, output):
    config = json.loads(config_path.read_bytes())
    source = load_pinned_json(Path(config["source_config"]), config["source_config_sha256"])
    if sha256(Path(config["archive_path"]).read_bytes()).hexdigest() != source["archive_sha256"]:
        raise ValueError("archive fingerprint mismatch")
    pins = dict(line.split("==") for line in Path("requirements.txt").read_text().splitlines())
    if platform.python_version() != source["python"] or any(version(k) != v for k, v in pins.items()):
        raise ValueError("use the pinned Python and package versions")
    by_name = {c["instance"]: c for c in source["cases"]}
    output.mkdir(parents=True)
    tracked = (config_path, Path(__file__), *(Path(name) for name in (
        "blocking_chains.py", "neighborhoods.py", "repair.py", "move_scan.py", "baseline.py", "fjsp.py", "requirements.txt"
    )))
    (output / "run.json").write_text(json.dumps({
        "config": config, "archive_sha256": source["archive_sha256"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "packages": pins},
        "sha256": {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in tracked},
    }, indent=2) + "\n", encoding="utf-8")
    failed = False
    with (output / "plans.jsonl").open("x", encoding="utf-8") as plans_log, \
            (output / "labels.jsonl").open("x", encoding="utf-8") as labels_log:
        for name in config["instances"]:
            case = by_name[name]
            instance = parse_instance(load_pinned_json(Path(source["instance_dir"]) / f"{name}.json", case["sha256"]))
            ect = solve_ect(instance, perf_counter() + 2)
            plans = [] if ect.schedule is None else _plans(instance, ect.schedule, config)
            if ect.schedule is None:
                failed = True
            else:
                validate_schedule(instance, ect.schedule)
            plans_log.write(json.dumps({"instance": name, "input_sha256": case["sha256"], "ect_status": ect.status,
                                        "initial": ect.schedule, "plans": plans}) + "\n")
            plans_log.flush()
            releases = {tuple(p["base"]) for p in plans if p["groups"]}
            releases.update(tuple(g[key]) for p in plans for g in p["groups"] for key in ("chain", "random"))
            tasks = list(product(sorted(releases), config["repair_budgets_ms"], range(1, config["repeats"] + 1)))
            Random(f"chain-run:{config['seed']}:{name}").shuffle(tasks)
            for released, budget, repeat in tasks:
                trial = _measure(instance, ect.schedule, released, budget, repeat, config)
                failed |= trial["gain"] is None
                labels_log.write(json.dumps(trial) + "\n")
                labels_log.flush()
            print(f"{name}: {len(plans)} anchors, {len(releases)} distinct neighborhoods, {len(tasks)} repairs", flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--config", type=Path, default=Path("experiments/v3_chain_pilot.json"))
    args = parser.parse_args()
    return run_pilot(args.config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
