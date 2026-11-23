"""Fixed CPU forward-cost measurement; no optimizer or CP solve."""

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from random import Random
import shutil
from statistics import median
from time import perf_counter
import traceback


CONFIG = Path("experiments/conditional_model_probe.json")
SOURCES = [
    str(CONFIG), "experiments/conditional_model_probe.py", "conditional_model.py", "conditional_selector.py",
    "conditional_repair_feedback.py", "v3_ranker.py", "joint_selection.py", "fjsp.py", "repair.py",
    "retained_bounds.py", "move_scan.py", "experiments/resched_qualification.py",
    "tests/test_conditional_model.py", "tests/test_conditional_selector.py",
    "tests/test_conditional_repair_feedback.py", "tests/test_v3_ranker.py", "requirements-py314.txt",
]


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pinned(path, fingerprint):
    if digest(path) != fingerprint:
        raise ValueError(f"input fingerprint changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_hash(model):
    result = sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def storage_bytes(tensors):
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    return sum(storages.values())


def run():
    from experiments.resched_qualification import configure_cpu
    affinity = configure_cpu()
    import psutil
    import torch
    from baseline import load_pinned_json
    from conditional_model import BoundConditionalPolicy, ConditionalGraphPolicy, build_policy_graph
    from conditional_selector import ConditionalSelector, SelectionState
    from fjsp import parse_instance, validate_schedule

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["model"]["seed"])
    try:
        source_hashes = {}
        for filename in SOURCES:
            target = root / "source" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, target)
            source_hashes[filename] = digest(target)
        save(root / "plan.json", cfg)
        save(root / "source-hashes.json", source_hashes)
        case_config = pinned(Path(cfg["case_config"]), cfg["case_config_sha256"])
        cases = {case["instance"]: case for case in case_config["cases"]}
        process = psutil.Process()
        boot = perf_counter()
        model = ConditionalGraphPolicy(cfg["model"]["width"], cfg["model"]["layers"]).eval()
        model_setup_ms = (perf_counter() - boot) * 1000
        initial_hash = parameter_hash(model)
        torch.save(model.state_dict(), root / "untrained-initial-state.pt")
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        records = []
        for name in cfg["cases"]:
            case = cases[name]
            instance = parse_instance(load_pinned_json(Path(case["input"]), case["sha256"]))
            anchor_path = Path(cfg["anchor_root"]) / f"{name}.json"
            anchor = pinned(anchor_path, cfg["anchor_sha256"][name])
            assert validate_schedule(instance, anchor) == anchor["makespan"]
            directory = root / name
            directory.mkdir()
            shutil.copyfile(case["input"], directory / "instance.json")
            shutil.copyfile(anchor_path, directory / "anchor.json")
            samples = [process.memory_info().rss]
            started = perf_counter()
            graph = build_policy_graph(instance, anchor)
            first_graph_ms = (perf_counter() - started) * 1000
            with torch.inference_mode():
                started = perf_counter()
                encoded = model.encode(graph)
                first_encoding_ms = (perf_counter() - started) * 1000
                assert torch.isfinite(encoded.operations).all()
            graph_times, encode_times = [], []
            for _ in range(cfg["timing_repeats"]):
                started = perf_counter()
                repeated_graph = build_policy_graph(instance, anchor)
                graph_times.append((perf_counter() - started) * 1000)
                with torch.inference_mode():
                    started = perf_counter()
                    repeated_encoding = model.encode(repeated_graph)
                    encode_times.append((perf_counter() - started) * 1000)
                samples.append(process.memory_info().rss)
            order = list(graph.operation_ids)
            Random(101).shuffle(order)
            n = len(order)
            prefixes = {}
            for size in cfg["prefix_sizes"]:
                k = n if size == "all" else size
                times = []
                for _ in range(cfg["timing_repeats"]):
                    with torch.inference_mode():
                        started = perf_counter()
                        output = model.score_encoded(encoded, order[:k])
                        times.append((perf_counter() - started) * 1000)
                        assert torch.isfinite(output.operations).all() and torch.isfinite(output.stop)
                    samples.append(process.memory_info().rss)
                prefixes[str(k)] = {"median_ms": median(times), "min_ms": min(times), "max_ms": max(times)}
            bind_started = perf_counter()
            bound = BoundConditionalPolicy(model, graph)
            bind_ms = (perf_counter() - bind_started) * 1000
            trajectory_times = []
            started = perf_counter()
            for k in range(n + 1):
                state = SelectionState(graph.anchor_sha256, tuple(order[:k]), (), None)
                entered = perf_counter()
                bound(state)
                trajectory_times.append((perf_counter() - entered) * 1000)
                samples.append(process.memory_info().rss)
            trajectory_ms = (perf_counter() - started) * 1000
            selector = ConditionalSelector(instance, anchor)
            selections = []
            for _ in range(2):
                entered = perf_counter()
                proposal = selector.choose(
                    bound, deadline=entered + cfg["selector_deadline_seconds"],
                    max_states=cfg["selector_max_states"])
                selections.append(asdict(proposal))
                samples.append(process.memory_info().rss)
            ready = [row["released"] for row in selections if row["released"] is not None]
            assert len({tuple(group) for group in ready}) == len(ready)
            info = process.memory_info()
            cache_bytes = storage_bytes([encoded.operations, encoded.global_state, encoded.queries,
                                         encoded.keys, encoded.values])
            record = {
                "instance": name, "operations": n, "anchor_Cmax": anchor["makespan"],
                "input_sha256": case["sha256"], "anchor_sha256": cfg["anchor_sha256"][name],
                "nodes": len(graph.graph.nodes),
                "edges": sum(len(links[0]) for links in graph.graph.edges.values()),
                "first_graph_ms": first_graph_ms, "first_encoding_ms": first_encoding_ms,
                "warm_graph_median_ms": median(graph_times), "warm_encoding_median_ms": median(encode_times),
                "prefix_scoring": prefixes, "inference_binding_ms": bind_ms,
                "complete_additive_timing_trajectory": {
                    "prefixes": n + 1, "order_seed": 101, "wall_ms_including_rss_observations": trajectory_ms,
                    "sum_callback_ms": sum(trajectory_times), "median_callback_ms": median(trajectory_times),
                    "max_callback_ms": max(trajectory_times),
                },
                "selector_calls": selections, "distinct_ready_groups": len(ready),
                "rss_before_bytes": samples[0], "observed_rss_max_bytes": max(samples),
                "process_peak_working_set_bytes": getattr(info, "peak_wset", None),
                "encoded_tensor_storage_bytes": cache_bytes,
                "parameter_bytes": parameter_bytes, "gpu_memory_bytes": None,
            }
            save(directory / "measurement.json", record)
            records.append(record)
            print(json.dumps({"instance": name, "encoding_ms": record["warm_encoding_median_ms"],
                              "trajectory_ms": trajectory_ms,
                              "selector_statuses": [row["status"] for row in selections]}, ensure_ascii=False),
                  flush=True)
        final_hash = parameter_hash(model)
        assert final_hash == initial_hash and all(parameter.grad is None for parameter in model.parameters())
        summary = {
            "protocol": cfg["protocol"], "python": platform.python_version(), "torch": torch.__version__,
            "numpy": version("numpy"), "device": "cpu", "cuda_available": torch.cuda.is_available(),
            "affinity": affinity, "model_setup_ms": model_setup_ms,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_sha256": initial_hash, "weights_unchanged": final_hash == initial_hash,
            "optimizer_steps": 0, "cp_calls": 0, "cases": records,
        }
        save(root / "summary.json", summary)
    except Exception:
        save(root / "error.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    run()
