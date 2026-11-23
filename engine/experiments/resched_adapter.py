"""Adapt local chain instances to unchanged, pinned ReSched greedy inference."""

import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
import sys
from time import perf_counter

from fjsp import validate_schedule


def verify_snapshot(snapshot, expected_manifest_sha256):
    snapshot = Path(snapshot)
    raw = (snapshot / "acquisition.json").read_bytes()
    if sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("acquisition manifest changed")
    manifest = json.loads(raw)
    if not manifest["complete"]:
        raise ValueError("source acquisition is incomplete")
    for row in manifest["files"]:
        if sha256((snapshot / "upstream" / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"upstream file changed: {row['path']}")
    return manifest


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(snapshot, manifest_sha256, seed):
    started = perf_counter()
    manifest = verify_snapshot(snapshot, manifest_sha256)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    source = Path(snapshot) / "upstream"
    tree = ast.parse((source / "REINFORCE/configs/fjsp.py").read_text(encoding="utf-8"))
    params = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == "model_params" for target in node.targets))
    model_module = import_source("_fjsp_resched_original", source / "ckpt/SchedulingModel_old.py")
    sys.path.insert(0, str(source / "REINFORCE"))
    try:
        environment_module = import_source("_fjsp_resched_environment", source / "REINFORCE/SchedulingEnvironment.py")
    finally:
        sys.path.pop(0)
    model = model_module.Model(**params).cpu().eval()
    state = torch.load(source / "ckpt/REINFORCE/FJSP/Old-SD1-10x05.pth",
                       map_location="cpu", weights_only=True)
    matching = model.load_state_dict(state, strict=True)
    model.set_decode_type("greedy")
    return {
        "model": model, "module": model_module, "environment_module": environment_module,
        "load_info": {"revision": manifest["revision"], "model_params": params,
                      "state_tensors": len(state), "parameters": sum(p.numel() for p in model.parameters()),
                      "missing_keys": list(matching.missing_keys), "unexpected_keys": list(matching.unexpected_keys),
                      "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                      "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                      "load_ms": (perf_counter() - started) * 1000},
    }


def encode_instance(instance):
    import numpy as np

    job_ids = sorted(instance.jobs)
    machine_ids = sorted({m for options in instance.options.values() for m in options})
    machine_dense = {m: i for i, m in enumerate(machine_ids)}
    operation_ids, job_idx, operation_idx, lookup = [], [], [], {}
    for dense_job, job in enumerate(job_ids):
        for local_operation, operation in enumerate(instance.jobs[job]):
            lookup[dense_job, local_operation] = len(operation_ids)
            operation_ids.append(operation)
            job_idx.append(dense_job)
            operation_idx.append(local_operation)
    count = len(operation_ids)
    if count < 2:
        raise ValueError("the unchanged upstream final-step routine requires at least two operations")
    duration = np.zeros((1, count, len(machine_ids)), dtype=np.int64)
    dependency = np.zeros((1, count, count), dtype=bool)
    connection = np.zeros_like(dependency)
    for i, operation in enumerate(operation_ids):
        for machine, value in instance.options[operation].items():
            duration[0, i, machine_dense[machine]] = value
        same_job_later = [j for j in range(i + 1, count) if job_idx[j] == job_idx[i]]
        if same_job_later:
            dependency[0, i, same_job_later[0]] = True
            connection[0, i, same_job_later] = True
    return {"data": (np.array([job_idx]), np.array([operation_idx]), duration, dependency, connection),
            "operation_ids": operation_ids, "machine_ids": machine_ids, "lookup": lookup}


def export_schedule(instance, encoded, env):
    import numpy as np

    assignments = []
    for job, operation, machine in zip(env.solution.job_idx[0], env.solution.operation_idx[0],
                                       env.solution.machine_idx[0], strict=True):
        flat = encoded["lookup"][int(job), int(operation)]
        dense_machine = int(machine)
        start = env.state.start_time[0, flat, dense_machine]
        end = env.state.finish_time[0, flat, dense_machine]
        if not np.isfinite(start) or not np.isfinite(end) or start != int(start) or end != int(end):
            raise ValueError("upstream output is not an integer schedule")
        assignments.append({"operation": encoded["operation_ids"][flat],
                            "machine": encoded["machine_ids"][dense_machine],
                            "start": int(start), "end": int(end)})
    schedule = {"instance": instance.name, "assignments": assignments,
                "makespan": int(env.state.finish_time.max())}
    validate_schedule(instance, schedule)
    # Replay the exported construction sequence independently of tensor indexing.
    predecessor = {after: before for chain in instance.jobs.values() for before, after in zip(chain, chain[1:])}
    finish, machine_free = {}, {}
    for row in assignments:
        operation, machine = row["operation"], row["machine"]
        previous = predecessor.get(operation)
        if previous is not None and previous not in finish:
            raise ValueError("construction sequence violates job order")
        start = max(finish.get(previous, 0), machine_free.get(machine, 0))
        end = start + instance.options[operation][machine]
        if (start, end) != (row["start"], row["end"]):
            raise ValueError("exported times differ from an independent construction replay")
        finish[operation] = machine_free[machine] = end
    return schedule


def construct(runtime, instance, seed, deadline, progress=None):
    import torch

    profile = {"encode_ms": 0.0, "environment_init_ms": 0.0, "preprocess_ms": 0.0,
               "forward_ms": 0.0, "environment_step_ms": 0.0, "export_verify_ms": 0.0}
    started = perf_counter()
    encoded = encode_instance(instance)
    profile["encode_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    env = runtime["environment_module"].FJSPEnv()
    env.set_seed(seed)
    env.load_data(1, *encoded["data"])
    state = env.reset_state()
    profile["environment_init_ms"] = (perf_counter() - started) * 1000
    trace, done = [], False
    with torch.no_grad():
        while not done:
            if perf_counter() >= deadline:
                raise TimeoutError("construction deadline reached before a complete schedule")
            started = perf_counter()
            model_input, _ = runtime["module"].build_model_input(state, env)
            profile["preprocess_ms"] += (perf_counter() - started) * 1000
            started = perf_counter()
            action, probability = runtime["model"](model_input)
            profile["forward_ms"] += (perf_counter() - started) * 1000
            seq, machine = int(action[0].item()), int(action[1].item())
            if not state.action_mask[0, seq, machine]:
                raise ValueError("policy selected an action masked as illegal")
            flat = encoded["lookup"][int(state.job_idx[0, seq]), int(state.operation_idx[0, seq])]
            trace.append({"operation": encoded["operation_ids"][flat],
                          "machine": encoded["machine_ids"][machine], "probability": float(probability.item())})
            started = perf_counter()
            state, _, done = env.step(action)
            profile["environment_step_ms"] += (perf_counter() - started) * 1000
            if progress and (len(trace) % 25 == 0 or done):
                progress({"policy_steps": len(trace), "done": done, "profile": profile})
    started = perf_counter()
    schedule = export_schedule(instance, encoded, env)
    profile["export_verify_ms"] = (perf_counter() - started) * 1000
    return schedule, {"profile": profile, "policy_steps": len(trace), "trace": trace,
                      "automatic_final_operation": schedule["assignments"][-1],
                      "upstream_solution_start_entries": int(env.solution.start_time.size),
                      "upstream_solution_end_entries": int(env.solution.end_time.size)}
