"""Hidden search child for the independent deadline/delivery controller."""

import json
from pathlib import Path
import sys
from time import perf_counter

from delivery_guard import CandidatePublisher, atomic_json


def direct_environment():
    """Keep the pure-Python policy free of an unused Torch startup cost."""
    import ctypes
    import os
    import platform
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    affinity = None
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError("cannot inspect direct-worker affinity")
        selected = process_mask.value & -process_mask.value
        if not kernel.SetProcessAffinityMask(handle, selected):
            raise OSError("cannot pin direct worker")
        affinity = {"original_mask": process_mask.value, "selected_mask": selected,
                    "logical_processor": selected.bit_length() - 1}
    return {"python": platform.python_version(), "platform": platform.platform(), "affinity": affinity,
            "torch_loaded": "torch" in sys.modules, "runtime": "Python standard library"}


def main(request_path):
    request = json.loads(Path(request_path).read_bytes())
    output = Path(request["output"])
    spec = request["spec"]
    from direct_search import DIRECT_MODES
    if spec["method"] in DIRECT_MODES:
        environment = direct_environment()
    else:
        from experiments.v4_active_baseline import configure_cpu, execute
        environment = configure_cpu()
    atomic_json(output / "worker_ready.json", {"environment": environment,
                "ready_ms": (perf_counter() - request["started"]) * 1000})
    publisher = CandidatePublisher(output / "worker", request["started"])
    if spec["method"] in DIRECT_MODES:
        from baseline import load_pinned_json
        from direct_search import search_segments
        from fjsp import parse_instance
        instance = parse_instance(load_pinned_json(Path(request["input_path"]), request["case"]["sha256"]))
        (output / "segments").mkdir()

        def save_segment(record):
            compact = {k:v for k,v in record.items() if k not in ("schedule", "working_schedule")}
            compact.update(best_makespan=record["schedule"]["makespan"],
                           working_makespan=record["working_schedule"]["makespan"])
            atomic_json(output / "segments" / f"segment-{record['index']:06d}.json", compact)

        result = search_segments(instance, request["initial"], deadline=request["worker_deadline"],
                    seed=spec["seed"], batch_size=spec.get("batch_size",32), segment_steps=spec.get("segment_steps",20),
                    mode=DIRECT_MODES[spec["method"]], on_incumbent=publisher, on_segment=save_segment,
                    retain_segments=False, trial_started=request["started"])
    else:
        result = execute(Path(request["input_path"]), request["case"], spec["method"], spec["seed"],
                         spec["config"], spec["base_config"], output / "worker" / "final.json",
                         trial_started=request["started"], trial_deadline=request["worker_deadline"],
                         prepared_initial=request["initial"], on_incumbent=publisher)
    atomic_json(output / "worker_result.json", result)
    if result["stop"] == "ERROR":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main(sys.argv[1])
