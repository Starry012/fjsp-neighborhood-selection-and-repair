"""Small GPU qualification; only an explicitly identified RTX 5090 is used."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


FIELDS = ("index", "uuid", "name", "memory.total", "memory.used", "memory.free", "utilization.gpu")


def parse_gpus(raw):
    rows = []
    for cells in csv.reader(raw.splitlines()):
        if not cells:
            continue
        if len(cells) != len(FIELDS):
            raise ValueError("Unexpected nvidia-smi columns")
        record = dict(zip(FIELDS, (cell.strip() for cell in cells)))
        record["index"] = int(record["index"])
        for field in FIELDS[3:]:
            record[field] = int(record[field]) if record[field].isdigit() else None
        rows.append(record)
    if len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate GPU UUIDs")
    return rows


def choose_gpu(rows):
    allowed = [row for row in rows if row["name"] == "NVIDIA GeForce RTX 5090"]
    if len(allowed) != 4:
        raise ValueError("Expected four RTX 5090 devices; stop and report current enumeration")
    candidates = [row for row in allowed if row["memory.free"] is not None
                  and row["memory.free"] >= 4096
                  and row["memory.used"] is not None and row["memory.used"] <= 1024
                  and row["utilization.gpu"] is not None and row["utilization.gpu"] <= 10]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (-row["memory.free"], row["index"]))[0]


def memory_snapshot(path=Path("/proc/meminfo")):
    wanted = {"MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            values[key + "_bytes"] = int(value.split()[0]) * 1024
    return values


def inspect_resources():
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    return {"time_utc": datetime.now(timezone.utc).isoformat(),
            "host_memory": memory_snapshot(), "gpus": parse_gpus(result.stdout)}


def qualify(output):
    output = Path(output)
    report_path = output / "gpu-check.json"
    if report_path.exists():
        raise ValueError("GPU report already exists; do not overwrite")
    report = {"status": "STARTED", "scope": "Small numeric/autograd qualification, not FJSP training"}

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError("Existing CUDA_VISIBLE_DEVICES restriction: preserve it and report before proceeding")
        report["before"] = inspect_resources()
        selected = choose_gpu(report["before"]["gpus"])
        report["selected"] = selected
        if selected is None:
            report["status"] = "SKIPPED_BUSY"
            save()
            print("FJSP_GPU_CHECK_SKIPPED_BUSY: no suitable idle RTX 5090; no GPU calculation performed")
            return 0
        os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # CUDA visibility is restricted before any Torch import.
        import numpy as np
        import torch
        expected = Path(".benchmark-work/envs/fjsp-train")
        if Path(sys.prefix).resolve() != expected:
            raise ValueError("Wrong Python environment")
        if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0" or np.__version__ != "2.5.4":
            raise ValueError("GPU framework versions differ from the pinned configuration")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expected exactly one visible, available GPU")
        if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
            raise ValueError("Unexpected device; refusing calculations")
        torch.cuda.reset_peak_memory_stats(0)
        report["framework"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                               "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                               "capability": list(torch.cuda.get_device_capability(0)),
                               "compiled_architectures": torch.cuda.get_arch_list()}
        # Small deterministic matrix forward/backward versus the same CPU calculation.
        def matrix(device):
            x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, device=device, requires_grad=True)
            w = torch.tensor([[2., -1.], [.5, 3.]], dtype=torch.float64, device=device, requires_grad=True)
            loss = (x @ w).square().sum()
            loss.backward()
            return loss.detach().cpu(), x.grad.detach().cpu(), w.grad.detach().cpu()
        cpu, gpu = matrix("cpu"), matrix("cuda:0")
        for expected_value, actual in zip(cpu, gpu):
            torch.testing.assert_close(actual, expected_value, rtol=1e-9, atol=1e-9)
        # Native graph-style gather/aggregation and its gradient.
        def aggregate(device):
            x = torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3).requires_grad_()
            source = torch.tensor([0, 2, 2, 3], device=device)
            target = torch.tensor([1, 1, 3, 0], device=device)
            result = torch.zeros_like(x).index_add(0, target, x[source])
            result.square().sum().backward()
            return result.detach().cpu(), x.grad.detach().cpu()
        for expected_value, actual in zip(aggregate("cpu"), aggregate("cuda:0")):
            torch.testing.assert_close(actual, expected_value, rtol=1e-6, atol=1e-6)
        array = np.arange(6, dtype=np.float32)
        np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
        torch.cuda.synchronize(0)
        report["checks"] = {"matrix_and_gradient": True, "index_add_and_gradient": True,
                            "numpy_interop": True}
        report["tensor_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["after"] = inspect_resources()
        report["status"] = "PASSED"
        save()
        print("Selected RTX 5090 index:", selected["index"])
        print("Selected RTX 5090 UUID:", selected["uuid"])
        print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
        print("Tensor allocation peak MiB:", round(report["tensor_peak_allocated_bytes"] / 1024**2, 3))
        print("FJSP_SINGLE_GPU_CHECK_OK")
        return 0
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(qualify(args.output))
