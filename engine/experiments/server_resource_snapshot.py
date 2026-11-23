"""Read-only Linux CPU/GPU telemetry, without importing CUDA or Torch."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter, sleep

from experiments.server_gpu_probe import FIELDS, memory_snapshot, parse_gpus


ROOT = Path(".benchmark-work")


def cpu_counters(raw):
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if parts and (parts[0] == "cpu" or parts[0][3:].isdigit() and parts[0].startswith("cpu")):
            values = [int(v) for v in parts[1:9]]  # guest counters already included in user/nice
            if len(values) != 8 or min(values) < 0:
                raise ValueError("invalid CPU counters")
            result[parts[0]] = values
    if "cpu" not in result:
        raise ValueError("missing aggregate CPU counters")
    return result


def cpu_delta(before, after):
    rows = {}
    for name, end in after.items():
        delta = [b - a for a, b in zip(before.get(name, end), end)]
        total = sum(delta)
        if total <= 0 or min(delta) < 0:
            rows[name] = {"status": "UNKNOWN", "busy_percent": None, "iowait_percent": None}
        else:
            rows[name] = {"status": "MEASURED", "busy_percent": 100 * (total - delta[3] - delta[4]) / total,
                          "iowait_percent": 100 * delta[4] / total}
    return rows


def query_gpu(argument, *, runner=subprocess.run):
    result = runner(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise ValueError(f"nvidia-smi exit {result.returncode}")
    return result.stdout


def process_rows(raw):
    rows = []
    for cells in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not cells:
            continue
        if len(cells) != 4:
            raise ValueError("invalid GPU process columns")
        uuid, pid, name, used = (c.strip() for c in cells)
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name,
                     "used_gpu_memory_MiB": int(used) if used.isdigit() else None})
    return rows


def snapshot(*, phase="manual", proc=Path("/proc"), runner=subprocess.run,
             sleeper=sleep, sample_seconds=.25, affinity_reader=None):
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
              "gpu": {"status": "UNKNOWN", "devices": None}, "cpu": {"status": "UNKNOWN"}}
    try:
        report["gpu"] = {"status": "MEASURED", "devices": parse_gpus(query_gpu(
            "--query-gpu=" + ",".join(FIELDS), runner=runner))}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["gpu"]["reason"] = f"{type(error).__name__}: {error}"
    try:
        report["gpu_processes"] = {"status": "MEASURED", "rows": process_rows(query_gpu(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", runner=runner)),
            "scope": "NVIDIA compute process list only; not proof of exclusive device ownership"}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["gpu_processes"] = {"status": "UNKNOWN", "rows": None, "reason": f"{type(error).__name__}: {error}"}
    try:
        report["host_memory"] = {"status": "MEASURED", **memory_snapshot(proc / "meminfo")}
    except (OSError, ValueError, IndexError) as error:
        report["host_memory"] = {"status": "UNKNOWN", "reason": type(error).__name__}
    try:
        first = cpu_counters((proc / "stat").read_text())
        begin = perf_counter()
        sleeper(sample_seconds)
        rows = cpu_delta(first, cpu_counters((proc / "stat").read_text()))
        report["cpu"] = {"status": rows["cpu"]["status"], "sample_seconds": perf_counter() - begin,
                         "aggregate": rows["cpu"], "logical_cpus": {k: v for k, v in rows.items() if k != "cpu"}}
    except (OSError, ValueError) as error:
        report["cpu"]["reason"] = f"{type(error).__name__}: {error}"
    try:
        values = (proc / "loadavg").read_text().split()
        report["load_average"] = {"status": "MEASURED", "one_five_fifteen": [float(v) for v in values[:3]],
                                  "runnable_total_tasks": values[3]}
    except (OSError, ValueError, IndexError) as error:
        report["load_average"] = {"status": "UNKNOWN", "reason": type(error).__name__}
    try:
        allowed = sorted((affinity_reader or os.sched_getaffinity)(0))
        report["cpu_affinity"] = {"status": "MEASURED", "allowed_logical_cpus": allowed,
                                  "existing_worker_default_cpu": min(allowed)}
    except (AttributeError, OSError, ValueError) as error:
        report["cpu_affinity"] = {"status": "UNKNOWN", "reason": type(error).__name__}
    return report


def save(path, report):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def print_snapshot(report):
    cpu = report["cpu"].get("aggregate", {}).get("busy_percent")
    text = "未知" if cpu is None else f"{cpu:.1f}%"
    print(f"资源[{report['phase']}]：CPU使用率 {text}；负载 {report['load_average'].get('one_five_fifteen', '未知')}", flush=True)
    for row in report["gpu"].get("devices") or []:
        print(f"GPU {row['index']} {row['name']} {row['uuid']}：已用/空闲显存 "
              f"{row['memory.used']}/{row['memory.free']} MiB；使用率 {row['utilization.gpu']}%", flush=True)
    print(f"GPU查询 {report['gpu']['status']}；计算进程查询 {report['gpu_processes']['status']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after", "manual"), default="manual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use Linux and save in personal project runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = snapshot(phase=args.phase)
    save(args.output, report)
    print_snapshot(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
