"""Prepare frozen inputs and run the two 60-second benchmark batches."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))


def read(path):
    return json.loads(path.read_bytes())


def digest(contents):
    return sha256(contents).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def cases():
    plan = read(ENGINE / "final60-blocks.json")
    return {block["case"]["input"]: block["case"] for group in plan.values() for block in group}


def verify_sources():
    manifest = read(ENGINE / "source-freeze.json")["frozen_files_sha256"]
    for name, expected in manifest.items():
        check(digest((ENGINE / name).read_bytes()) == expected, f"engine file differs: {name}")


def save_input(path, raw, expected):
    check(digest(raw) == expected, f"unexpected frozen input identity: {path.name}")
    if path.exists():
        check(path.read_bytes() == raw, f"existing input differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def prepare(benchmark):
    from experiments.sd2_reference_data import canonical, generate

    verify_sources()
    check(version("numpy") == "2.5.4", "SD2-mix needs NumPy 2.5.4")
    for relative, case in cases().items():
        dest = ENGINE / relative
        if dest.is_file():
            raw = dest.read_bytes()
        elif case["source"] == "sd2mix":
            p = case["provenance"]
            raw = canonical(generate(case["jobs"], p["machines"], p["split"], p["index"], p["attempt"]))
        else:
            raw = (benchmark / "instances/json" / dest.name).read_bytes()
        save_input(dest, raw, case["sha256"])

    bks_path = benchmark / "solutions/bks.json"
    if bks_path.is_file():
        contents = bks_path.read_bytes()
        check(digest(contents) == "b703cb9ba4e6aad9daecbf244fc94e781cb93336cf3948e0bd3aaddfd0e08f9a",
              "optional BKS archive differs from the frozen source")
        names = {entry["instance"] for entry in cases().values() if entry["input"].startswith("inputs/behnke")}
        bks = {row["instance"]: row for row in json.loads(contents) if row["instance"] in names}
        check(bks.keys() == names, "optional BKS archive omits a retained instance")
    else:
        bks = {}
    from experiments.offline_job_state import durable

    durable(ENGINE / "archive-bks.json", bks)
    print(f"Verified {len(cases())} inputs, archived references: {len(bks)}")


def verify_inputs():
    expected = cases()
    for name, case in expected.items():
        path = ENGINE / name
        check(path.is_file() and digest(path.read_bytes()) == case["sha256"],
              f"missing or modified input: {name}; run prepare")
    check((ENGINE / "archive-bks.json").is_file(), "run prepare to register benchmark inputs")
    return len(expected)


def configure_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)


def local_topology():
    from experiments import stop8_shared_cpu as cpu
    from experiments.alns_cpu_lane import topology

    allowed = os.sched_getaffinity(0)
    raw = Path("/proc/cpuinfo").read_text()
    seen = set()
    for logical in sorted(allowed):
        try:
            reference = topology(logical, cpuinfo=raw)
        except (OSError, KeyError, ValueError):
            continue
        siblings = tuple(reference["thread_siblings"])
        if len(siblings) != 2 or siblings in seen:
            continue
        seen.add(siblings)
        comparable, physical = cpu.inventory(allowed, reference)
        if comparable and len(physical) >= 2:
            return reference
    raise ValueError("Benchmark needs a Linux SMT2 core and a separate visible physical core for monitoring")


def run(output, minizinc=None, *, followup=False):
    check(platform.system() == "Linux" and platform.machine() == "x86_64", "use Linux x86_64")
    expected = {"numpy": "2.5.4", "ortools": "9.15.6755",
                "pandas": "3.1.1", "protobuf": "6.33.6", "python-dateutil": "4.0.0"}
    if not followup:
        expected["torch"] = "2.14.1+cu130"
    check(platform.python_version() == "3.14.10", "use CPython 3.14.10")
    for name, pinned in expected.items():
        check(version(name) == pinned, f"{name} differs from frozen environment: expected {pinned}")
    verify_sources()
    verify_inputs()
    configure_threads()

    from experiments.offline_job_state import durable
    from experiments.random_acquisition import exclusive

    topology = local_topology()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "identity.json"
    identity = {"batch": "early-switch" if followup else "final60",
                "manifest_sha256": digest((ENGINE / "source-freeze.json").read_bytes()),
                "archived_bks_sha256": digest((ENGINE / "archive-bks.json").read_bytes())}
    working_lock = output.parent / f".{output.name}.run.lock"
    with exclusive(working_lock):
        check(not (output / "SUCCESS.json").exists(), "output already contains a completed run")
        if receipt.is_file():
            check(read(receipt) == identity, "current engine or archived BKS differs from this run")
        else:
            check(not any(output.iterdir()), "select an empty output directory")
            durable(receipt, identity)
        os.chdir(ENGINE)
        if followup:
            from experiments import early_switch_job

            return early_switch_job.run(ENGINE, output, topology=topology)
        from experiments import final60_job

        return final60_job.run(ENGINE, output, minizinc=minizinc, topology=topology)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ready = commands.add_parser("prepare", help="verify local Behnke files and generate frozen SD2-mix inputs")
    ready.add_argument("--benchmarks", required=True, type=Path, help="local flexible-jobshop directory")
    commands.add_parser("check", help="verify code and all prepared inputs")
    evaluate = commands.add_parser("run", help="run or resume the original nine methods")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--minizinc", required=True, type=Path, help="path to MiniZinc 2.11.0 executable")
    followup = commands.add_parser("run-early-switch", help="run or resume the four 5 s / 10 s handoffs")
    followup.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.benchmarks.resolve())
    elif args.command == "check":
        verify_sources()
        print(f"Verified code and {verify_inputs()} frozen inputs")
    elif args.command == "run":
        return run(args.output.resolve(), args.minizinc.resolve())
    else:
        return run(args.output.resolve(), followup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
