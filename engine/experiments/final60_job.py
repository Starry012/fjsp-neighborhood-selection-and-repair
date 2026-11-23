"""Final paired 60-second FJSP comparison and separately budgeted references."""

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from random import Random
import subprocess
import sys
from time import perf_counter, sleep
from traceback import format_exc
from uuid import uuid4

from experiments import final_server_qualification as qualification
from experiments import stall80_warmstart_runtime as warm
from experiments import stop8_shared_job as shared
from experiments.final60_worker import check, read
from experiments.offline_job_state import JobState, durable
from fjsp import parse_instance, validate_schedule


NAME = "final-60s-benchmark-20261121"
METHODS = ("Stall35-MinOptions2", "MinOptions2", "Stall80", "Stall20-MinOptions2",
           "Stall35-Tree2WithoutClock", "Tree2WithoutClock", "active-strict",
           "full-cp-sat-hinted", "Chuffed")
TIMES = (5, 10, 20, 35, 50, 60)
SEEDS = (42, 43, 44)
CP_MODES = ("CP-SAT-off", "CP-SAT-on")
ORDER = (0, 1, 8, 2, 7, 3, 6, 4, 5)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def make_plan(cases, development):
    generator = Random(20261121)
    groups = [[case for case in cases if case["stratum"] == (source + str(size))]
              for source in ("behnke", "sd2mix") for size in (250, 500)]
    check(all(len(group) == 12 for group in groups), "expected 12 cases in each stratum")
    generator.shuffle(groups)
    for group in groups:
        generator.shuffle(group)
    cases = [case for group in zip(*groups, strict=True) for case in group]
    presolve = [{"id": f"presolve-{case['id']}-{seed}", "case": case, "seed": seed, "seconds": 60,
                 "arms": list(CP_MODES[::1 if (number + seed) % 2 else -1])}
                for number, case in enumerate(development) for seed in SEEDS]
    preflight = [{"id": f"integration-{case['id']}", "case": case, "seed": 42, "seconds": 60,
                  "arms": [METHODS[0], METHODS[4], METHODS[6]]} for case in development[::7]]
    main = []
    for i, case in enumerate(cases):
        for j, seed in enumerate(SEEDS):
            template = ORDER if (i + j) % 2 == 0 else ORDER[::-1]
            # A reversal changes position, not the nine-arm balance over each cycle.
            offset = (3 * i + j) % len(METHODS)
            order = [METHODS[(position + offset) % len(METHODS)] for position in template]
            main.append({"id": f"main-{case['id']}-{seed}", "case": case, "seed": seed, "seconds": 60, "arms": order})
    reference = [{"id": f"reference-{case['id']}", "case": case, "seed": 42, "seconds": 300, "arms": ["Chuffed"]}
                 for case in cases]
    check(all(max(count.values()) == min(count.values()) == 16 for count in
              (Counter(block["arms"][position] for block in main) for position in range(len(METHODS)))),
          "main arm position imbalance")
    return {"integration": preflight, "presolve": presolve, "main": main, "reference": reference}


def guarded_external(bundle, trial, case, seed, method, seconds, which, core, fds, exe):
    work = trial / "external"
    work.mkdir()
    request_path = work / "request.json"
    started = perf_counter()
    request_path.write_text(json.dumps({"case": case, "seed": seed, "method": method,
        "started": started, "seconds": seconds, "presolve": which, "minizinc": str(exe),
        "input": str(bundle / case["input"]), "cpu": core, "parent": os.getpid()}), encoding="utf-8")
    with (work / "worker.log").open("wb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-m", "experiments.final60_worker", str(request_path)],
                                 cwd=bundle, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 pass_fds=fds, start_new_session=True)
        code = None
        timed_out = False
        try:
            code = child.wait(timeout=max(.001, started + seconds - .1 - perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out = True
            code = child.poll()
        finally:
            qualification.reap_group(child)
    end = perf_counter()
    durable(work / "worker-exit.json", {"wait_timed_out": timed_out, "code_before_reap": code,
            "reaped_returncode": child.returncode, "reaped_ms": (end - started) * 1000})
    return started, end, code


def audit_external(bundle, trial, case, started, end, code, seconds):
    work = trial / "external"
    instance = parse_instance(shared.cost.pinned(bundle / case["input"], case["sha256"]))
    frames = work / "frames.jsonl"
    trace = []
    error = None
    try:
        for line in frames.read_text(encoding="utf-8").splitlines():
            frame = json.loads(line)
            path = work / f"frame-{len(trace):04d}.json"
            check(path.name == frame["file"] and digest(path) == frame["sha256"], "delivery frame identity differs")
            verified = validate_schedule(instance, read(path))
            check(verified == frame["Cmax"] and 0 <= frame["ms"] <= seconds * 1000, "late or invalid frame")
            check(not trace or (trace[-1]["Cmax"] > verified and trace[-1]["ms"] <= frame["ms"]),
                  "delivery trace order differs")
            trace.append(frame)
    except (OSError, ValueError, KeyError) as exc:
        error = str(exc)
    path = work / "worker-result.json"
    try:
        receipt = read(path) if path.is_file() else {}
    except (ValueError, OSError) as exc:
        error, receipt = str(exc), {}
    native = receipt if code == 0 else {}
    score = trace[-1]["Cmax"] if trace else None
    bound = native.get("lower_bound")
    okay = bool(trace and native and end <= started + seconds and error is None and
                (bound is None or bound <= score + 1e-6))
    return {"engineering_passed": okay, "credited_Cmax": score, "wall_ms": (end - started) * 1000,
            "error": None if okay else (error or f"child_exit={code}; inspect external/worker.log"),
            "status": native.get("status"), "native_schedules_validated": native.get("native_schedules_validated"),
            "worker_reported_ms": receipt.get("last_ms"), "worker_exit": read(work / "worker-exit.json"),
            "global_bound": bound if okay else None, "ECT": trace[0]["Cmax"] if trace else None,
            "checkpoints": {str(t): min((f["Cmax"] for f in trace if f["ms"] <= t * 1000), default=None)
                            for t in (*TIMES, seconds)},
            "final_frame": trace[-1] if trace else None}, trace


def measure(bundle, trial, case, seed, method, seconds, cfg, settings, hardware, monitor, fds, binary, exe):
    trial.mkdir()
    core = hardware["logical_processor"]
    sibling, = set(hardware["thread_siblings"]) - {core}
    cpus = [core, sibling]
    monitor_path = trial / "monitor"
    before = shared.cgroup_cpu()
    with shared.cpu.bind(core), shared.cpu.watch(monitor_path, monitor, cpus):
        own_start = shared.cpu.own_cpu_seconds()
        started = perf_counter()
        guarded = ext = None
        error = None
        try:
            if method in warm.ARMS:
                guarded = warm.run_trial(bundle, trial, case, method, seed, cfg, binary, core, fds)
            else:
                ext = guarded_external(bundle, trial, case, seed, method, seconds,
                                       method == "CP-SAT-on" or (method == METHODS[7] and settings["presolve"]),
                                       core, fds, exe)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = str(exc)
        end = perf_counter()
        own = shared.cpu.own_cpu_seconds() - own_start
        # A solver can finish or prove optimality before one monitor window.
        # Keep observing the retained core lease for a complete window.
        observed_end = max(end, started + 1.8)
        sleep(max(0, observed_end - perf_counter()))
    after = shared.cgroup_cpu()
    row = {"engineering_passed": False, "error": error, "credited_Cmax": None}
    trace, queue = [], None
    try:
        if guarded is not None:
            started, end = guarded["started_ns"] / 1e9, guarded["started_ns"] / 1e9 + guarded["wall_ms"] / 1000
            row, trace, queue = warm.audit_trial(bundle, trial, case, method, seed, cfg, guarded, core)
            row["credited_Cmax"] = row["checkpoints"]["60"]
        if ext is not None:
            started, end, code = ext
            row, trace = audit_external(bundle, trial, case, started, end, code, seconds)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        row["error"] = str(exc)
    samples_path = monitor_path / "cpu-samples.json"
    samples = read(samples_path) if samples_path.exists() else {"cpus": cpus, "samples": []}
    check(samples["cpus"] == cpus and read(monitor_path / "monitor-ready.json")["affinity"] == [monitor],
          "CPU monitor affinity differs")
    monitor_end = max(end, observed_end)
    cpu = shared.cpu.summarize(samples["samples"], cpus, started, monitor_end, own, queue)
    cpu["post_trial_observation_seconds"] = monitor_end - end
    durable(trial / "anytime.json", trace)
    durable(trial / "cpu-review.json", {**cpu, "cgroup_before": before, "cgroup_after": after})
    return {**row, "method": method, "instance": case["instance"], "seed": seed,
            "input_sha256": case["sha256"], "wall_ms": (end - started) * 1000,
            "cpu": {**{k: v for k, v in cpu.items() if k != "windows"}, "selected_cpu": core,
                    "sibling_cpu": sibling, "monitor_cpu": monitor},
            "anytime_sha256": digest(trial / "anytime.json"),
            "cpu_review_sha256": digest(trial / "cpu-review.json")}


def evidence(root, block):
    attempts = shared.previous(root / "blocks" / block["id"], block)
    for attempt in attempts:
        for i, row in enumerate(attempt["rows"], 1):
            trial = root / "blocks" / block["id"] / attempt["attempt"] / f"trial-{i:02d}"
            check(row["input_sha256"] == block["case"]["sha256"]
                  and digest(trial / "anytime.json") == row["anytime_sha256"]
                  and digest(trial / "cpu-review.json") == row["cpu_review_sha256"], "archived audit differs")
            row["raw_directory"] = str(trial)
    return attempts


def accepted(history):
    return next((attempt for attempt in history if attempt["status"] in ("QUALIFIED", "ENGINEERING_FAILED")), None)


def execute_block(bundle, root, block, cfg, settings, topology, allowed, state, binary, exe, *, since=None):
    work = root / "blocks" / block["id"]
    work.mkdir(parents=True, exist_ok=True)
    while True:
        history = evidence(root, block)
        if since is None:
            failures = [a for a in history if a["status"] in ("CONTENDED", "MONITOR_INCOMPLETE")]
            if accepted(history) or len(failures) >= 3:
                return
        else:
            failures = [a for a in history[since:]
                        if a["status"] in ("CONTENDED", "MONITOR_INCOMPLETE", "ENGINEERING_FAILED")]
            if (any(a["status"] == "QUALIFIED" for a in history) or len(failures) >= 3
                    or sum(a["status"] == "ENGINEERING_FAILED" for a in failures) >= 2):
                return
        shared.check_stop(root)
        excluded = set().union(*(a["hardware"]["thread_siblings"] for a in failures))
        admission = work / f"admission-{uuid4().hex}"
        admission.mkdir()
        state.update(current_case=block["id"])
        with shared.cpu.lane(root, root.parent / "locks", state, topology, allowed, excluded, admission) as lane:
            hardware, monitor, fds = lane
            attempt = work / f"attempt-{len(history)+1:04d}"
            attempt.mkdir()
            records, verdict = {}, "QUALIFIED"
            for i, method in enumerate(block["arms"], 1):
                shared.check_stop(root)
                state.update(state="RUNNING", phase=block["id"].split("-", 1)[0],
                             current_case=f"{block['id']}/{i}/{method}")
                trial = attempt / f"trial-{i:02d}"
                row = measure(bundle, trial, block["case"], block["seed"], method, block["seconds"],
                              cfg, settings, hardware, monitor, fds, binary, exe)
                durable(trial / "record.json", row)
                records[f"{trial.name}/record.json"] = digest(trial / "record.json")
                if row["cpu"]["contention_reasons"] or not row["cpu"]["coverage_passed"]:
                    verdict = "CONTENDED" if row["cpu"]["contention_reasons"] else "MONITOR_INCOMPLETE"
                    break
                if not row["engineering_passed"]:
                    verdict = "ENGINEERING_FAILED"
            durable(attempt / "finished.json", {"status": verdict, "block": block["id"],
                                               "hardware": hardware, "sha256": records})


def lock_presolve(root, blocks):
    from fractions import Fraction

    scores = {mode: [] for mode in CP_MODES}
    receipts = {}
    for block in blocks:
        attempt = accepted(evidence(root, block))
        for row in attempt["rows"] if attempt else []:
            if row["engineering_passed"]:
                scores[row["method"]].append(Fraction(row["credited_Cmax"], row["ECT"]))
        receipts[block["id"]] = attempt["sha256"] if attempt else None
    eligible = {mode: sum(values) / len(values) for mode, values in scores.items() if len(values) == len(blocks)}
    check(eligible, "neither CP-SAT presolve configuration passed all 24 engineering trials")
    chosen = min(eligible, key=lambda mode: (eligible[mode], mode != "CP-SAT-on"))
    result = {"presolve": chosen == "CP-SAT-on", "case_equal_means": {m: float(v) for m, v in eligible.items()},
              "qualified_counts": {m: len(v) for m, v in scores.items()}, "presolve_records": receipts}
    path = root / "presolve-lock.json"
    if path.exists():
        check(read(path) == result, "presolve lock changed since the first freeze")
    else:
        durable(path, result)
    return result


def snapshots(path, phase, workspace):
    from experiments import server_resource_snapshot, server_storage_snapshot

    path.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, module in (("storage", server_storage_snapshot), ("resource", server_resource_snapshot)):
        try:
            reports[name] = (module.snapshot(phase=phase, account=workspace, project=workspace)
                             if name == "storage" else module.snapshot(phase=phase))
        except Exception as error:
            reports[name] = {"status": "UNKNOWN", "reason": str(error)}
        durable(path / f"{name}.json", reports[name])
    return reports


def check_space(reports):
    storage = reports["storage"]
    free = storage.get("filesystem", {}).get("free_bytes")
    personal = storage.get("personal_data2", {}).get("allocated_bytes")
    if personal is not None and personal >= 250_000_000_000:
        print(f"个人目录已达到 250 GB 提示线，实测 {personal} 字节", flush=True)
    if (personal is not None and personal >= 300_000_000_000) or (free is not None and free < 5_000_000_000):
        raise InterruptedError("measured storage threshold reached; inspect snapshot")


def run(bundle, root, *, minizinc, topology):
    from experiments import final60_report

    plan = read(bundle / "final60-blocks.json")
    cfg = read(bundle / "plan.json")
    total = sum(len(b["arms"]) for group in plan.values() for b in group)
    state = JobState(root, total=total)
    session = root / "sessions" / uuid4().hex
    session.mkdir(parents=True)
    state_name, exit_code, reason = "FAILED", 2, "run failed; see session error"
    done = 0
    try:
        check_space(snapshots(session / "before", "before", root.parent))
        env = qualification.environment(minizinc)
        env.update(numpy=version("numpy"), torch=version("torch"))
        expected = cfg["expected_server_versions"]
        check(all(env.get(name) == pinned for name, pinned in expected.items()), f"server environment differs: {env}")
        check(env["binary_sha256"] == cfg["minizinc_sha256"] and
              digest(minizinc.with_name("fzn-chuffed")) == cfg["chuffed_sha256"], "solver binary hash differs")
        durable(session / "environment.json", {**env, "single_cpu": True, "torch_build": env["torch"]})
        binary, compiler = warm.glnsa.compile_native(bundle / "experiments/glnsa.cpp", session)
        durable(session / "compiler.json", compiler)
        runtime_id = {**{name: env[name] for name in ("python", "ortools", "numpy", "torch", "minizinc", "chuffed")},
                      "python_build": sys.version, "compiler": compiler["version"], "flags": compiler["flags"]}
        runtime_lock = root / "runtime-lock.json"
        if runtime_lock.exists():
            check(read(runtime_lock) == runtime_id, "qualified environment changed since this run began")
        else:
            durable(runtime_lock, runtime_id)
        allowed = os.sched_getaffinity(0)
        settings = {}
        next_snapshot = perf_counter() + 3600
        for stage in ("integration", "presolve", "main", "reference"):
            for block in plan[stage]:
                shared.check_stop(root)
                if perf_counter() >= next_snapshot:
                    check_space(snapshots(session / f"periodic-{uuid4().hex[:8]}", "manual", root.parent))
                    next_snapshot = perf_counter() + 3600
                execute_block(bundle, root, block, cfg["runtime"], settings, topology, allowed, state, binary, minizinc)
                done += len(block["arms"])
                state.update(completed=done)
            if stage == "integration":
                check(all((trial := accepted(evidence(root, block))) and trial["status"] == "QUALIFIED"
                          for block in plan[stage]), "archived active CP or a hybrid failed integration qualification")
            if stage == "presolve":
                settings = lock_presolve(root, plan[stage])
        state_name, exit_code, reason = "COMPLETED", 0, "full plan processed; see per-method final acceptance"
    except (InterruptedError, KeyboardInterrupt) as exc:
        state_name, exit_code, reason = "PAUSED", 3, str(exc)
    except Exception:
        durable(session / "error.json", {"traceback": format_exc()})
    try:
        state.update(state="FINALIZING", phase="snapshots and compact audit")
        snapshots(session / "after", "after", root.parent)
        archive_hash, counts = final60_report.export(bundle, root, plan)
        if state_name == "COMPLETED" and (counts["main_qualified"] != 1296 or counts["reference_qualified"] != 48
                                          or counts["bound_conflicts"]):
            state_name, exit_code, reason = "FAILED", 2, "plan exhausted with failures or bound contradictions; see report"
        state.finish(state_name, exit_code, phase="finished", reason=reason, archive_sha256=archive_hash, **counts)
    except Exception:
        state.finish("FAILED", 2, phase="report failure", reason=format_exc())
        return 2
    return exit_code

