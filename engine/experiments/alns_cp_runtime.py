"""Pure CPU neighborhood search with the existing guarded, streaming CP path."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from random import Random
import signal
import sys
from time import perf_counter
import traceback

from alns_neighborhoods import anchor_hash, select_neighborhood
from delivery_guard import CandidatePublisher, atomic_json, run_guarded
from experiments import conditional_selection_cost_pilot as cost


def execute_worker(request, publisher, repair_observer=None, *, controller_factory=None):
    from fjsp import parse_instance, validate_schedule
    if repair_observer is None:
        from experiments.cp_streaming_delivery_probe import observed_repair
        repair_observer = observed_repair
    root, spec = Path(request["output"]), request["spec"]
    cfg, method, seed = spec["config"], spec["method"], spec["seed"]
    instance = parse_instance(cost.pinned(request["input_path"], request["case"]["sha256"]))
    current = request["initial"]
    cost.check(validate_schedule(instance, current) == current["makespan"], "invalid ECT anchor")
    started, cutoff = request["started"], request["worker_deadline"]
    work_until = cutoff - cfg["worker_finish_reserve_ms"] / 1000
    rng, count, updates, late = Random(seed), 0, 0, 0
    controller = controller_factory(cfg, method) if controller_factory else None
    selection_seconds = 0.0
    with (root / "calls.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while perf_counter() + cfg["minimum_remaining_ms"] / 1000 < work_until:
            if (method == "full-cp" and count) or (cfg.get("max_calls") is not None and count >= cfg["max_calls"]):
                break
            selected_at = perf_counter()
            anchor = current
            selection = (controller.select(instance, anchor, rng) if controller else
                         select_neighborhood(instance, anchor, method, cfg["rho"], rng))
            selection_duration = perf_counter() - selected_at
            selection_seconds += selection_duration
            if perf_counter() + cfg["minimum_remaining_ms"] / 1000 >= work_until:
                break
            count += 1
            directory = root / "calls" / f"call-{count:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "anchor.json", anchor)
            call_started = perf_counter()
            local_seconds = controller.target_seconds if controller else cfg["repair_target_ms"] / 1000
            target = work_until if method == "full-cp" else min(work_until, call_started + local_seconds)
            base = {"call": count, "anchor_sha256": anchor_hash(anchor), "anchor_Cmax": anchor["makespan"],
                    "selection": selection, "selection_seconds": selection_duration,
                    "started_ms": (call_started - started) * 1000, "local_target_ms": (target - started) * 1000}
            if controller and hasattr(controller, "complete_observation"):
                base["action_started_ms"] = (selected_at - started) * 1000
            log.write(json.dumps({**base, "event": "CALL_STARTED"}) + "\n")
            sequence, adopted = 0, None

            def publish(schedule, kind, details):
                nonlocal current, updates, sequence, adopted
                value = cost.verify_candidate(instance, anchor, selection["released"], schedule)
                if value >= current["makespan"]:
                    return False
                sequence += 1
                scope = "global" if len(selection["released"]) == len(instance.options) else "neighborhood"
                atomic_json(directory / f"candidate-{sequence:06d}.json",
                            {"call": count, "kind": kind, "released": selection["released"],
                             "schedule": schedule, **details, "bound_scope": scope})
                if perf_counter() >= cutoff:
                    return False
                publisher(schedule, (perf_counter() - started) * 1000, f"alns:{count}:{sequence}")
                current, adopted = schedule, sequence
                updates += 1
                return True

            report = repair_observer(instance, anchor, selection["released"], target,
                                     seed=seed, stream=True, publish=publish)
            finished = perf_counter()
            late += finished > target
            cost.check(report["result"]["status"] not in ("INFEASIBLE", "MODEL_INVALID"), "feasible anchor rejected")
            feedback = {}
            if controller and hasattr(controller, "complete_observation"):
                # Include choice, neighborhood, anchor, model, solve, candidate
                # verification/publication and native report persistence.
                atomic_json(directory / "report.json", report)
                reward_finished = perf_counter()
                feedback = {"reward_finished_ms": (reward_finished - started) * 1000,
                            "reward": controller.complete_observation(report["result"]["status"],
                                anchor["makespan"], current["makespan"],
                                ((reward_finished - started) * 1000 - base["action_started_ms"]) / 1000)}
            elif controller:
                controller.complete(report["result"]["status"], current["makespan"] < anchor["makespan"])
            if not feedback:
                atomic_json(directory / "report.json", report)
            log.write(json.dumps({**base, "event": "CALL_COMPLETED", "finished_ms": (finished - started) * 1000,
                                 "local_target_exceeded": finished > target,
                                 "native_status": report["result"]["status"], "current_Cmax": current["makespan"],
                                 "adopted_frame": adopted,
                                 **feedback,
                                 **({"controller_after": controller.snapshot()} if controller else {})}) + "\n")
    return {"repair_calls": count, "worker_updates": updates, "local_target_exceeded": late,
            "selection_seconds": selection_seconds, "schedule": current, "torch_loaded": "torch" in sys.modules,
            "worker_finished_ms": (perf_counter() - started) * 1000}


def worker(request_path, *, controller_factory=None):
    from experiments.resched_qualification import configure_cpu
    request = cost.read(request_path)
    root = Path(request["output"])
    try:
        if os.name != "nt":
            import ctypes
            cost.check(ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0, "cannot set parent-death signal")
            cost.check(os.getppid() == request["spec"]["parent_pid"], "owning parent changed")
        affinity = configure_cpu()
        atomic_json(root / "worker_ready.json", {"pid": os.getpid(), "affinity": affinity,
                    "kernel_affinity": sorted(os.sched_getaffinity(0)) if os.name != "nt" else None,
                    "python": platform.python_version(), "ortools": version("ortools")})
        result = execute_worker(request, CandidatePublisher(root / "worker", request["started"]),
                                controller_factory=controller_factory)
        atomic_json(root / "worker_result.json", result)
        return 0
    except BaseException:
        atomic_json(root / "worker_failure.json", {"traceback": traceback.format_exc()})
        return 1


def run_trial(bundle, output, case, method, seed, cfg, *, fixture_initial=None, worker_module="experiments.alns_cp_runtime", worker_pass_fds=()):
    from experiments.conditional_runtime import child_environment
    interrupted = []

    def interrupt(signum, frame):
        interrupted.append(signum)
        raise RuntimeError(f"trial_interrupted_{signum}")

    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in old_handlers:
            signal.signal(signum, interrupt)
        with child_environment(bundle) as inherited:
            command = [inherited[0], "-B", "-m", worker_module, "--worker"]
            result = run_guarded(Path(bundle) / case["input"], case, output,
                                 {"method": method, "seed": seed, "config": cfg, "parent_pid": os.getpid()},
                                 total_ms=cfg["total_ms"], initial_schedule=fixture_initial,
                                 worker_command=command, worker_pass_fds=worker_pass_fds, **cfg["guard"])
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    observed = (perf_counter() - result["started_perf_counter"]) * 1000
    result.update(caller_observed_ms=observed, interrupt_signals=interrupted)
    result["delivered_on_time"] &= observed <= cfg["total_ms"]
    result["credited_makespan"] = result["schedule"]["makespan"] if result["delivered_on_time"] else None
    cost.save(Path(output) / "result.json", result)
    return result


def audit_trial(output, case, method, seed, cfg, *, controller_factory=None):
    """Replay the selector and validate every saved CP and publication frame."""
    from fjsp import parse_instance, validate_schedule
    from repair import RepairResult
    root = Path(output)
    result, request = cost.read(root / "result.json"), cost.read(root / "request.json")
    cost.check(request["spec"]["method"] == method and request["spec"]["seed"] == seed and
               request["spec"]["config"] == cfg, "trial request differs")
    instance = parse_instance(cost.pinned(request["input_path"], case["sha256"]))
    initial = result["initial"]
    cost.check(initial is not None, "initial ECT missing")
    validate_schedule(instance, initial)
    events = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()] if (root / "calls.jsonl").exists() else []
    starts = [e for e in events if e["event"] == "CALL_STARTED"]
    finishes = {e["call"]: e for e in events if e["event"] == "CALL_COMPLETED"}
    cost.check([e["call"] for e in starts] == list(range(1, len(starts) + 1)) and
               len(finishes) == sum(e["event"] == "CALL_COMPLETED" for e in events), "call inventory differs")
    rng, anchor, calls, frames = Random(seed), initial, [], 0
    controller = controller_factory(cfg, method) if controller_factory else None
    for event in starts:
        directory = root / "calls" / f"call-{event['call']:06d}"
        saved = cost.read(directory / "anchor.json")
        cost.check(saved == anchor and event["anchor_sha256"] == anchor_hash(saved), "stale call anchor")
        expected = (controller.select(instance, saved, rng) if controller else
                    select_neighborhood(instance, saved, method, cfg["rho"], rng))
        cost.check(expected == event["selection"], "neighborhood replay differs")
        if controller:
            work_until_ms = (request["worker_deadline"] - request["started"]) * 1000 - cfg["worker_finish_reserve_ms"]
            expected_target_ms = min(work_until_ms, event["started_ms"] + controller.target_seconds * 1000)
            cost.check(abs(event["local_target_ms"] - expected_target_ms) < 0.001,
                       "controller target escaped remaining global budget")
        released = expected["released"]
        for path in sorted(directory.glob("candidate-*.json")):
            frame = cost.read(path)
            cost.check(frame["call"] == event["call"] and frame["released"] == released, "candidate identity differs")
            value = cost.verify_candidate(instance, saved, released, frame["schedule"])
            bound = frame.get("lower_bound")
            cost.check(bound is None or 0 <= bound <= value + 1e-6, "invalid native bound")
            cost.check(frame["bound_scope"] == ("global" if len(released) == len(instance.options) else "neighborhood"), "bound scope differs")
            frames += 1
        finish = finishes.get(event["call"])
        if finish is not None:
            native = RepairResult(**cost.read(directory / "report.json")["result"])
            cost.check(native.status == finish["native_status"] and native.status not in ("INFEASIBLE", "MODEL_INVALID"), "native status differs")
            cost.check(native.initial_makespan == saved["makespan"] and
                       native.bound_scope == ("global" if len(released) == len(instance.options) else "neighborhood"), "native anchor/scope differs")
            if native.schedule is not None:
                value = cost.verify_candidate(instance, saved, released, native.schedule)
                cost.check(native.lower_bound is None or native.lower_bound <= value + 1e-6, "native bound exceeds objective")
            if finish["adopted_frame"] is not None:
                anchor = cost.read(directory / f"candidate-{finish['adopted_frame']:06d}.json")["schedule"]
                cost.check(anchor["makespan"] < saved["makespan"], "worker accepted nonimprovement")
            cost.check(anchor["makespan"] == finish["current_Cmax"], "worker update differs")
            if controller:
                if hasattr(controller, "complete_observation"):
                    cost.check(0 <= event["action_started_ms"] <= event["started_ms"] <= finish["finished_ms"]
                               <= finish["reward_finished_ms"] <= result["caller_observed_ms"],
                               "invalid action reward clock")
                    cost.check(event["selection_seconds"] * 1000 <= event["started_ms"] - event["action_started_ms"] + 1e-3,
                               "selection time excluded from reward")
                    if event["call"] > 1:
                        cost.check(event["action_started_ms"] >= finishes[event["call"] - 1]["reward_finished_ms"],
                                   "overlapping action clocks")
                    reward = controller.complete_observation(native.status, saved["makespan"], anchor["makespan"],
                        (finish["reward_finished_ms"] - event["action_started_ms"]) / 1000)
                    cost.check(abs(reward - finish["reward"]) < 1e-7, "reward differs from verified gain/full action clock")
                else:
                    controller.complete(native.status, anchor["makespan"] < saved["makespan"])
                cost.check(finish["controller_after"] == controller.snapshot(), "controller replay differs")
        calls.append({"call": event["call"], "operator": expected["operator"], "released": len(released),
                      "anchor_Cmax": saved["makespan"], "selection_seconds": event["selection_seconds"],
                      "native_status": finish["native_status"] if finish else "INTERRUPTED",
                      "CP_request_seconds": (finish["finished_ms"] - event["started_ms"]) / 1000 if finish else None,
                      "after_Cmax": finish["current_Cmax"] if finish else None,
                      "local_target_exceeded": finish["local_target_exceeded"] if finish else None,
                      **({"phase": expected["phase"], "rho": expected["controller"]["rho"],
                          "target_seconds": expected["controller"]["theta_seconds"]} if controller else {})})
    if method == "full-cp":
        cost.check(len(starts) <= 1, "full CP was restarted")
    best, trace = initial["makespan"] + 1, []
    for receipt in result["receipts"]:
        schedule = cost.pinned(root / receipt["file"], receipt["sha256"])
        value = validate_schedule(instance, schedule)
        cost.check(value == receipt["makespan"], "receipt objective differs")
        if not receipt["eligible"]:
            continue
        cost.check(value < best and receipt["delivery_written_ms"] <= cfg["total_ms"], "invalid receipt deadline/order")
        if receipt["source"] in ("ECT", "fixture-initial", "Stall80"):
            cost.check(not trace and schedule == initial, "initial receipt differs")
        else:
            prefix, call, sequence = receipt["source"].split(":")
            cost.check(prefix == "alns", "unknown publication source")
            directory = root / "calls" / f"call-{int(call):06d}"
            frame = cost.read(directory / f"candidate-{int(sequence):06d}.json")
            cost.check(frame["schedule"] == schedule, "receipt lacks its original frame")
            cost.verify_candidate(instance, cost.read(directory / "anchor.json"), frame["released"], schedule)
        best = value
        trace.append({"ms": receipt["delivery_written_ms"], "Cmax": value})
    cost.check(cost.read(root / "delivery.json") == result["schedule"] and best == result["schedule"]["makespan"], "delivery differs")
    cost.check(result["credited_makespan"] == (best if result["delivered_on_time"] else None), "incorrect trial credit")
    ready = cost.read(root / "worker_ready.json") if (root / "worker_ready.json").exists() else None
    worker_result = cost.read(root / "worker_result.json") if (root / "worker_result.json").exists() else None
    if ready:
        cost.check(ready["pid"] == result["worker"]["pid"], "worker ownership differs")
    if worker_result:
        cost.check(not worker_result["torch_loaded"] and worker_result["schedule"] == anchor and
                   worker_result["repair_calls"] == len(starts) == len(finishes), "worker final evidence differs")
    rejections = sum(e["event"] == "REJECTED" for e in result["events"])
    passed = bool(result["delivered_on_time"] and result["error"] is None and
                  ready and worker_result and result["worker"]["exit_code"] == 0 and
                  not result["worker"]["terminated"] and not result["worker"]["still_running"] and
                  not rejections and not (root / "worker_failure.json").exists() and not result.get("interrupt_signals"))
    return {"instance": case["instance"], "operations": case["operations"], "method": method, "seed": seed,
            "initial_Cmax": initial["makespan"], "initial_sha256": anchor_hash(initial),
            "credited_Cmax": result["credited_makespan"], "delivered_Cmax": best,
            "caller_observed_ms": result["caller_observed_ms"], "delivered_on_time": result["delivered_on_time"],
            "worker_reaped": not result["worker"]["still_running"], "worker_exit_code": result["worker"]["exit_code"],
            "worker_terminated": result["worker"]["terminated"], "publication_rejections": rejections,
            "engineering_passed": passed, "repair_calls": len(starts), "completed_calls": len(finishes),
            "candidate_frames_checked": frames, "selection_seconds": sum(c["selection_seconds"] for c in calls),
            "initial_source": result["receipts"][0]["source"], "anytime_trace": trace, "calls": calls,
            "training": False, "scope": "six exposed development instances; fixed rho; no generalization claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    raise SystemExit(worker(parser.parse_args().worker))
