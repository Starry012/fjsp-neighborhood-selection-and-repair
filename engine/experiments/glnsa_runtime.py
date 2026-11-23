"""Single-process native GLNSA trials with independently validated, timed delivery."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import monotonic, monotonic_ns, sleep

from experiments.offline_job_state import durable
from fjsp import parse_instance, validate_schedule


# (generation ceiling, stagnation counter threshold, tabu-depth divisor).
# Zero disables that stop. Order also defines the preregistered exact-tie rule.
POLICIES = {"R250": (250,40,2), "R500": (500,40,2), "Stall40": (0,40,2),
            "Stall80": (0,80,2), "Deep": (0,0,2), "Stall40-TS1": (0,40,1)}


def check(condition, why):
    if not condition:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def compile_native(source, directory):
    errors = []
    for name in ("g++", "clang++", "c++"):
        compiler = shutil.which(name)
        if not compiler:
            continue
        binary = directory / "glnsa"
        flags = ["-std=c++17", "-O3", "-DNDEBUG"]
        result = subprocess.run([compiler, *flags, str(source), "-o", str(binary)],
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            version = subprocess.check_output([compiler, "--version"], text=True, timeout=5).splitlines()[0]
            return binary, {"compiler": compiler, "version": version, "flags": flags,
                            "source_sha256": digest(source), "binary_sha256": digest(binary)}
        errors.append([compiler, result.stderr[-1800:]])
    raise RuntimeError(f"C++17 build failed; checked g++, clang++, c++; diagnostics={errors}")


def encode(instance):
    machines = sorted({m for choices in instance.options.values() for m in choices})
    chains = [instance.jobs[j] for j in sorted(instance.jobs)]
    operations = [op for chain in chains for op in chain]
    lines = [f"{len(chains)} {len(machines)}"]
    for chain in chains:
        lines.append(str(len(chain)))
        for op in chain:
            lines.append(" ".join(str(instance.options[op].get(m, 0)) for m in machines))
    return "\n".join(lines) + "\n", operations, machines


def frame(instance, path, operations, machines):
    lines = path.read_text("ascii").splitlines()
    generation, reported = map(int, lines[0].split())
    assignments = []
    for op, line in zip(operations, lines[1:], strict=True):
        m, start, end = map(int, line.split())
        check(0 <= m < len(machines), "machine index out of range")
        assignments.append(dict(operation=op, machine=machines[m], start=start, end=end))
    schedule = dict(instance=instance.name, assignments=assignments, makespan=reported)
    validate_schedule(instance, schedule)
    return generation, schedule


def read_rounds(directory, seed, started_ns, policy=None):
    rounds = []
    cap, stall, _ = POLICIES[policy] if policy else (250,40,2)
    for line in (directory / "rounds.txt").read_text("ascii").splitlines():
        number, run_seed, begin, end, reason, gens, candidates, moves, best, frames = line.split()
        number, run_seed, begin, end, gens, candidates, moves, best, frames = map(
            int, (number, run_seed, begin, end, gens, candidates, moves, best, frames))
        check(number == len(rounds)+1 and run_seed == (seed + (number-1)*0x9e3779b9) % 2**32, "restart seed/order")
        begin_ms, end_ms = (begin-started_ns)/1e6, (end-started_ns)/1e6
        check((rounds[-1]["end_ms"] if rounds else 0) <= begin_ms <= end_ms and
              (not rounds or rounds[-1]["reason"] in ("generation_limit","stagnation")), "restart boundary")
        check(reason in ("generation_limit","stagnation","wall_clock") and
              (reason != "generation_limit" or (cap and gens == cap)) and
              (reason != "stagnation" or stall) and gens >= 0 and (not cap or gens <= cap)
              and min(candidates, moves, frames) >= 0 and
              (best > 0 or (best == -1 and reason == "wall_clock" and gens == frames == 0)), "invalid native round")
        rounds.append(dict(round=number, seed=run_seed, start_ms=begin_ms, end_ms=end_ms,
                           reason=reason, generations=gens, global_candidates=candidates, tabu_moves=moves,
                           best_Cmax=best if best > 0 else None, frames=frames))
    check(rounds, "no native round started")
    if policy:
        decisions = (directory/"decisions.txt").read_text("ascii").splitlines()
        check(tuple(map(int,decisions[0].split())) == POLICIES[policy], "native policy differs")
        for r, line in zip(rounds, decisions[1:], strict=True):
            number, active_gen, last_gain, counter, retained = map(int,line.split())
            gen = r["generations"]
            check(number == r["round"] and gen <= active_gen <= gen+1 and 0 <= last_gain <= gen and
                  counter == gen-last_gain+1 and
                  (r["reason"] == "wall_clock" or active_gen == gen) and
                  ((r["reason"] == "stagnation") == bool(stall and counter >= stall)), "native stopping decision")
            check(retained == -1 or (r["best_Cmax"] is not None and r["best_Cmax"] <= retained),
                  "fully decoded incumbent worse than retained population")
            r.update(active_generation=active_gen, last_population_gain_generation=last_gain,
                     stagnant_counter=counter,population_Cmax=None if retained == -1 else retained)
    return rounds


def run_trial(binary, case, source, directory, seed, *, total_ms=60000, pass_fds=(), policy=None,
              common_start_ns=None, handoff_ms=None, expected_cpu=None):
    directory.mkdir(parents=True, exist_ok=False)
    started_ns = monotonic_ns() if common_start_ns is None else common_start_ns
    started = started_ns / 1e9
    deadline = started + total_ms / 1000
    # Same margins as the registered Tree2 controller (publication 1200, stop 600,
    # worker finish 1200 ms earlier than publication cutoff). Scaled only in smoke tests.
    scale = min(1, total_ms / 6000)
    publish_until = deadline - 1.2 * scale
    stop_at = deadline - .6 * scale
    native_cutoff_ms = total_ms - 2400 * scale
    if handoff_ms is not None:
        check(policy == "Stall80" and 0 < handoff_ms < native_cutoff_ms, "invalid prefix")
        native_cutoff_ms = handoff_ms
        publish_until = started + handoff_ms / 1000
        stop_at = min(stop_at, publish_until + .6)
    native_until_ns = started_ns + round(native_cutoff_ms * 1e6)
    elapsed = lambda: (monotonic_ns() - started_ns) / 1e6
    raw = source.read_bytes()
    check(sha256(raw).hexdigest() == case["sha256"], "instance digest changed")
    instance = parse_instance(json.loads(raw))
    check(instance.name == case["instance"] and len(instance.options) == case["operations"], "instance identity changed")
    encoded, operations, machines = encode(instance)
    input_file = directory / "native-input.txt"
    input_file.write_text(encoded, "ascii")
    trace = []
    last = 0
    eligible_schedule = None

    def consume():
        nonlocal last, eligible_schedule
        while monotonic() < publish_until:
            path = directory / f"frame-{last + 1}"
            if not path.exists():
                break
            generation, schedule = frame(instance, path, operations, machines)
            check(not trace or schedule["makespan"] < trace[-1]["Cmax"], "non-improving native frame")
            if monotonic() >= publish_until:
                break
            durable(directory / "delivery.json", schedule)
            written = elapsed()
            if handoff_ms is not None and written > handoff_ms:
                # The write straddled the switch. Restore the last timely incumbent.
                check(eligible_schedule is not None, "no timely prefix incumbent")
                durable(directory / "delivery.json", eligible_schedule)
                break
            check(written <= total_ms, "late publication")
            eligible_schedule = schedule
            last += 1
            trace.append(dict(ms=written, Cmax=schedule["makespan"], generation=generation, frame=last))

    with (directory / "native.log").open("wb") as log:
        args = [str(binary), "budget60", str(input_file), str(directory), str(seed),
                str(native_until_ns), str(os.getpid())]
        if policy:
            args[1] = "deep60"
            args.extend(map(str,POLICIES[policy]))
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, pass_fds=pass_fds)
        try:
            affinity = sorted(os.sched_getaffinity(proc.pid)) if expected_cpu is not None else None
            check(expected_cpu is None or affinity == [expected_cpu], "native worker affinity differs")
            while proc.poll() is None and monotonic() < stop_at:
                consume()
                sleep(min(.010, max(0, stop_at - monotonic())))
            consume()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=max(.001, deadline - monotonic() - .1 * scale))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
    observed = elapsed()
    check(proc.returncode == 0 and observed <= total_ms and trace,
          f"delivery/exit failure: {observed:.1f}ms, frames={len(trace)}, exit={proc.returncode}")
    rounds = read_rounds(directory, seed, started_ns, policy)
    end_ms = (int((directory/"budget-end.txt").read_text())-started_ns)/1e6
    offset = 0
    for row in rounds:
        for receipt in trace[offset:offset+row["frames"]]:
            receipt["round"] = row["round"]
        offset += row["frames"]
    first = [r["Cmax"] for r in trace if r["round"] == 1]
    check(first, "missing first run delivery")
    result = dict(instance=instance.name, seed=seed, method="GLNSA-MultiStart60", input_sha256=case["sha256"],
                  budget_ms=total_ms, anytime_trace=trace, started_monotonic_ns=started_ns,
                  credited_Cmax=trace[-1]["Cmax"], delivery_sha256=digest(directory / "delivery.json"),
                  native_stop="shared_wall_cutoff", search_cutoff_ms=native_cutoff_ms, native_end_ms=end_ms,
                  rounds=rounds, first_round_delivered_Cmax=first[-1], best_round=trace[-1]["round"],
                  generations=sum(r["generations"] for r in rounds), global_candidates=sum(r["global_candidates"] for r in rounds),
                  tabu_moves=sum(r["tabu_moves"] for r in rounds), native_frames=offset, worker_exit=proc.returncode)
    if policy:
        result.update(method=policy, milestones=read_milestones(directory))
    if handoff_ms is not None:
        result["handoff_ms"] = handoff_ms
    result["kernel_affinity"] = affinity
    result["caller_observed_ms"] = elapsed()
    check(native_cutoff_ms-1 <= rounds[-1]["end_ms"] <= end_ms <= result["caller_observed_ms"] <= total_ms,
          "search stopped early or delivery exceeded shared budget")
    return result


def read_milestones(directory):
    milestones = {}
    for line in (directory/"milestones.txt").read_text("ascii").splitlines():
        gen, signature, best, neighbors, steps = map(int,line.split())
        check(str(gen) not in milestones, "duplicate first-round milestone")
        milestones[str(gen)] = [signature,best,neighbors,steps]
    return milestones


def audit(case, source, directory, record):
    raw = source.read_bytes()
    check(sha256(raw).hexdigest() == record["input_sha256"] == case["sha256"], "audit input")
    instance = parse_instance(json.loads(raw))
    _, ops, machines = encode(instance)
    trace = record["anytime_trace"]
    check(instance.name == record["instance"] == case["instance"] and trace
          and record["worker_exit"] == 0 and record["caller_observed_ms"] <= record["budget_ms"], "audit budget or identity")
    check(len(list(directory.glob("frame-*"))) == record["native_frames"] >= len(trace), "frame count")
    policy = None if record["method"]=="GLNSA-MultiStart60" else record["method"]
    rounds = read_rounds(directory, record["seed"], record["started_monotonic_ns"], policy)
    if policy:
        marks = read_milestones(directory)
        check(marks == record["milestones"] and set(marks) ==
              {str(g) for g in (0,1,20,100,250,251,500,501) if g<=rounds[0]["generations"]},
              "missing or changed first-round state")
    cutoff = record["budget_ms"] - 2400*min(1, record["budget_ms"]/6000)
    if "handoff_ms" in record:
        check(policy == "Stall80" and 0 < record["handoff_ms"] < cutoff and
              all(p["ms"] <= record["handoff_ms"] for p in trace), "late or invalid warm start")
        cutoff = record["handoff_ms"]
    end_ms = (int((directory/"budget-end.txt").read_text())-record["started_monotonic_ns"])/1e6
    check(rounds == record["rounds"] and
          cutoff == record["search_cutoff_ms"] and record["native_end_ms"] == end_ms and
          cutoff-1 <= rounds[-1]["end_ms"] <= end_ms <= record["caller_observed_ms"], "audit search window")
    frame_rounds = [r for r in rounds for _ in range(r["frames"])]
    check(len(frame_rounds) == record["native_frames"], "round frame count")
    last_value, last_position, last_ms = float("inf"), (0,-1), -1
    round_bests = {}
    for i in range(1, record["native_frames"] + 1):
        gen, schedule = frame(instance, directory / f"frame-{i}", ops, machines)
        native_round = frame_rounds[i-1]
        position = (native_round["round"],gen)
        check(schedule["makespan"] < last_value and (position>=last_position if policy else position>last_position)
              and 0 <= gen <= native_round.get("active_generation",native_round["generations"])
              and (i != 1 or position == (1,0)), "invalid search trace")
        last_value, last_position = schedule["makespan"], position
        round_bests[position[0]] = last_value
        if i <= len(trace):
            row = trace[i - 1]
            check((row["Cmax"], row["round"], row["generation"], row["frame"]) == (last_value, position[0], gen, i)
                  and max(last_ms,native_round["start_ms"]) <= row["ms"] <= record["caller_observed_ms"], "invalid receipt")
            last_ms = row["ms"]
            if i == len(trace):
                check(schedule == read(directory / "delivery.json"), "final frame mismatch")
    check(digest(directory / "delivery.json") == record["delivery_sha256"]
          and trace[-1]["Cmax"] == record["credited_Cmax"], "delivery changed")
    incumbent = float("inf")
    for r in rounds:
        if r["best_Cmax"] is not None and r["best_Cmax"] < incumbent:
            check(round_bests.get(r["round"]) == r["best_Cmax"], "missing improved restart incumbent")
            incumbent = r["best_Cmax"]
        else:
            check(r["round"] not in round_bests, "spurious restart incumbent")
    check(trace[-1]["round"] == record["best_round"] and
          min(t["Cmax"] for t in trace if t["round"]==1) == record["first_round_delivered_Cmax"], "restart credit")
