"""Overnight follow-up using the final benchmark's timed trial, audit and CPU lane."""

from collections import Counter
from importlib.metadata import version
import json
import os
import platform
import sys
import tarfile
from time import monotonic
from traceback import format_exc
from uuid import uuid4

from experiments import final60_job as base
from experiments.early_switch_report import COUNT, markdown, summarize, timing_figure
from experiments.final60_worker import check, read
from experiments.offline_job_state import JobState, atomic_publish, durable


def publish_text(path, text):
    temp = path.with_name("." + path.name + "-" + uuid4().hex)
    temp.write_bytes(text.encode("utf-8"))
    atomic_publish(temp, path)


def save_summary(root, name, summary):
    durable(root / (name + ".json"), summary)
    publish_text(root / (name + ".md"), markdown(summary))


def qualified_rows(blocks, histories):
    rows = []
    for block in blocks:
        accepted = [a for a in histories.get(block["id"], []) if a["status"] == "QUALIFIED"]
        check(len(accepted) <= 1, "duplicate qualified block")
        if accepted:
            for r in accepted[0]["rows"]:
                check(r["handoff_sha256"] and r["switch_ms"] in (5000, 10000)
                      and r["first_native_cp_ms"] >= r["switch_ms"] and r["wall_ms"] <= 60000
                      and r["credited_Cmax"] == r["checkpoints"]["60"], "invalid handoff or timely endpoint")
                rows.append(r)
    return rows


def lock_decision(root, report, rows):
    if report["selected_seconds"] is None:
        return
    data = {"seconds": report["selected_seconds"], "count_10_vs_5": report["count_10_vs_5"]["case_effects_pct"],
            "count_trials": sorted((r["instance"], r["seed"], r["method"], r["credited_Cmax"],
                                    r["anytime_sha256"]) for r in rows if r["method"] in COUNT)}
    data["count_trials"] = [list(r) for r in data["count_trials"]]
    path = root / "count-selection.json"
    if path.exists():
        check(read(path) == data, "full Count decision or its observations changed")
    else:
        durable(path, data)


def prepare(bundle, root, session):
    base.warm.verify_clocks()
    binary, compiler = base.warm.glnsa.compile_native(bundle / "experiments/glnsa.cpp", session)
    runtime = {"python": platform.python_version(), "python_build": sys.version,
               "ortools": version("ortools"), "numpy": version("numpy"),
               "compiler": compiler["version"], "flags": compiler["flags"]}
    durable(session / "environment.json", {**runtime, "native": compiler})
    path = root / "runtime-lock.json"
    if path.exists():
        check(read(path) == runtime, "runtime changed on resume")
    else:
        durable(path, runtime)
    return binary


def export(bundle, root, histories, summary):
    durable(root / "audit.json", {"manifest_sha256": base.digest(bundle / "source-freeze.json"),
                                  "planned_trials": 576, "attempts": histories,
                                  "attempt_counts": dict(Counter(a["status"] for h in histories.values() for a in h)),
                                  "raw_root": str(root / "blocks"), "qualified_trials": summary["qualified_trials"]})
    save_summary(root, "report", summary)
    chart = timing_figure(summary)
    if chart:
        publish_text(root / "objective-timeline.svg", chart)
        publish_text(root / "report.md", markdown(summary) + "\n![早切换四组的工期曲线](objective-timeline.svg)\n")
    files = {p.name: p for p in root.iterdir() if p.is_file() and
             (p.stem in ("identity", "runtime-lock", "audit", "report", "count-selection") or p.name.startswith("count-at-"))}
    files.update({p.relative_to(root).as_posix(): p for p in (root / "sessions").rglob("*.json")})
    files.update({n: bundle / n for n in ("early-blocks.json", "early-history.json", "early-source.json")})
    if chart:
        files["objective-timeline.svg"] = root / "objective-timeline.svg"
    durable(root / "report-manifest.json", {n: base.digest(p) for n, p in files.items()})
    files["report-manifest.json"] = root / "report-manifest.json"
    temp = root / ".brief.tmp"
    with tarfile.open(temp, "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    target = root / "brief.tar.gz"
    atomic_publish(temp, target)
    checksum = base.digest(target)
    publish_text(root / "brief.tar.gz.sha256", f"{checksum}  brief.tar.gz\n")
    return checksum


def run(bundle, root, *, topology):
    cfg = read(bundle / "plan.json")["runtime"]
    plan = read(bundle / "early-blocks.json")
    blocks = plan["count"] + plan["tree"]
    cases = {b["case"]["instance"]: b["case"] for b in plan["count"]}
    archived = {(r["instance"], r["seed"], r["method"]): r["credited_Cmax"]
                for r in read(bundle / "early-history.json")}
    state = JobState(root, total=576)
    session = root / "sessions" / uuid4().hex
    session.mkdir(parents=True)
    histories = {}
    terminal, code, reason = "FAILED", 2, "run incomplete"
    try:
        base.check_space(base.snapshots(session / "before", "before", root.parent))
        binary = prepare(bundle, root, session)
        allowed = os.sched_getaffinity(0)
        histories = {b["id"]: base.evidence(root, b) for b in blocks}
        completed = len(qualified_rows(blocks, histories))
        state.update(completed=completed)
        next_snapshot = monotonic() + 3600
        for stage in ("count", "tree"):
            for index, block in enumerate(plan[stage], 1):
                history = histories[block["id"]]
                if not any(a["status"] == "QUALIFIED" for a in history):
                    while True:
                        base.shared.check_stop(root)
                        if monotonic() >= next_snapshot:
                            base.check_space(base.snapshots(session / ("periodic-" + uuid4().hex[:8]), "manual", root.parent))
                            next_snapshot = monotonic() + 3600
                        try:
                            base.execute_block(bundle, root, block, cfg, {}, topology, allowed, state, binary, None,
                                               since=len(history))
                            break
                        except base.shared.cpu.CpuUnavailable:
                            state.update(state="WAITING_RESOURCE", phase="rescan compatible shared cores")
                    history = histories[block["id"]] = base.evidence(root, block)
                    if any(a["status"] == "QUALIFIED" for a in history):
                        completed += 2
                    state.update(completed=completed)
                if index % 12 == 0:
                    rows = qualified_rows(blocks, histories)
                    save_summary(root, "progress", summarize(cases, archived, rows))
                if stage == "count" and index in (48, 72, 144):
                    prefix = plan[stage][:index]
                    rows = qualified_rows(prefix, histories)
                    if len(rows) == 2 * len(prefix):
                        prefix_cases = {b["case"]["instance"]: b["case"] for b in prefix}
                        report = summarize(prefix_cases, archived, rows, intervals=True)
                        save_summary(root, f"count-at-{len(prefix_cases)}", report)
                        lock_decision(root, report, rows)
        if completed == 576:
            terminal, code, reason = "COMPLETED", 0, "all four arms qualified"
        else:
            reason = "unqualified pairs remain; resume to retry only those pairs"
    except (KeyboardInterrupt, InterruptedError) as exc:
        terminal, code, reason = "PAUSED", 3, str(exc) or "interrupted"
    except Exception:
        reason = format_exc()
        durable(session / "error.json", {"traceback": reason})
    try:
        state.update(state="FINALIZING", phase="end snapshots and compact results")
        base.snapshots(session / "after", "after", root.parent)
        histories = {b["id"]: base.evidence(root, b) for b in blocks}
        rows = qualified_rows(blocks, histories)
        summary = summarize(cases, archived, rows, intervals=True)
        lock_decision(root, summary, rows)
        if terminal == "COMPLETED":
            check(len(rows) == 576 and summary["selected_tree_vs_count"]["n"] == 48, "incomplete final comparison")
        signature = export(bundle, root, histories, summary)
        state.finish(terminal, code, phase="finished", completed=len(rows), archive_sha256=signature,
                     selected_seconds=summary["selected_seconds"], reason=reason)
    except Exception:
        state.finish("FAILED", 2, phase="report failure", reason=format_exc())
        return 2
    return code

