"""Read personal data2 allocated space and filesystem free space; never clean."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess


ACCOUNT = Path(".benchmark-work")
PROJECT = ACCOUNT / "fjsp"
TARGET = 300_000_000_000
WARNING = 250_000_000_000


def du_bytes(path, *, runner=None, timeout=20):
    runner = runner or subprocess.run
    try:
        result = runner(["du", "-s", "-B1", "--", str(path)], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"status": "UNKNOWN", "allocated_bytes": None, "reason": "du_failed", "returncode": result.returncode}
        value = int(result.stdout.split()[0])
        if value < 0:
            raise ValueError("negative allocated size")
        return {"status": "MEASURED", "allocated_bytes": value}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        return {"status": "UNKNOWN", "allocated_bytes": None, "reason": type(error).__name__}


def budget_status(personal_bytes):
    if personal_bytes is None:
        return {"level": "UNKNOWN", "remaining_to_target_bytes": None}
    return {"level": "TARGET_REACHED" if personal_bytes >= TARGET else "WARNING" if personal_bytes >= WARNING else "OK",
        "remaining_to_target_bytes": max(0, TARGET - personal_bytes), "over_target_bytes": max(0, personal_bytes - TARGET)}


def snapshot(*, account=ACCOUNT, project=PROJECT, phase="manual", runner=None, disk_reader=None):
    personal = du_bytes(account, runner=runner)
    project_usage = du_bytes(project, runner=runner)
    try:
        disk = (disk_reader or shutil.disk_usage)(account)
        filesystem = {"status": "MEASURED", "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}
    except OSError as error:
        filesystem = {"status": "UNKNOWN", "reason": type(error).__name__}
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
        "personal_data2_directory": str(account), "personal_data2": personal,
        "project_directory": str(project), "project": project_usage, "filesystem": filesystem,
        "personal_target_bytes": TARGET, "personal_warning_bytes": WARNING, **budget_status(personal["allocated_bytes"]),
        "units": "decimal GB=1e9 bytes; du allocated disk usage, not apparent file length",
        "interpretation": "Filesystem free is shared availability, not personal quota.300GB is the user's target. This snapshot covers personal data2; no storage is deleted."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after", "manual"), default="manual")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-if-at-target", action="store_true")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or ACCOUNT.resolve() != ACCOUNT or PROJECT.resolve() != PROJECT:
        raise ValueError("Use the known personal Ubuntu data2 directories")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "runs"):
        raise ValueError("Save storage reports inside personal project runs")
    report = snapshot(phase=args.phase)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    size = report["personal_data2"]["allocated_bytes"]
    size_text = "未知（检测失败或超时）" if size is None else f"{size / 1e9:.3f} GB"
    free = report["filesystem"].get("free_bytes")
    free_text = "未知" if free is None else f"{free / 1e9:.3f} GB"
    print(f"存储[{args.phase}]：个人data2占用 {size_text}／300GB目标；共享磁盘可用 {free_text}；状态 {report['level']}", flush=True)
    return 3 if args.fail_if_at_target and report["level"] == "TARGET_REACHED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
