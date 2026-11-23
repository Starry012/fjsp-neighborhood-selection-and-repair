"""Rebuild the homepage table and curves from both batches of scored runs."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean

from early_switch import COUNT, DIRECT, TREE
from summarize import METHODS, RESULTS, ROOT, SEEDS, TIMES, case_effects, outcomes, read_json


LABELS = {
    DIRECT: "直接 Count-LNS",
    "Tree2WithoutClock": "直接 Tree-LNS",
    COUNT[0]: "GLNSA 5 秒 → Count-LNS",
    COUNT[1]: "GLNSA 10 秒 → Count-LNS",
    "Stall20-MinOptions2": "GLNSA 20 秒 → Count-LNS",
    "Stall35-MinOptions2": "GLNSA 35 秒 → Count-LNS",
    TREE[1]: "GLNSA 10 秒 → Tree-LNS",
    "Stall80": "GLNSA",
    "Chuffed": "全模型 MiniZinc＋Chuffed",
    "full-cp-sat-hinted": "全模型 OR-Tools CP-SAT",
}


def overview(main_run=None, early_run=None):
    cases = [row["instance"] for row in read_json(ROOT / "engine/cases.json")]
    index = {}
    sources = {}
    batches = (
        ("正式", "observations.jsonl", METHODS,
         read_json(RESULTS / "audit.json")["public_data_sha256"]["observations.jsonl"]),
        ("补充", "early-switch.jsonl", (*COUNT, *TREE),
         read_json(RESULTS / "early-switch-audit.json")["observations_sha256"]),
    )
    for source, filename, methods, digest in batches:
        if source == "正式" and main_run is not None:
            blocks = read_json(main_run / "report.json")["stages"]["main"]
            histories = [block["history"] for block in blocks]
        elif source == "补充" and early_run is not None:
            histories = read_json(early_run / "audit.json")["attempts"].values()
        else:
            histories = None
        if histories is None:
            raw = (RESULTS / filename).read_bytes()
            if sha256(raw).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {filename}")
            rows = [json.loads(line) for line in raw.splitlines()]
        else:
            rows = [r for history in histories for attempt in history
                    if attempt["status"] == "QUALIFIED" for r in attempt["rows"]]
        batch = {(r["instance"], r["seed"], r["method"]): r for r in rows}
        expected = {(name, seed, m) for name in cases for seed in SEEDS for m in methods}
        if len(batch) != len(rows) or batch.keys() != expected or index.keys() & batch.keys():
            raise ValueError(f"Incomplete or duplicate runs: {filename}")
        index.update(batch)
        sources.update(dict.fromkeys(methods, source))

    summary = []
    for method, label in LABELS.items():
        runs = [index[name, seed, method] for name in cases for seed in SEEDS]
        recorded = set.intersection(*({int(t) for t, v in r["checkpoints"].items() if v is not None} for r in runs))
        if not recorded.issuperset(TIMES) or any(r["checkpoints"]["60"] != r["credited_Cmax"] for r in runs):
            raise ValueError(f"Missing or inconsistent checkpoints: {method}")
        curve = {t: mean(r["checkpoints"][str(t)] for r in runs) for t in sorted(recorded)}
        effect = outcomes(case_effects(index, cases, DIRECT, method).values())
        summary.append(dict(method=method, label=label, source=sources[method], curve=curve, **effect))
    return sorted(summary, key=lambda row: row["curve"][60])


def markdown_table(rows):
    yield "| 方法 | 批次 | 10 秒平均工期 | 20 秒平均工期 | 60 秒平均工期 ↓ | 直接 Count-LNS 的逐题中位降幅 | 直接 Count-LNS 胜／平／负 |"
    yield "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    for row in rows:
        c = row["curve"]
        effect = f"{row['median']:.2f}%"
        wins = f"{row['wins']}／{row['ties']}／{row['losses']}"
        if row["method"] == DIRECT:
            effect = wins = "—"
        yield (f"| {row['label']} | {row['source']} | {c[10]:.2f} | {c[20]:.2f} | {c[60]:.2f} | "
               f"{effect} | {wins} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, help="use a completed reproduce.py run output")
    parser.add_argument("--early-run", type=Path, help="use a completed reproduce.py run-early-switch output")
    args = parser.parse_args()
    print("\n".join(markdown_table(overview(args.main_run, args.early_run))))
