"""Predeclared endpoint analysis for the 5/10 s warm-start follow-up."""

from collections import Counter
from fractions import Fraction
from statistics import mean, median
from xml.sax.saxutils import escape


SEEDS = (42, 43, 44)
COUNT = ("Stall05-MinOptions2", "Stall10-MinOptions2")
TREE = ("Stall05-Tree2WithoutClock", "Stall10-Tree2WithoutClock")
HISTORICAL = ("MinOptions2", "Stall20-MinOptions2", "Stall35-MinOptions2", "Tree2WithoutClock")
LABELS = dict(zip((*COUNT, *TREE, *HISTORICAL),
                  ("5 秒 → Count-LNS", "10 秒 → Count-LNS", "5 秒 → Tree-LNS", "10 秒 → Tree-LNS",
                   "直接 Count-LNS", "20 秒 → Count-LNS", "35 秒 → Count-LNS", "直接 Tree-LNS"), strict=True))


def comparison(scores, cases, first, baseline, interval=False):
    from experiments.final60_report import stratified_interval

    values = {}
    for name in cases:
        diffs = []
        for seed in SEEDS:
            a = scores.get((name, seed, first))
            b = scores.get((name, seed, baseline))
            if a is not None and b is not None:
                diffs.append(Fraction(100 * (b - a), b))
        if len(diffs) == len(SEEDS):
            values[name] = median(diffs)
    middle = median(values.values()) if values else None
    def describe(items):
        items = list(items)
        return {"n": len(items), "median_pct": float(median(items)) if items else None,
                "wins": sum(v > 0 for v in items), "ties": items.count(0), "losses": sum(v < 0 for v in items)}
    strata = sorted({c["stratum"] for c in cases.values()})
    return {"first": first, "baseline": baseline, **describe(values.values()),
            "sign": None if middle is None else (middle > 0) - (middle < 0),
            "strata": {s: describe(v for n, v in values.items() if cases[n]["stratum"] == s) for s in strata},
            "case_effects_pct": {n: float(v) for n, v in values.items()},
            "interval95": stratified_interval({n: float(v) for n, v in values.items()}, cases, .95)
                          if interval and len(values) > 1 else None}


def summarize(cases, history_scores, rows, *, intervals=False):
    scores = dict(history_scores)
    handoffs = {}
    for row in rows:
        key = row["instance"], row["seed"], row["method"]
        if key in scores:
            raise ValueError(f"duplicate qualified observation: {key}")
        scores[key] = row["credited_Cmax"]
        handoffs[key] = row["handoff_sha256"]
    compare = lambda a, b: comparison(scores, cases, a, b, intervals)
    primary = compare(COUNT[1], COUNT[0])
    chosen = None
    if primary["n"] == 48:
        chosen = 10 if primary["sign"] > 0 else 5
    contrasts = [compare(m, ref) for m in COUNT for ref in HISTORICAL[:3]]
    slot = int(chosen == 10)
    selected = None
    if chosen is not None:
        selected = compare(TREE[slot], COUNT[slot])
        selected["vs_direct_tree"] = compare(TREE[slot], HISTORICAL[3])
        selected["vs_direct_count"] = compare(TREE[slot], HISTORICAL[0])
        selected["matched_handoffs"] = sum(handoffs.get((n, s, COUNT[slot])) == h
            for (n, s, m), h in handoffs.items() if m == TREE[slot])
    curves = {}
    for method in (*COUNT, *TREE):
        subset = [r for r in rows if r["method"] == method]
        if subset:
            curves[method] = {t: mean(r["checkpoints"][t] for r in subset)
                              for t in subset[0]["checkpoints"] if all(r["checkpoints"][t] is not None for r in subset)}
    return {"qualified_trials": len(rows), "counts": dict(Counter(r["method"] for r in rows)),
            "count_10_vs_5": primary, "selected_seconds": chosen,
            "selected_by": "full 48-case Count paired median only; exact tie selects 5 s",
            "selected_tree_vs_count": selected, "count_vs_historical": contrasts,
            "mean_Cmax_by_second": curves}


def markdown(report):
    def pct(value):
        return "—" if value is None else f"{value:+.2f}%"
    def line(c, title):
        ci = c["interval95"]
        bounds = f"[{pct(ci[0])}, {pct(ci[1])}]" if ci else "—"
        return (f"| {title} | {c['n']} | {pct(c['median_pct'])} | {bounds} | "
                f"{c['wins']} / {c['ties']} / {c['losses']} |")
    primary = report["count_10_vs_5"]
    chosen = report["selected_seconds"]
    if chosen is None:
        decision = "全量时点待定。"
        if primary["n"]:
            direction = {1: "10 秒组领先", -1: "5 秒组领先", 0: "中位数持平"}[primary["sign"]]
            decision += f" 已完成的 {primary['n']} 道配对题中，{direction}。"
    else:
        decision = f"全量 Count-LNS 选择 {chosen} 秒交接。"
        if primary["sign"] == 0:
            decision += " 题级中位降幅恰为零，按预设平局规则采用 5 秒。"
    lines = ["# GLNSA 早切换补充实验", "", f"合格场次 {report['qualified_trials']} / 576。{decision}", "",
             "每题三个种子先取配对中位，再按题取中位。正值表示前者工期较短。区间为按四层、以题为单位重采样的描述性 95% 区间。", "",
             "| 60 秒终点比较 | 完整题数 | 中位降幅 | 95% 区间 | 胜 / 平 / 负 |",
             "| --- | ---: | ---: | --- | --- |", line(primary, "10 秒对 5 秒，同批 Count-LNS")]
    for c in report["count_vs_historical"]:
        lines.append(line(c, LABELS[c["first"]] + " 对 " + LABELS[c["baseline"]] + "（跨批）"))
    selected = report["selected_tree_vs_count"]
    if selected and selected["n"]:
        lines.append(line(selected, f"{chosen} 秒 Tree-LNS 对同时间 Count-LNS（跨阶段）"))
        for name in ("vs_direct_tree", "vs_direct_count"):
            c = selected[name]
            lines.append(line(c, LABELS[c["first"]] + " 对 " + LABELS[c["baseline"]] + "（跨批）"))
        lines.extend(["", f"选定时点的树组与计数组中，交接排程指纹相同的题×种子有 {selected['matched_handoffs']} 对。"])
    lines.extend(["", "| 10 秒对 5 秒 Count-LNS | 题数 | 中位降幅 | 胜 / 平 / 负 |",
                  "| --- | ---: | ---: | --- |"])
    for label, c in primary["strata"].items():
        lines.append(f"| {label} | {c['n']} | {pct(c['median_pct'])} | {c['wins']} / {c['ties']} / {c['losses']} |")
    lines.extend(["", "| 新组 | 合格 / 144 | 平均 60 秒工期 |", "| --- | ---: | ---: |"])
    for m in (*COUNT, *TREE):
        avg = report["mean_Cmax_by_second"].get(m, {}).get("60")
        lines.append(f"| {LABELS[m]} | {report['counts'].get(m, 0)} | " + (f"{avg:.2f} |" if avg is not None else "— |"))
    lines.extend(["", "原 48 题已曝光。本报告属于九组正式比较后的探索性补充，树组只按计数结果预定的时点作主要对照。"])
    return "\n".join(lines) + "\n"


def timing_figure(report):
    """A standalone SVG from actual delivered checkpoints, without server plotting dependencies."""
    curves = report["mean_Cmax_by_second"]
    if not curves:
        return None
    all_times = sorted({int(t) for curve in curves.values() for t in curve})
    values = [v for curve in curves.values() for v in curve.values()]
    low, high = min(values), max(values)
    pad = max(1, (high-low) * .08)
    low, high = low-pad, high+pad
    def x(t):
        return 84 + 610 * int(t) / 60
    def y(v):
        return 326 - 224 * (v-low) / (high-low)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="420" viewBox="0 0 1040 420">',
             '<title>早切换四组的 60 秒已交付工期</title><rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#193b48">',
             '<text x="35" y="40" font-size="24">早切换四组的 60 秒已交付工期</text>',
             '<text x="35" y="72" font-size="14">逐时点平均工期（越低越好）。圆点为实测值；图例列出各组已合格场数。</text>']
    for i in range(5):
        v = low + i * (high-low) / 4
        parts.extend([f'<path d="M84 {y(v):.2f}h610" stroke="#dbe6e8"/>',
                      f'<text x="74" y="{y(v)+5:.2f}" text-anchor="end">{v:.1f}</text>'])
    for t in all_times:
        parts.append(f'<text x="{x(t):.2f}" y="357" text-anchor="middle">{t}</text>')
    parts.append('<text x="388" y="397" text-anchor="middle">已用时间（秒）</text>')
    for i, (m, color) in enumerate(zip((*COUNT, *TREE), ("#086869","#195bab","#a45414","#784d92"), strict=True)):
        if m not in curves:
            continue
        pts = [(x(t),y(v)) for t,v in sorted(curves[m].items(), key=lambda it:int(it[0]))]
        coords = " ".join(f"{a:.2f},{b:.2f}" for a,b in pts)
        dash = 'stroke-dasharray="7 4"' if m in TREE else ""
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>')
        parts.extend(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3" fill="white" stroke="{color}"/>' for a,b in pts)
        row = 130 + 40*i
        parts.extend([f'<path d="M721 {row}h29" stroke="{color}" stroke-width="3" {dash}/>',
                      f'<text x="760" y="{row+5}" font-size="15">{escape(LABELS[m])} · {report["counts"][m]} 场</text>'])
    return "\n".join([*parts, "</g></svg>"]) + "\n"
