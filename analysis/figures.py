"""Render the benchmark, follow-up, and combined homepage figures."""

from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from summarize import RESULTS, ROOT, TIMES, read_json


OUT = ROOT / "docs/figures"
INK, MUTED, LINE = "#173649", "#506777", "#d9e3e8"
TEAL, RUST = "#006f70", "#a84724"
STRATA = ("behnke250", "behnke500", "sd2mix250", "sd2mix500")
STRATUM_LABELS = ("Behnke · 250", "Behnke · 500", "SD2-mix · 250", "SD2-mix · 500")


def setup():
    choices = ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei")
    families = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in choices if f in families), None)
    if font is None:
        raise SystemExit("Install a Chinese plotting font, e.g. Noto Sans CJK SC, before regenerating the PNGs.")
    plt.rcParams.update({"font.family": font, "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.unicode_minus": False,
                         "axes.spines.right": False, "axes.spines.top": False})
    OUT.mkdir(exist_ok=True)


def header(fig, title, subtitle):
    fig.text(.035, .965, title, va="top", fontsize=20, weight="bold")
    fig.text(.035, .892, subtitle, va="top", fontsize=11, color=MUTED)


def save(fig, name):
    fig.savefig(OUT / name, dpi=165, facecolor="white")
    plt.close(fig)


def objective_timeline(result):
    methods = result["methods"]
    # Vary marker sizes as well as line styles so overlapping trajectories
    # (notably the four GLNSA starts at 5–20 s) can share their true positions.
    styles = {
        "G1": ("#4774af", "--", "s", 7),
        "G2": (TEAL, "-", "o", 6),
        "G3": ("#343c50", "-", "x", 4),
        "G4": ("#96701a", "-.", "D", 9),
        "G5": ("#98597a", ":", "^", 10),
        "G6": ("#b14d25", "--", "D", 4),
        "G7": ("#666d78", ":", "v", 5),
        "G8": ("#477da6", "-.", "P", 6),
        "G9": ("#76649a", "--", "s", 5),
    }
    fig = plt.figure(figsize=(13.2, 6.5))
    header(fig, "九组求解器在 60 秒内交付的最好工期",
           "各点为同一批 48 题 × 3 种子的平均值。纵轴越低，表示到该时刻已交付的排程越短。")
    ax = fig.add_axes([.08, .21, .55, .58])
    key = fig.add_axes([.665, .21, .30, .58])
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.axis("off")
    key.text(0, 1, "方法", weight="bold", fontsize=11)
    key.text(1, 1, "60 秒平均工期", ha="right", weight="bold", fontsize=11)
    key.plot([0, 1], [.97, .97], color=LINE, lw=.8)

    # Draw larger, open symbols first; do not displace overlapping data.
    for group in sorted(styles, key=lambda g: -styles[g][3]):
        color, line, marker, size = styles[group]
        scores = methods[group]["mean_by_second"]
        times = sorted(map(int, scores))
        values = [scores[str(t)] for t in times]
        style = dict(color=color, ls=line, marker=marker, markersize=size, markerfacecolor="none",
                     markeredgewidth=1.3, linewidth=2.1 if group == "G2" else 1.65)
        ax.plot(times, values, **style)
        row = 1 - int(group[1:]) * .10
        key.plot([.005, .065, .125], [row] * 3, markevery=[1], **style)
        key.text(.15, row, f"{group}  {methods[group]['short_name']}", va="center", fontsize=10)
        key.text(1, row, f"{values[-1]:.2f}", va="center", ha="right", fontsize=10)

    ax.set(xlim=(0, 61), xticks=[0, 5, 10, 20, 30, 35, 50, 60], ylim=(578, 860),
           yticks=[600, 650, 700, 750, 800, 850], xlabel="已用时间（秒）", ylabel="平均已交付最好工期（Cmax）")
    ax.grid(axis="y", color=LINE, lw=.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    fig.text(.08, .057, "标记为实测检查点，连线只表示相邻记录的变化。各组首个共同记录在第 5 秒，第 30 秒另有 G1–G6 的实测值。",
             color=MUTED, fontsize=10)
    save(fig, "nine-methods-objective-timeline.png")


def paired(result):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.345, right=.92, bottom=.18, top=.80)
    header(fig, "计数规则选组修复与七种方法的逐题对照", "Count-LNS 为计数规则选组修复，Tree-LNS 为树模型选组修复。每行 48 题，正值表示 Count-LNS 工期更短。")
    order = (8, 9, 3, 1, 5, 4, 6)
    rng = Random(18)
    for y, arm in enumerate(order):
        group = result["comparisons"][f"G2-G{arm}"]
        vals = list(group["case_effects"].values())
        ax.scatter(vals, [y + rng.uniform(-.20, .20) for _ in vals], s=15, color="#8fabb6", alpha=.60, linewidth=0)
        ax.plot(group["interval"], [y, y], color=TEAL, lw=3, solid_capstyle="round")
        ax.scatter(group["median"], y, s=35, marker="D", edgecolor="white", lw=.6, color=TEAL, zorder=4)
        ax.text(1.012, y, f"{group['median']:+.2f}%", va="center", transform=ax.get_yaxis_transform(),
                fontsize=10, color=TEAL, weight="bold", clip_on=False)
    ax.axvline(0, color=RUST, lw=1.2, zorder=0)
    ax.set_yticks(range(len(order)), [f"G{k}   {result['methods'][f'G{k}']['short_name']}" for k in order])
    ax.invert_yaxis()
    ax.margins(x=.05)
    ax.set_xlabel("相对各行对照的 60 秒工期降幅（%）")
    ax.grid(axis="x", color=LINE, lw=.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(LINE); ax.tick_params(axis="y", length=0, pad=9)
    fig.text(.035, .045, "每题取三个配对种子的中位降幅。菱形为 48 题中位数，横线为分层自助法描述性 95% 区间。此图为追加对照。", fontsize=10, color=MUTED)
    save(fig, "paired-comparisons.png")


def heatmap(result):
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    fig.subplots_adjust(left=.36, right=.97, top=.76, bottom=.16)
    header(fig, "同一种方法的收益随题族变化", "每格 12 道题，数值为计数规则选组修复（Count-LNS）相对该行对照的题级中位工期降幅（%）。")
    order = (1, 3, 6, 8, 9)
    data = [[result["comparisons"][f"G2-G{k}"]["strata"][s]["median"] for s in STRATA] for k in order]
    cmap = LinearSegmentedColormap.from_list("effect", ["#e1a68d", "#f7f8f5", "#086568"])
    ax.imshow(data, cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=36), aspect="auto")
    for y, line in enumerate(data):
        for x, value in enumerate(line):
            ax.text(x, y, f"{value:+.2f}" if value else "0.00", ha="center", va="center",
                    color="white" if value > 21 else INK, weight="bold")
    ax.set_xticks(range(4), STRATUM_LABELS); ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [f"G{k}   {result['methods'][f'G{k}']['short_name']}" for k in order])
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.035, .06, "Tree-LNS 为树模型选组修复。深绿表示 Count-LNS 优势较大，负值表示该行对照领先。250 和 500 指工序数。", color=MUTED, fontsize=10)
    save(fig, "stratified-comparisons.png")


def handoff(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.7), sharey=True)
    fig.subplots_adjust(left=.11, right=.97, bottom=.21, top=.72, wspace=.12)
    header(fig, "GLNSA 热启动在什么时候领先？", "G1 前 35 秒使用 GLNSA，G2 从 ECT 开始。两组均用计数规则选组修复，正值表示 G1 领先。")
    for ax, family, label in zip(axes, ("behnke", "sd2mix"), ("Behnke–Geiger", "SD2-mix"), strict=True):
        ax.axhline(0, color=INK, lw=.9)
        ax.axvline(35, color=MUTED, ls=":", lw=1)
        ax.set_title(label, pad=9, fontsize=13, weight="bold")
        for size, color, marker in ((250, TEAL, "o"), (500, RUST, "s")):
            curve = result["G1_vs_G2_by_second"][family + str(size)]
            ax.plot(TIMES, [curve[str(t)]["median"] for t in TIMES], lw=2.3, color=color,
                    marker=marker, markersize=5, label=f"{size} 道工序")
        ax.set_xticks(TIMES); ax.set_xlabel("已用时间（秒）"); ax.set_xlim(3, 62)
        ax.set_ylim(-30, 21); ax.set_yticks([-30, -20, -10, 0, 10, 20])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(axis="y", color=LINE, lw=.6); ax.set_axisbelow(True)
        for k in ("left", "bottom"):
            ax.spines[k].set_color(LINE)
    axes[0].set_ylabel("G1 相对 G2 的题级中位降幅")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.09, .013), frameon=False, ncol=2)
    fig.text(.57, .05, "竖虚线为 G1 切入局部修复的时点。", color=MUTED, fontsize=10)
    save(fig, "warmstart-timeline.png")


def combined_timeline():
    from overview import overview

    rows = overview()
    styles = {
        "MinOptions2": (TEAL, "-", "o", 7),
        "Tree2WithoutClock": (RUST, "--", "x", 5),
        "Stall05-MinOptions2": ("#856b16", "-.", "v", 7),
        "Stall10-MinOptions2": ("#305daa", "-", "s", 8),
        "Stall20-MinOptions2": ("#935575", "--", "D", 7),
        "Stall35-MinOptions2": ("#598149", ":", "^", 9),
        "Stall10-Tree2WithoutClock": ("#7160ae", "--", "+", 6),
        "Stall80": ("#354253", "-", "x", 4),
        "Chuffed": ("#916548", ":", "p", 6),
        "full-cp-sat-hinted": ("#57778a", "-.", "d", 6),
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 7.6))
    fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.36, wspace=.20)
    header(fig, "60 秒内，哪种方法交付的排程更短？",
           "同一批 48 道题、每题三个种子。每个点是该方法 144 场在此时已经交付的最好工期的平均值。")
    handles = {}
    for row in sorted(rows, key=lambda r: -styles[r["method"]][3]):
        method, curve = row["method"], row["curve"]
        color, line, marker, size = styles[method]
        style = dict(color=color, ls=line, marker=marker, markersize=size, markerfacecolor="none",
                     linewidth=2 if method == "MinOptions2" else 1.6, markeredgewidth=1.2)
        handles[method], = left.plot(curve.keys(), curve.values(), **style)
        late = [t for t in curve if t >= 20]
        if min(curve[t] for t in late) <= 642:
            right.plot(late, [curve[t] for t in late], **style)
    left.set(title="全部 10 种方法", ylabel="平均工期（越低越好）",
             xlim=(3, 62), ylim=(580, 825), xticks=[5, 10, 20, 30, 35, 50, 60])
    right.set(title="20 秒后局部放大", xlim=(19, 61), ylim=(585, 642),
              xticks=[20, 30, 35, 50, 60], yticks=[590, 600, 610, 620, 630, 640])
    for ax in (left, right):
        ax.set_xlabel("已用时间（秒）")
        ax.grid(axis="y", color=LINE, lw=.8)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LINE)
    # Use one legend for both scales, pairing related series in adjacent rows.
    order = ("MinOptions2", "Tree2WithoutClock", "Stall10-MinOptions2", "Stall10-Tree2WithoutClock",
             "Stall05-MinOptions2", "Stall20-MinOptions2", "Stall35-MinOptions2", "Stall80",
             "Chuffed", "full-cp-sat-hinted")
    labels = {row["method"]: row["label"] for row in rows}
    fig.legend([handles[m] for m in order], [labels[m] for m in order], ncol=3, loc="lower center",
               bbox_to_anchor=(.5, .095), frameon=False, columnspacing=3.2, handlelength=3,
               fontsize=10.5, labelspacing=.7)
    fig.text(.075, .048, "首个共同记录在第 5 秒。连线连接实测点，两个全模型求解器在第 30 秒没有记录。",
             fontsize=10, color=MUTED)
    save(fig, "combined-timeline.png")


if __name__ == "__main__":
    setup()
    result = read_json(RESULTS / "statistics.json")
    paired(result)
    heatmap(result)
    handoff(result)
    objective_timeline(result)
    combined_timeline()
