# 复现 13 种运行配置

## 从逐场记录复核结果

在仓库根目录运行下面的命令。统计脚本只需 Python 标准库。逐场表的 `method` 可在算法说明里对应到组号，`record_sha256` 对应原服务器记录。 [实验报告](benchmark.md) · [方法与运行标识](algorithms.md)

```bash
python analysis/summarize.py
python analysis/early_switch.py
python analysis/overview.py
```

`summarize.py` 从原九组的 1,296 场重算成绩、配对与最优性差距。`early_switch.py` 用四种早切换配置的 576 场和原九组成绩重算交接时点的比较。`overview.py` 核对两批数据指纹及题目、种子和方法的完整组合，再生成首页表格。 [原九组数据](../results/observations.jsonl) · [早切换数据](../results/early-switch.jsonl) · [逐题上下界](../results/bounds.json)

运行 `python analysis/figures.py` 可重新绘图。绘图另需 Matplotlib 和一种可用的中文字体，例如 Noto Sans CJK SC 或微软雅黑。首页曲线与表格使用同一份计算结果，每个点都需要该组 144 场均有记录。 [制图代码](../analysis/figures.py)

## 所需环境

| 组件 | 版本或要求 |
| --- | --- |
| 平台 | Linux x86-64，系统能读取逻辑核、物理核和 NUMA 拓扑。需要一个 SMT 双线程核心，以及另一个可用于采样的物理核心。 |
| Python | CPython 3.14.10。 |
| Python 依赖 | OR-Tools 9.15.6755、NumPy 2.5.4 等，见依赖清单。原九组的 G7 用 PyTorch 2.14.1+cu130 在 CPU 上推理。两批运行均隐藏 GPU。 |
| C++ | C++17 编译器。原实验使用 GCC 11.4.0 和 `-O3 -DNDEBUG`，在正式计时外编译。 |
| MiniZinc / Chuffed | 官方 Linux x86-64 MiniZinc 2.11.0 捆绑 Chuffed 0.14.0。运行入口验证二者的可执行文件版本和 SHA-256。 |

[Python 版本](../.python-version) · [Python 依赖](../requirements-linux.txt) · [PyTorch 官方构建列表](https://download.pytorch.org/whl/cu130/torch/) · [MiniZinc 官方发行包](https://github.com/MiniZinc/libminizinc/releases/tag/2.11.0)

## 测试数据来源

原九组和四种早切换配置使用同一份 48 题清单。工序数在两类数据里各分 250 和 500 两档，每档 12 题。种子 42、43、44 用于重复运行求解器，合成题的生成种子在输入清单中另有记录。 [48 题清单与指纹](../engine/cases.json)

| 测试集 | 工序数 | 实例数 | 作业数 | 机器数 |
| --- | ---: | ---: | ---: | --- |
| Behnke–Geiger | 250 | 12 | 50 | 20 台 3 题，40 台 4 题，60 台 5 题 |
| Behnke–Geiger | 500 | 12 | 100 | 20 台 3 题，40 台 5 题，60 台 4 题 |
| SD2-mix | 250 | 12 | 25 | 10 台 |
| SD2-mix | 500 | 12 | 50 | 10 台 |

### Behnke–Geiger 公共基准

原始出处为 Dennis Behnke 与 Martin Josef Geiger 2012 年的技术报告 *Test Instances for the Flexible Job Shop Scheduling Problem with Work Centers*（RR-12-01-01）。项目从 FJSPLib 收录的 60 道 Behnke 实例中，取此前保留的全部 24 道 250 或 500 工序题。选题依据是规模和先前的使用记录。全部实例名称可在统一报告逐题表中查到。 [作者机构的出版目录](https://www.hsu-hh.de/logistik/research/publications) · [逐题表](benchmark.md#与最优工期的差距)

求解器读取 FJSPLib 的 `flexible-jobshop/instances/json/` 下的 Behnke JSON。本地 ZIP 的归档注释记录了提交 `8a6e36ef31263d36975e56fb481e0ba04e188d6f`，原始 ZIP 的 SHA-256 为 `66a4da92ad0208d9c65681ff18565faa29058f39180bcf69283e2635a0252068`。复现时以每个实例的文件指纹为准。 [固定版本说明](https://github.com/ScheduleOpt/benchmarks/blob/8a6e36ef31263d36975e56fb481e0ba04e188d6f/flexible-jobshop/README.md) · [固定版本归档](https://github.com/ScheduleOpt/benchmarks/archive/8a6e36ef31263d36975e56fb481e0ba04e188d6f.zip)

报告的 Behnke 上下界采自同一归档的 `solutions/bks.json`，文件 SHA-256 为 `b703cb9ba4e6aad9daecbf244fc94e781cb93336cf3948e0bd3aaddfd0e08f9a`。该文件可选，用于复跑后重新计算档案差距。已公开的逐题表包含报告采信的 24 个条目。 [逐题上下界](../results/bounds.json)

### SD2-mix 项目生成数据

参考 Wang 等人的 SD2 分布，自行生成 24 道实例。每个作业有 10 道顺序工序。每道工序的合格机器数从 1 到 10 均匀抽取，再从 10 台机器中等概率、无放回抽出对应数量的机器。每个合格工序与机器组合独立抽取 1 到 99 的整数加工时长。 [SD2 分布参照](https://arxiv.org/abs/2305.05119) · [项目生成器](../engine/experiments/sd2_reference_data.py)

两个规模都使用 `sd2-mix-v1`、NumPy 2.5.4、`test` 分区及编号 `60000` 至 `60011`。每题以分区、作业数、机器数、编号与重试序号拼出种子材料，将其 SHA-256 转为整数种子交给 PCG64DXSM。这 24 题的重试序号都是 0。冻结前按文件、规范化问题及作业重排指纹核对与既有数据的重合。Tree-LNS 使用的合成训练题属于 `train` 分区，两个规模各取编号 `20000` 至 `20031`。 [生成参数及指纹](../engine/cases.json) · [训练实例清单](../engine/weights/tree-without-clock.json)

## 输入准备与复跑

准备命令接收本地 FJSPLib 的 `flexible-jobshop` 目录并校验 Behnke 文件，按清单现场生成 SD2-mix 测试题。完整计划还用 8 道已曝光开发题做运行资格检查，其中两道合成开发题随源码提供。正式评测的 48 题及其三次求解重复在两批之间复用。

```bash
python -m pip install -r requirements-linux.txt
python -B reproduce.py prepare --benchmarks /path/to/flexible-jobshop
python -B reproduce.py check
python -B reproduce.py run --output runs/final60 --minizinc /path/to/MiniZinc-2.11.0/bin/minizinc
python -B reproduce.py run-early-switch --output runs/early-switch
```

`run` 先在开发题上检查求解和验解，为全模型 OR-Tools CP-SAT 选择预处理设置，然后运行原九组的 1,296 场。它最后执行原计划的参考搜索。`run-early-switch` 在另一输出目录运行 576 场，先跑两个 Count-LNS 时点，再跑两个 Tree-LNS 时点。两个入口可独立运行。早切换运行报告默认对照已随仓库提供的原实验配对成绩。中断后对同一批次、同一输出路径重跑命令，运行器复用已合格区组。 [原九组区组及顺序](../engine/final60-blocks.json) · [早切换区组及顺序](../engine/early-blocks.json) · [早切换原始计分参照](../engine/early-history.json)

两批均使用一个逻辑 CPU 上的一个求解线程，在另一个物理核心监测。比较时按区组轮换组内顺序，达到争用排除条件时换核重跑整个区组。所有方法统一记录第 5、10、20、35、50、60 秒前交付的最低工期。支持的配置另记录第 30 秒。 [方法、时间预留与求解器调用](algorithms.md) · [实验计分与核验](benchmark.md)

原九组的 60 秒对照名义用时 21 小时 36 分，早切换实验 9 小时 36 分，另有开发题检查、参考搜索、等待和因争用重跑的时间。输出目录包含状态、环境、逐场排程及汇总压缩包。当前运行状态看 `status.json`，恢复前的暂停或失败标记作为历史保留。两批运行成功后，可从新输出重算同样口径的合并表：

```bash
python analysis/overview.py --main-run runs/final60 --early-run runs/early-switch
```

## 复现边界

两批以各自的服务器冻结包为源码基准，保留搜索实现、模型、权重、题目、种子、区组顺序和求解参数。公开入口允许在满足版本约束的 Linux 主机上复跑，记录本机编译器和 Python 构建信息，恢复时检查运行环境一致。时间限制下交付的工期可能随硬件和运行时负载变化。 [源码与适配指纹](../engine/source-freeze.json) · [早切换原运行环境](../engine/early-environment.json)

早切换源码记录中的成绩指纹对应当时使用的文件版本。公开逐场文件的指纹以核验摘要为准。早切换使用的 576 条原实验参照，与公开逐场表逐条核对了题目、种子、方法、工期和原记录指纹。 [原实验核验摘要](../results/audit.json) · [早切换来源记录](../engine/early-source.json)

Behnke 实例及历史参考值由本地副本提供。GLNSA 移植保留原论文和作者附件的出处，原始 MATLAB 附件由其出处单独提供。 [算法与引用](algorithms.md)

复跑使用仓库里的两个冻结模型权重。已按指纹核对全部 56 份输入，核对两批 1,872 场公开成绩与服务器报告。G1 至 G6、G8 和 G9 的实际求解与交付已有 Linux 短跑记录，G7 的冻结权重与修复已有 Windows CPU 短跑记录。公开代码中的四种早切换配置也通过了 Linux 短测，每组完成一次实际 CP-SAT 修复，并用已有 576 场记录模拟验证了暂停、恢复及报告导出。完整同时间预算的成绩来自原服务器运行，展示入口尚未完成全量重跑。
