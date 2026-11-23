# 算法实现与运行参数

## 计数规则选组修复（G2）

### 1. ECT 构造

维护每个作业的下一道工序、作业已安排部分的完工时刻，以及各机器队尾的完工时刻。每轮枚举各作业当前可安排的第一道工序和它的合格机器。对于工序和机器的一个组合，最早开始是该作业前序完工与机器队尾完工两者的较大值，最早完工再加此机的加工时长。

将最早完工的一对追加到机器队尾。平局依次按作业编号、机器编号和开始时刻决定。更新所选作业和机器的完工时刻，直到全部工序安排完毕。作业前序与机器互斥在构造时成立。 [ECT 源码](../engine/baseline.py)

### 2. 生成候选并选择

时间窗与并行块源自 Kasapidis 等的 ALNS-CP。该文第 4.1.1 节定义时间窗，第 4.1.2 节用时间重叠关系扩展并行块。本文每轮各生成一个半尺度候选，对并行块按目标道数截断，再二选一。原文还使用其他算子、精英池与算子调度机制。 [ALNS-CP 原文](https://doi.org/10.1016/j.ejor.2024.08.010)

从本场种子派生两条候选生成序列及独立的平局随机序列。设当前工期为 `H`，工序数为 `N`，当前规模参数为 `rho`。

| 候选 | 构造方式 |
| --- | --- |
| 半尺度时间窗 | 窗宽 `H × rho / 2`。在可容纳完整窗口的范围内均匀抽取左端点，释放与窗口相交的工序，端点相接算相交。 |
| 半尺度并行块 | 均匀选一道工序作为起点，按工序时间区间相交关系用队列扩展。每次打乱新邻居，最多保留 `ceil(N × rho / 2)` 道。队列耗尽时可提前结束。 |

**计数规则**对每组累加释放工序的原始合格机器数，选合计较小的一组，平局均匀随机。计数发生在 CP 安全删选项之前。释放 12 道、合计 24 个组合的组会优先于释放 10 道、合计 40 个组合的组。这个计数近似刻画子问题规模，候选真正的改善空间由修复结果体现。 [时间窗](../engine/alns_neighborhoods.py) · [并行块截断](../engine/experiments/candidate_study_pool.py) · [候选和选组](../engine/experiments/tree8_confirmation_runtime.py)

### 3. CP-SAT 修复

1. **保留决策。** 全部作业内的前序关系保持。未释放工序保留原机器和该机器上彼此的相对顺序，时刻可以移动。释放工序可在合格机器上重新插入。
2. **收紧时间域。** 用作业前序边与保留机器顺序边构成 DAG。释放工序用最短合法加工时长，其他工序用固定加工时长。正向和反向最长路径给出每道工序的必要头部和尾部时长。若某个机器选项的 `头部 + 加工 + 尾部 > H`，则在工期上界为 `H` 的子问题中删去该选项。
3. **紧凑建模。** 多个机器选项用可选区间，且恰选一个。固定时长使用 `end = start + p`，只剩一台机器的释放工序用必选区间，仍可改变在机器上的位置。加作业前序、保留的机器顺序及每台机器的 `NoOverlap`。目标 `Cmax` 取各作业末工序完成时刻的最大值。
4. **求解与交付。** 用当前排程给 CP-SAT 提示，工期上界为当前工期。求解使用 1 worker，原生 presolve 关闭。改善帧和最终返回解均经过可行性及保留顺序检查，严格改善且及时交付的排程才更新当前解。

[安全收紧](../engine/experiments/cp_preprocess_model.py) · [修复器 C](../engine/experiments/cp_rebuild_model.py) · [逐解观察与验证](../engine/experiments/cp_rebuild_observer.py) · [独立验解](../engine/fjsp.py)

### 4. 超参数与计时

| 参数 | 冻结设置 |
| --- | --- |
| 整场预算 | 单逻辑 CPU，60,000 ms，构造、选组、建模、搜索、验解、交付和回收均计入 |
| 单次修复 | 固定目标 3,000 ms，实际截止受外层剩余时间限制。预处理和建模消耗同一预算，绝对与相对 gap 停止阈值均为 0 |
| 邻域初始比例 | `rho = 0.20`，范围 `[0.05, 1.00]`，两个候选都使用 `rho / 2` |
| 规模反馈 | 每 20 次修复更新一次。若有 `k` 次返回局部 `OPTIMAL`，将 `rho + 0.10 × (2k − 20) / 20` 截到允许范围 |
| 种子 | 42、43、44，各组在同实例同种子配对。候选生成与 CP 均继承本场种子 |
| 截止保护 | 父进程每 10 ms 检查交付，发布预留 1,200 ms，停止预留 600 ms，返回预留 100 ms。修复进程还在发布截止前预留 1,200 ms 收尾 |

[完整参数](../engine/plan.json) · [规模反馈](../engine/experiments/alns_layers_runtime.py) · [时间与交付控制](../engine/experiments/stop8_guard.py)

## 13 种运行配置

每种配置每场有 60 秒。表中“双候选”均指半尺度时间窗和半尺度并行块。局部修复每轮只用选中的一组。G1 至 G9 是原实验的组号，G10 至 G13 是报告给四种早切换配置分配的组号。命令和记录使用下表的运行标识。 [60 秒结果](benchmark.md) · [运行命令](reproduction.md)

| 组 | 展示名称 | 起点与搜索对象 | 选择与求解 |
| --- | --- | --- | --- |
| G1 | GLNSA 35 秒 → 计数规则选组修复 | 从零运行 GLNSA，取 35 秒时本场已发布的最好排程，后段生成双候选 | 组合数少者，局部 CP-SAT，单次至多 3 秒 |
| G2 | 计数规则选组修复 | ECT，双候选 | 组合数少者，局部 CP-SAT，单次至多 3 秒 |
| G3 | GLNSA | 随机种群，工序序列和机器指派邻域 | 种群全局搜索与关键路径禁忌搜索，独立运行整场 |
| G4 | GLNSA 20 秒 → 计数规则选组修复 | 与 G1 相同，20 秒交接 | 与 G1 相同 |
| G5 | GLNSA 35 秒 → 树模型选组修复 | 与 G1 相同，35 秒交接 | 冻结树模型选高分组，同一局部 CP-SAT，单次至多 3 秒 |
| G6 | 树模型选组修复 | ECT，双候选 | 与 G5 相同 |
| G7 | 旧候选池选组修复 | ECT，旧结构候选池，后续随机释放 64 道 | 冻结表格模型排序，旧局部 CP-SAT，单次至多 1 秒 |
| G8 | 全模型 OR-Tools CP-SAT | ECT 提供初解、提示和上界，全部工序一次放开 | 一次全模型 CP-SAT 搜索 |
| G9 | 全模型 MiniZinc＋Chuffed | ECT 提供初解、提示和上界，全部工序进入 MiniZinc 模型 | MiniZinc 编译，Chuffed 搜索，逐个接收中间解 |
| G10 | GLNSA 5 秒 → Count-LNS | 本场 GLNSA 的 5 秒排程，之后生成双候选 | 同 G2 |
| G11 | GLNSA 10 秒 → Count-LNS | 本场 GLNSA 的 10 秒排程，之后生成双候选 | 同 G2 |
| G12 | GLNSA 5 秒 → Tree-LNS | 本场 GLNSA 的 5 秒排程，之后生成双候选 | 同 G6 |
| G13 | GLNSA 10 秒 → Tree-LNS | 本场 GLNSA 的 10 秒排程，之后生成双候选 | 同 G6 |

七种热启动配置均从零运行 GLNSA，交接点按本场起始时间计算。程序传入截至交接时已验证并发布的最好排程。后段把邻域比例重设为 0.20，使用本场剩余时间。进程启动、交接和验解均计入时间。第 5、10、20、35 秒是交接时刻，首个局部求解稍后发生。 [交接实现](../engine/experiments/stall80_warmstart_runtime.py) · [修复配置](../engine/plan.json)

| 组 | 运行标识 | 组 | 运行标识 |
| --- | --- | --- | --- |
| G1 | `Stall35-MinOptions2` | G8 | `full-cp-sat-hinted` |
| G2 | `MinOptions2` | G9 | `Chuffed` |
| G3 | `Stall80` | G10 | `Stall05-MinOptions2` |
| G4 | `Stall20-MinOptions2` | G11 | `Stall10-MinOptions2` |
| G5 | `Stall35-Tree2WithoutClock` | G12 | `Stall05-Tree2WithoutClock` |
| G6 | `Tree2WithoutClock` | G13 | `Stall10-Tree2WithoutClock` |
| G7 | `active-strict` | | |

### 树模型怎样选组（G5、G6、G12、G13）

把当前排程和每个候选转成特征，包括释放工序比例、关键与近关键工序覆盖、合格机器数、机器负载、释放时段跨度和松弛下界余量。标签是一次限时修复的工期改善百分比除以该次完整修复请求的秒数，再对同候选的 CP 种子取平均。训练用 64 道 SD2-mix 实例，两个规模各 32 道。特征按训练集均值与尺度标准化，80 棵浅树（训练深度上限 3）的叶子得分求和。打分使用排程和候选的特征，候选生成和单次 CP 预算沿用 Count-LNS 的设置。每轮选高分组，精确同分时均匀选取。如果下界表明两组均无法严格改善，两组得分都置零，仍按平局规则修复一组。所有测试题共用随仓库提供的冻结权重。 [特征](../engine/experiments/candidate_study_pool.py) · [打分](../engine/experiments/candidate_study_model.py) · [模型、训练实例与参数](../engine/weights/tree-without-clock.json)

### GLNSA 怎样搜索（G3 与七种热启动的前段）

GLNSA 是 Escamilla Serna 等提出的全局–局部邻域搜索算法。本项目根据作者随文发表的 MATLAB R2015a 附件移植为 C++17，沿用附件的解码方式。一个个体保存工序序列和各道工序的机器指派。序列中第几次读到某作业，就安排该作业的第几道工序。开工时刻取作业前序完工和所选机器队尾完工二者的较大值。 [原文](https://doi.org/10.7717/peerj-cs.574) · [作者 MATLAB 代码](https://doi.org/10.7717/peerj-cs.574/supp-1)

| 环节 | 本次执行设置 |
| --- | --- |
| 初始化与选择 | 每轮 40 个个体。随机打乱作业序列，各工序均匀抽取合格机器。每代保留最好的 2 个个体，其余位置通过二元锦标赛填充。 |
| 全局邻域 | 每个非精英个体生成两个邻居，保留较优者。生成工序序列的概率为插入 0.50、交换 0.25、路径重连 0.25。交换连续做 8 次，每次选不同作业的位置。每个邻居另以 0.10 的概率，给随机选出的 `floor(N/2)` 道工序重新抽取合格机器。 |
| 局部邻域 | 沿一条关键路径考察工序换机。第 `g` 代每个非精英个体最多走 `floor(g/2)` 个禁忌步，禁忌期限为本步路径工序数加被移动工序的合格机器数。 |
| 停止与重启 | 作者附件对应的控制配置为最多 250 代、停滞阈值 40。正式限时配置取消代数上限，把停滞阈值设为 80。计数从 1 开始，代末种群最好值严格刷新轮内记录时重置为 1。达到阈值且仍有时间就重建种群，跨轮保留已交付最好解。 |
| 随机种子与时间 | 第 `r` 轮（从 0 起）的种子为 `(本场种子 + r × 0x9e3779b9) mod 2^32`。独立 GLNSA 的原生搜索截止为第 57.6 秒，余时预留验证与回收。热启动按第 5、10、20 或 35 秒交接。 |

已解码的候选可随时发布改善，停滞计数按代末种群更新。读者可分别追踪交付成绩和重启时点。 [C++ 搜索与解码](../engine/experiments/glnsa.cpp) · [限时及重启参数](../engine/experiments/glnsa_runtime.py)

### 旧候选池选组修复（G7）

旧版从 ECT 出发，最多选 6 个锚点，围绕锚点生成阻塞工序对及随机配对，再生成上限 4、8 道的阻塞链与随机扩展。两次随机打乱各取 16、32 道工序。另从最多 6 个覆盖不同阻塞位置的核心各扩展上限 1、2、4、8 道的链，按释放工序集合去重。冻结表格模型给整池排序，同轮从同一排程依次尝试候选，取得严格改善后从该轮最好排程重建池。 [旧池的配对和随机组](../engine/legacy/experiments/v3_dev_replay.py) · [覆盖候选](../engine/legacy/experiments/v4_multiround.py) · [轮内与轮间更新](../engine/legacy/iterated_repair.py)

排序器使用种子 42 对应的权重，输入预算特征固定为 200 ms。实际每次 CP 修复目标为 1,000 ms。旧局部阶段结束后，若有余时，均匀无放回抽取 64 道工序并重复修复，保留顺序的下界已达到当前工期就跳过该组。CP 使用 1 个求解线程，原生预处理关闭。本次运行器在整场截止前 650 ms 结束旧流程，为独立交付和回收留出时间。G7 对照的是旧版的整体效果，涉及候选池、排序模型、修复时限和随机续搜。 [旧版配置](../engine/legacy/experiments/v4_active_baseline.json) · [本次运行入口](../engine/experiments/final60_worker.py)

### OR-Tools CP-SAT 怎样调用（局部修复与 G8）

通过 Python 包 `ortools.sat.python.cp_model` 构建 `CpModel`，用 `CpSolver.solve(model, callback)` 接收改善排程。双候选组使用前述紧凑修复器。G8 则在发布 ECT 后一次性把全部工序传给通用模型：每个合格的工序–机器组合建可选区间，每道工序恰选一种，每台机器施加 `NoOverlap`，保留全部作业前序。变量时域为 0 至 ECT 工期，目标最小化最大完工时刻。用 ECT 提示开始、结束、机器选择及工期。 [局部模型](../engine/experiments/cp_rebuild_model.py) · [局部解回调](../engine/experiments/cp_rebuild_observer.py) · [全模型与提示](../engine/repair.py)

G8 的正式配置为 OR-Tools 9.15.6755，`num_search_workers=1`，`random_seed` 取本场种子，`cp_model_presolve=False`，绝对和相对 gap 阈值均为 0。8 道已曝光开发题 × 3 种子筛选 presolve 开关，按题等权的工期／ECT 均值为关 0.93825、开 0.94430，随后锁定配置。G8 给模型的截止为整场截止前 0.85 秒，建模后才把余下秒数写入 `max_time_in_seconds`。局部修复则把该参数限制在单次预算的剩余部分。 [G8 正式入口](../engine/experiments/final60_worker.py) · [开发筛选凭据](../results/audit.json)

### MiniZinc 和 Chuffed 怎样调用（G9）

Python 把实例转为工序、机器、加工模式和作业前序的整数索引，同时写入 ECT 提示。MiniZinc 2.11.0 负责编译模型并启动 Chuffed 0.14.0。每道工序恰选一个加工模式；每台机器用容量 1 的 `cumulative`，需求为所选模式的 0/1 指示量。模型用作业前后缀最短时间收紧各开始时域，用机器负载及全局容量约束加强传播，以 ECT 为上界。模型提示机器选择和开始时刻，并给出依次搜索时长、机器选择、开始时刻的注解；命令启用 `--free-search`，允许求解器采用自身搜索策略。 [模型与搜索注解](../engine/experiments/chuffed_fjsp_tight.mzn) · [输入转换和解码](../engine/experiments/chuffed_development_probe.py)

下面是 MiniZinc 的调用参数。路径相对于运行器的工作目录。`remaining_ms` 是写完输入后的剩余毫秒数，向下取整后减去 2,000。建模、编译、搜索、解码和排程检查都计入整场时间。计分采用截止前交付的最好可行排程。 [进程调用与排程检查代码](../engine/experiments/final60_worker.py)

```bash
minizinc --solver org.chuffed.chuffed --random-seed "$seed" --free-search \
  --json-stream --intermediate-solutions --statistics --time-limit "$remaining_ms" \
  --cleanup-time-limit 300 experiments/chuffed_fjsp_tight.mzn "$trial_dir/model_input.json"
```

## 文献与借用范围

1. Kasapidis, G. A., Paraskevopoulos, D. C., Mourtos, I., & Repoussis, P. P. (2025). *A unified solution framework for flexible job shop scheduling problems with multiple resource constraints*. European Journal of Operational Research, 320(3), 479–495. 本项目用第 4.1 节中的时间窗和并行块构造候选，另行设计半尺度双候选、组内截断、选择规则与限时修复流程。 [文章与 DOI](https://doi.org/10.1016/j.ejor.2024.08.010) · [开放全文](https://openaccess.city.ac.uk/id/eprint/33482/16/1-s2.0-S0377221724006283-main.pdf)
2. Escamilla Serna, N. J., Seck-Tuoh-Mora, J. C., Medina-Marin, J., Hernandez-Romero, N., Barragan-Vite, I., & Corona Armenta, J. R. (2021). *A global-local neighborhood search algorithm and tabu search for flexible job shop scheduling problem*. PeerJ Computer Science, 7, e574. 本项目根据作者附件移植为 C++17，在独立运行和热启动中采用相同的限时搜索。 [文章与 DOI](https://doi.org/10.7717/peerj-cs.574) · [作者代码附件](https://doi.org/10.7717/peerj-cs.574/supp-1)

## 方法简称

三种项目方法都使用大邻域搜索（Large Neighborhood Search，LNS）。短名用于图例和组合方法名称。

| 中文全名 | 英文短名 | 英文说明 |
| --- | --- | --- |
| 计数规则选组修复 | **Count-LNS** | Large Neighborhood Search with Option-Count Selection |
| 树模型选组修复 | **Tree-LNS** | Large Neighborhood Search with Tree-Based Selection |
| 旧候选池选组修复 | **Legacy-LNS** | Large Neighborhood Search with the Legacy Candidate Pool |

Count 指候选组中合格工序–机器组合的总数。Tree 指给同一候选池打分的树模型。Legacy 指项目早期的候选池和修复流程。组合名按执行顺序书写，例如 **GLNSA (35 s) → Count-LNS** 表示 GLNSA 运行 35 秒，再将已交付的最好排程交给计数规则选组修复。

