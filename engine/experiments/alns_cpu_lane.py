"""A fixed Linux physical-core lane alongside the already running matched job.

Never changes the peer's affinity, files or processes. Unknown topology, peer
identity or legacy-lock ownership fails closed before another measured trial.
"""

from contextlib import ExitStack
import os
from pathlib import Path
from time import perf_counter, sleep, time

from experiments import conditional_selection_cost_pilot as cost
from experiments import random_acquisition as acquisition
from experiments.offline_job_state import durable, process_identity
from experiments.server_resource_snapshot import cpu_counters, cpu_delta


def cpu_list(value):
    result = set()
    for part in value.strip().split(","):
        ends = [int(v) for v in part.split("-")]
        cost.check(len(ends) in (1, 2) and min(ends) >= 0 and ends[0] <= ends[-1], "invalid CPU list")
        result.update(range(ends[0], ends[-1] + 1))
    cost.check(result, "empty CPU list")
    return sorted(result)


def topology(cpu, *, sys_cpu=Path("/sys/devices/system/cpu"), cpuinfo=None):
    path = Path(sys_cpu) / f"cpu{cpu}"
    package = int((path / "topology/physical_package_id").read_text())
    core = int((path / "topology/core_id").read_text())
    siblings = cpu_list((path / "topology/thread_siblings_list").read_text())
    nodes = sorted(int(p.name[4:]) for p in path.glob("node[0-9]*"))
    cost.check(package >= 0 and core >= 0 and cpu in siblings and len(nodes) == 1, "CPU package/core/NUMA topology unknown")
    if cpuinfo is None:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    records = {}
    for section in cpuinfo.strip().split("\n\n"):
        fields = dict(line.split(":", 1) for line in section.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields:
            records[int(fields["processor"])] = fields
    fields = records[cpu]
    model = {key: fields.get(key) for key in ("vendor_id", "cpu family", "model", "stepping", "model name")}
    cost.check(all(v is not None for v in model.values()), "CPU model identity unknown")
    optional = {}
    for relative in ("topology/core_type", "cpu_capacity", "cpufreq/cpuinfo_max_freq", "cpufreq/scaling_governor"):
        file = path / relative
        optional[relative] = file.read_text().strip() if file.exists() else None
    return {"logical_processor": cpu, "package_id": package, "core_id": core,
            "thread_siblings": siblings, "numa_node": nodes[0], "model": model, "static_attributes": optional}


def compatible(peer, candidate):
    return (peer["package_id"] == candidate["package_id"] and peer["numa_node"] == candidate["numa_node"]
            and peer["core_id"] != candidate["core_id"]
            and not set(peer["thread_siblings"]) & set(candidate["thread_siblings"])
            and len(peer["thread_siblings"]) == len(candidate["thread_siblings"])
            and peer["model"] == candidate["model"] and peer["static_attributes"] == candidate["static_attributes"])


def peer_state(policy, *, proc=Path("/proc"), affinity_reader=None, now=None):
    root, bundle = Path(policy["peer_run"]), Path(policy["peer_bundle"])
    cost.check(cost.digest(bundle / "manifest.json") == policy["peer_manifest_sha256"], "running first-layer bundle identity differs")
    cost.check(cost.read(root / "identity.json")["manifest_sha256"] == policy["peer_manifest_sha256"], "first-layer run identity differs")
    status = cost.read(root / "status.json")
    owner = status["owner"]
    observed = process_identity(owner["pid"], proc=proc)
    alive = (owner.get("start_ticks") is not None and owner.get("boot_id") is not None and
             all(owner.get(k) == observed.get(k) for k in ("pid", "start_ticks", "boot_id"))
             and observed.get("state") not in (None, "Z", "X"))
    if alive:
        cost.check((proc / str(owner["pid"])).stat().st_uid == os.getuid(), "peer process UID differs")
        cost.check((proc / str(owner["pid"]) / "cwd").resolve() == bundle.resolve(), "peer working directory differs")
        command = (proc / str(owner["pid"]) / "cmdline").read_bytes().split(b"\0")
        cost.check(b"experiments.alns_matched_job" in command and b"run" in command, "peer is not the authorized first-layer job")
        cost.check((time() if now is None else now) - status["heartbeat_epoch"] <= 90, "first-layer heartbeat stale")
        allowed = sorted((affinity_reader or os.sched_getaffinity)(owner["pid"]))
        cost.check(len(allowed) == 1, "first layer has not confirmed a single CPU binding")
        cpu = allowed[0]
    else:
        cost.check(status["state"] in ("COMPLETED", "PAUSED", "FAILED"), "first-layer owner absent without a terminal state")
        environments = list((root / "sessions").glob("*/environment.json"))
        cost.check(environments, "first-layer bound CPU evidence missing")
        latest = max(environments, key=lambda p: p.stat().st_mtime_ns)
        cpu = cost.read(latest)["affinity"]["logical_processor"]
    return {"run": str(root), "bundle": str(bundle), "cpu": cpu, "owner": owner,
            "alive": alive, "state": status["state"], "completed": status.get("completed"),
            "heartbeat_epoch": status.get("heartbeat_epoch"), "observed_at_epoch": time(), "observed_perf_counter": perf_counter()}


def lock_records(path, *, proc=Path("/proc")):
    stat = Path(path).stat()
    records = []
    for line in (proc / "locks").read_text().splitlines():
        parts = line.split()
        if (len(parts) < 8 or parts[1] not in ("FLOCK", "POSIX") or
                parts[2:4] != ["ADVISORY", "WRITE"] or parts[6:8] != ["0", "EOF"]):
            continue
        major, minor, inode = parts[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) == (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
            records.append({"pid": int(parts[4]), "kind": parts[1]})
    return records


def flock_owners(path, *, proc=Path("/proc")):
    # Some backends expose a whole-file flock as a POSIX lock. Peer ownership
    # remains checkable; our inherited core lease itself requires native FLOCK.
    return [row["pid"] for row in lock_records(path, proc=proc)]


def sampled_load(proc=Path("/proc"), sleeper=sleep):
    first = cpu_counters((proc / "stat").read_text())
    sleeper(.25)
    return cpu_delta(first, cpu_counters((proc / "stat").read_text()))


def candidate_cpus(allowed, peer_topology, first_load, second_load, *, topology_reader=topology, max_busy=10):
    candidates, rejected = [], []
    for cpu in sorted(allowed):
        try:
            candidate = topology_reader(cpu)
            if not compatible(peer_topology, candidate):
                continue
            busy = []
            for sibling in candidate["thread_siblings"]:
                for sample in (first_load, second_load):
                    row = sample.get(f"cpu{sibling}", {})
                    cost.check(row.get("status") == "MEASURED" and row.get("busy_percent") is not None, "CPU load unknown")
                    busy.append(row["busy_percent"])
            if max(busy) <= max_busy:
                candidates.append((max(busy), cpu, candidate))
        except (OSError, ValueError, KeyError) as error:
            rejected.append({"cpu": cpu, "reason": str(error)})
    return sorted(candidates, key=lambda row: (row[0], row[1])), rejected


class CpuLane:
    def __init__(self, bundle, root, cfg, session, reports):
        self.root, self.session, self.policy = Path(root), Path(session), cfg["cpu_policy"]
        self.stack = ExitStack()
        self.binding, self.core_lock, self.global_lock = None, None, None
        self.global_owned = False

    def __enter__(self):
        import fcntl
        try:
            cost.check(os.name == "posix" and hasattr(os, "sched_getaffinity"), "parallel CPU lane requires Linux")
            allowed = sorted(os.sched_getaffinity(0))
            peer = peer_state(self.policy)
            peer_topology = topology(peer["cpu"])
            first_load, second_load = sampled_load(), sampled_load()
            candidates, rejected = candidate_cpus(allowed, peer_topology, first_load, second_load,
                                                   max_busy=self.policy["max_initial_sibling_busy_percent"])
            saved = self.root / "cpu-binding.json"
            if saved.exists():
                old = cost.read(saved)
                cost.check(old["peer_topology"] == peer_topology, "peer CPU changed across resume")
                candidates = [r for r in candidates if r[1] == old["selected_topology"]["logical_processor"]]
            durable(self.session / "cpu-admission.json", {"allowed_before_binding": allowed, "peer": peer,
                    "peer_topology": peer_topology, "first_load": first_load, "second_load": second_load,
                    "candidates": [c[2] for c in candidates], "unknown_rejections": rejected})
            cost.check(candidates, "no idle allowed physical core matching peer package/NUMA/model/profile; no solver started")
            for _, cpu, selected in candidates:
                try:
                    lock_path = Path(self.policy["lock_directory"]) / f"physical-cpu-{selected['package_id']}-{selected['core_id']}.lock"
                    self.core_lock = self.stack.enter_context(acquisition.exclusive(lock_path))
                    cost.check(lock_records(lock_path) == [{"pid": os.getpid(), "kind": "FLOCK"}],
                               "core lease filesystem lacks confirmed native FLOCK inheritance; no solver started")
                    break
                except BlockingIOError:
                    if saved.exists():
                        raise ValueError("fixed CPU lane occupied; resume never changes CPU")
            cost.check(self.core_lock is not None, "all matching CPU lanes already leased")
            # Linux affinity belongs to a thread: include the already-created heartbeat.
            for task in Path("/proc/self/task").iterdir():
                os.sched_setaffinity(int(task.name), {cpu})
            from experiments.resched_qualification import configure_cpu
            configure_cpu()
            self.binding = {"policy": self.policy, "allowed_before_binding": allowed,
                            "peer_topology": peer_topology, "selected_topology": selected,
                            "logical_processor": cpu, "first_peer_observation": peer,
                            "boot_id": process_identity()["boot_id"], "binding_created_epoch": time()}
            if saved.exists():
                cost.check(old["selected_topology"] == selected, "fixed CPU topology changed across resume")
                self.binding = old
            else:
                durable(saved, self.binding)
            self.global_lock = Path(self.policy["legacy_global_lock"]).open("a+b")
            self.stack.callback(self.global_lock.close)
            self.check()
            durable(self.session / "cpu-binding.json", self.binding)
            return self
        except BaseException:
            self.stack.close()
            raise

    @property
    def pass_fds(self):
        descriptors = [self.core_lock.fileno()]
        if self.global_owned:
            descriptors.append(self.global_lock.fileno())
        return tuple(descriptors)

    def check(self):
        import fcntl
        cpu = self.binding["logical_processor"]
        for task in Path("/proc/self/task").iterdir():
            cost.check(os.sched_getaffinity(int(task.name)) == {cpu}, "parent/heartbeat CPU affinity changed")
        peer = peer_state(self.policy)
        cost.check(peer["cpu"] == self.binding["peer_topology"]["logical_processor"], "first-layer CPU binding changed")
        cost.check(topology(cpu) == self.binding["selected_topology"], "fixed selected CPU topology changed")
        if not self.global_owned:
            owners = flock_owners(self.policy["legacy_global_lock"])
            if peer["alive"] and peer["state"] not in ("FINALIZING", "COMPLETED", "PAUSED", "FAILED"):
                # Do not race the running peer by briefly taking a free global lock.
                cost.check(owners == [peer["owner"]["pid"]], "active first-layer legacy lock not confirmed; no overlap authorized")
            else:
                try:
                    fcntl.flock(self.global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.global_owned = True
                except BlockingIOError:
                    cost.check(peer["alive"] and owners == [peer["owner"]["pid"]], "legacy CPU lock held by unknown job; no overlap authorized")
        return {"peer": peer, "selected_cpu": cpu, "selected_core": self.binding["selected_topology"],
                "legacy_global_lock_mode": "owned" if self.global_owned else "verified_first_layer_overlap",
                "epoch": time(), "perf_counter": perf_counter(), "boot_id": self.binding["boot_id"]}

    def __exit__(self, *args):
        self.stack.close()
