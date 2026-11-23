"""CPU admission and observation for one Stop8 block on a shared host."""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import select
import subprocess
import sys
from time import perf_counter, sleep

from experiments.alns_cpu_lane import lock_records, sampled_load, topology
from experiments.offline_job_state import durable
from experiments.random_acquisition import exclusive
from experiments.server_resource_snapshot import cpu_counters, cpu_delta


PERIOD = .25


class CpuUnavailable(InterruptedError):
    """The admission window ended before a suitable shared core became free."""


def counters(cpus):
    data = cpu_counters(Path('/proc/stat').read_text())
    return [data[f'cpu{cpu}'] for cpu in cpus]


def optional_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def inventory(allowed, reference):
    raw = Path('/proc/cpuinfo').read_text()
    physical = {}
    for cpu in sorted(allowed):
        try:
            info = topology(cpu, cpuinfo=raw)
        except (OSError, KeyError, ValueError):
            continue
        physical.setdefault(tuple(info['thread_siblings']), info)
    candidates = [info for info in physical.values()
                  if all(info[key] == reference[key] for key in ('model', 'static_attributes'))
                  and len(info['thread_siblings']) == len(reference['thread_siblings'])]
    return candidates, list(physical.values())


def ranked(candidates, samples, reference):
    rows = []
    for info in candidates:
        busy = [sample.get(f'cpu{cpu}', {}).get('busy_percent')
                for sample in samples for cpu in info['thread_siblings']]
        if None not in busy and max(busy) <= 10:
            rows.append((info['package_id'] != reference['package_id'],
                         info['numa_node'] != reference['numa_node'], max(busy),
                         info['logical_processor'], info))
    return sorted(rows)


@contextmanager
def lane(root, locks, state, reference, allowed, excluded, session):
    deadline = perf_counter() + 300
    candidates, physical = inventory(allowed, reference)
    attempt = 0
    while perf_counter() < deadline:
        if (root / 'STOP').exists():
            raise InterruptedError('stop requested before CPU admission')
        state.update(state='WAITING_RESOURCE', phase='shared CPU admission')
        samples = [sampled_load(), sampled_load()]
        choices = ranked(candidates, samples, reference)
        reason = 'no idle compatible unleased core and separate monitor CPU'
        for _, _, _, cpu, info in choices:
            if set(info['thread_siblings']) & excluded:
                continue
            peers = [p for p in physical if cpu not in p['thread_siblings']]
            if not peers:
                continue
            monitor_load = {p['logical_processor']: samples[-1].get(f"cpu{p['logical_processor']}", {}).get('busy_percent')
                            for p in peers}
            monitor = min(monitor_load, key=lambda k: (101 if monitor_load[k] is None else monitor_load[k], k))
            with ExitStack() as held:
                try:
                    project = held.enter_context(exclusive(locks / 'random-acquisition.lock'))
                    path = locks / f"physical-cpu-{info['package_id']}-{info['core_id']}.lock"
                    lease = held.enter_context(exclusive(path))
                except BlockingIOError:
                    continue
                if lock_records(path) != [{'pid': os.getpid(), 'kind': 'FLOCK'}]:
                    continue
                confirm = sampled_load()
                if not ranked([info], [confirm], reference):
                    continue
                durable(session / 'admission.json', {'hardware': info, 'monitor_cpu': monitor,
                    'pre_lock': [{str(k): s.get(f'cpu{k}') for k in info['thread_siblings']} for s in samples],
                    'post_lock': {str(k): confirm.get(f'cpu{k}') for k in info['thread_siblings']},
                    'scans': attempt + 1, 'allowed': sorted(allowed)})
                yield info, monitor, (project.fileno(), lease.fileno())
                return
        attempt += 1
        state.update(state='WAITING_RESOURCE', phase=reason)
        sleep(min(5, max(0, deadline - perf_counter())))
    raise CpuUnavailable('no suitable shared CPU after 5 minutes')


@contextmanager
def bind(cpu):
    masks = {}
    try:
        for task in Path('/proc/self/task').iterdir():
            tid = int(task.name)
            try:
                masks[tid] = os.sched_getaffinity(tid)
                os.sched_setaffinity(tid, {cpu})
            except ProcessLookupError:
                continue
        yield
    finally:
        for tid, mask in masks.items():
            try:
                os.sched_setaffinity(tid, mask)
            except ProcessLookupError:
                continue


def sampler(output, monitor, cpus):
    os.sched_setaffinity(0, {monitor})
    samples = [[perf_counter(), counters(cpus)]]
    durable(output / 'monitor-ready.json', {'affinity': sorted(os.sched_getaffinity(0))})
    while not select.select([sys.stdin], [], [], PERIOD)[0]:
        samples.append([perf_counter(), counters(cpus)])
    samples.append([perf_counter(), counters(cpus)])
    durable(output / 'cpu-samples.json', {'cpus': cpus, 'samples': samples})


@contextmanager
def watch(output, monitor, cpus):
    output.mkdir()
    with (output / 'monitor.log').open('wb') as log:
        proc = subprocess.Popen([sys.executable, '-B', '-m', __name__, '--sample', str(output),
                                 str(monitor), *map(str, cpus)], stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = perf_counter() + 5
            while not (output / 'monitor-ready.json').exists():
                if proc.poll() is not None or perf_counter() >= deadline:
                    raise RuntimeError('separate CPU monitor failed to start; inspect monitor.log')
                sleep(.025)
            yield
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def own_cpu_seconds():
    from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
    return sum(r.ru_utime + r.ru_stime for r in (getrusage(RUSAGE_SELF), getrusage(RUSAGE_CHILDREN)))


def summarize(samples, cpus, started, finished, own_seconds, queue_ms):
    windows = []
    inside = [row for row in samples if started <= row[0] <= finished]
    spans = [b[0] - a[0] for a, b in zip(inside, inside[1:])]
    covered = bool(inside and inside[0][0] - started <= .75 and finished - inside[-1][0] <= .75
                   and max(spans, default=0) <= .75)
    selected, sibling = map(str, cpus)
    anchor = 0
    for index, (at, current) in enumerate(inside[1:], 1):
        since, previous = inside[anchor]
        if at - since < 1:
            continue
        before, after = (dict(zip((selected, sibling), sample, strict=True)) for sample in (previous, current))
        data = cpu_delta(before, after)
        windows.append({'from_s': since-started, 'to_s': at-started,
                        'selected_pct': data[selected]['busy_percent'], 'sibling_pct': data[sibling]['busy_percent']})
        anchor = index
    covered = covered and bool(windows) and all(w['sibling_pct'] is not None and
                                                 w['selected_pct'] is not None for w in windows)
    reasons = []
    if any(all(w['sibling_pct'] is not None and w['sibling_pct'] >= 20 for w in pair)
           for pair in zip(windows, windows[1:])):
        reasons.append('sibling_busy_in_two_consecutive_windows')
    unexplained = None
    delayed = queue_ms is not None and queue_ms >= 100
    if covered:
        before, after = inside[0][1][0], inside[-1][1][0]
        ticks = sum(after[k] - before[k] for k in (0, 1, 2))
        # Over-subtract our whole trial time, including the unsampled edge intervals.
        unexplained = max(0, ticks / os.sysconf('SC_CLK_TCK') - own_seconds)
        if unexplained >= max(.75, .02 * (finished - started)):
            reasons.append('selected_CPU_unaccounted_user_or_system_time')
        elif delayed and unexplained >= .25:
            reasons.append('selected_CPU_unaccounted_time_with_native_runqueue_delay')
    return {'coverage_passed': covered, 'samples_in_trial': len(inside), 'windows': windows,
            'unaccounted_cpu_seconds_lower_estimate': unexplained, 'own_cpu_seconds': own_seconds,
            'max_native_runqueue_ms': queue_ms, 'queue_delay_flag': delayed,
            'contention_reasons': reasons}


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--sample':
        raise SystemExit('expected --sample OUTPUT MONITOR CPU SIBLING')
    sampler(Path(sys.argv[2]), int(sys.argv[3]), list(map(int, sys.argv[4:])))
