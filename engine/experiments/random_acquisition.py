"""Detached, checkpointed CPU acquisition using the existing random-branch CP path."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from queue import Queue
import signal
import subprocess
import sys
from threading import Thread
from time import perf_counter, sleep
import traceback
from uuid import uuid4

from experiments import conditional_selection_cost_pilot as cost

ROOT = Path('.benchmark-work')
MODULE = 'experiments.random_acquisition'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def durable(path, value):
    """Replace a checkpoint atomically and fsync its directory on POSIX."""
    path = Path(path)
    # A crashed write must not leave a fixed .tmp name blocking recovery.
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def exclusive(path):
    """Kernel-owned lock: no PID guessing or stale-lock deletion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


class Budget:
    """Crash-safe conservative reservation for the next bounded operation."""
    def __init__(self, path, limit, clock=perf_counter):
        self.path, self.limit, self.clock = Path(path), limit, clock
        old = cost.read(path) if self.path.exists() else {'charged_seconds': 0}
        self.base = old['charged_seconds']
        cost.check(0 <= self.base <= limit, 'invalid cumulative budget')
        self.started = clock()

    @property
    def spent(self):
        return self.base + self.clock() - self.started

    @property
    def left(self):
        return max(0, self.limit - self.spent)

    def reserve(self, seconds, phase):
        if self.left < seconds:
            return False
        durable(self.path, {'charged_seconds': min(self.limit, self.spent + seconds),
            'limit_seconds': self.limit, 'phase': phase, 'pending_lease_seconds': seconds,
            'time_utc': datetime.now(timezone.utc).isoformat()})
        return True

    def settle(self):
        durable(self.path, {'charged_seconds': min(self.limit, self.spent),
            'limit_seconds': self.limit, 'phase': 'session_settled', 'pending_lease_seconds': 0,
            'measured_or_precharged_active_seconds': self.spent,
            'over_limit_seconds': max(0, self.spent - self.limit)})


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = cost.read(bundle / 'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        path = (bundle / name).resolve()
        cost.check(path.is_relative_to(bundle) and cost.digest(path) == expected, f'bundle changed: {name}')
    return manifest


def make_unit(case, ids, cfg):
    """Existing branch sampling, with the user-requested empty starting prefix."""
    from experiments.branch_collection import make_case
    sampling = dict(cfg['sampling'])
    def seed(value):
        material = f"{cfg['protocol']}|{case['sha256']}|{value}"
        return int(sha256(material.encode()).hexdigest()[:16], 16)
    for key in ('prefix_seed', 'candidate_seed', 'task_order_seed'):
        sampling[key] = seed(sampling[key])
    sampling['suffix_seeds'] = [seed(value) for value in sampling['suffix_seeds']]
    unit = make_case(case, ids, sampling)
    unit['per_problem_sampling_seeds'] = {key: sampling[key] for key in
        ('prefix_seed', 'candidate_seed', 'task_order_seed', 'suffix_seeds')}
    cost.check(unit['prefix'] == [], 'must start empty')
    empty = next(g for g in unit['groups'] if not g['released'])
    unit['empty_root_state'] = {'group_key': empty['group_key'], 'selected': [],
                               'CP_status': 'NOT_APPLICABLE', 'STOP_legal': False}
    unit['groups'] = [g for g in unit['groups'] if g['released']]
    unit['tasks'] = [t for t in unit['tasks'] if t['released']]
    unit['empty_root_CP_observations'] = 0
    return unit


def receipt(root, task, row):
    """Only a validated COMPLETED measurement can be checkpointed."""
    cost.check(row['state'] == 'COMPLETED' and row['task'] == task, 'incomplete measurement')
    directory = Path(root) / 'requests' / task['id']
    cost.check(row['before'] > 0 and row['value'] <= row['before'], 'invalid outcome')
    row = dict(row, G_pct=100 * (row['before'] - row['value']) / row['before'],
        latency_scope='Warm offline CP+IPC+independent validation+best-schedule saving; excludes policy inference and checkpoint receipts',
        training_label_created=False)
    durable(directory / 'receipt.json', row)
    hashes = {p.name: cost.digest(p) for p in sorted(directory.glob('*.json'))}
    durable(directory / 'COMPLETED.json', {'task': task, 'files_sha256': hashes})
    return row


def verify_completed(directory, task, instance, anchor):
    """Resume never trusts a status flag without hashes and independent legality."""
    marker = cost.read(Path(directory) / 'COMPLETED.json')
    cost.check(marker['task'] == task, 'foreign completion marker')
    for name, digest in marker['files_sha256'].items():
        path = (Path(directory) / name).resolve()
        cost.check(path.parent == Path(directory).resolve() and cost.digest(path) == digest,
                   'completed data changed')
    row = cost.read(Path(directory) / 'receipt.json')
    cost.check(row['task'] == task and row['state'] == 'COMPLETED', 'foreign receipt')
    value = cost.verify_candidate(instance, anchor, task['released'], cost.read(Path(directory) / 'best-saved.json'))
    cost.check(value == row['value'] and row['before'] == anchor['makespan'], 'receipt objective differs')
    for path in Path(directory).glob('candidate-*.json'):
        cost.verify_candidate(instance, anchor, task['released'], cost.read(path)['schedule'])
    return row


def history(case_root, tasks, instance, anchor):
    completed, attempts = {}, {t['id']: [] for t in tasks}
    by_id = {t['id']: t for t in tasks}
    for attempt in sorted((Path(case_root) / 'attempts').glob('*')):
        for directory in sorted((attempt / 'requests').glob('*')):
            cost.check(directory.name in by_id, 'unplanned task directory')
            attempts[directory.name].append(directory)
            if (directory / 'COMPLETED.json').exists():
                cost.check(directory.name not in completed, 'duplicate successful task; review before resuming')
                row = verify_completed(directory, by_id[directory.name], instance, anchor)
                completed[directory.name] = (directory, row)
    return completed, attempts


def worker(plan_path, parent_pid):
    """No GPU import. Parent death ends this owned warm CP worker on Linux."""
    if os.name != 'nt':
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        cost.check(libc.prctl(1, signal.SIGTERM) == 0, 'cannot arm parent-death signal')
        if os.getppid() != parent_pid:
            return 3
    from experiments.conditional_selection_collect import worker as original_worker
    return original_worker(plan_path)


def reap(process):
    forced = process.poll() is None
    if forced:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    else:
        process.wait()
    return {'pid': process.pid, 'exit_code': process.returncode,
            'reaped': process.poll() is not None, 'forced': forced}


def run_case(case_root, unit, pending, cfg, budget, should_stop, progress, lock_fd=None):
    """One owned warm process; checkpoints after every solver-seed observation."""
    from fjsp import parse_instance
    from experiments import conditional_selection_collect as legacy
    case_root = Path(case_root)
    output = case_root / 'attempts' / uuid4().hex
    output.mkdir(parents=True)
    plan = {'case': unit['case'], 'tasks': pending, 'config': {
        'repair_target_seconds': cfg['sampling']['repair_target_seconds'],
        'proposed_abnormal_request_watchdog_seconds': cfg['request_watchdog_seconds']}}
    durable(output / 'plan.json', plan)
    plan['_instance'] = parse_instance(cost.read(unit['case']['input']))
    plan['_initial'] = cost.read(unit['case']['anchor'])
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update({key: '1' for key in THREADS})
    command = [sys.executable, '-B', '-m', MODULE, 'worker', '--run', str(output / 'plan.json'),
               '--parent-pid', str(os.getpid())]
    # Windows venv launchers add an intermediary PID; use the qualified base
    # executable with the venv launcher identity so the worker stays directly owned.
    if os.name == 'nt':
        env['__PYVENV_LAUNCHER__'] = sys.executable
        command[0] = sys._base_executable
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {
        'pass_fds': () if lock_fd is None else (lock_fd,)}
    mailbox, process = Queue(), None
    ended = 'case_completed'
    try:
        with (output / 'worker.stderr.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, **options)
            thread = Thread(target=legacy.reader, args=(process.stdout, mailbox), daemon=True)
            thread.start()
            ready = legacy.receive(mailbox, perf_counter() + cfg['worker_boot_seconds'])
            cost.check(ready['event'] == 'ready' and ready['pid'] == process.pid, 'worker identity mismatch')
            cost.check(ready['ortools'] == version('ortools') and ready['python'] == platform.python_version(),
                       'worker version differs')
            durable(output / 'ready.json', ready)
            for task in pending:
                reason = should_stop()
                if reason:
                    ended = reason
                    break
                if not budget.reserve(cfg['request_watchdog_seconds'] + 8, task['id']):
                    ended = 'budget_exhausted'
                    break
                row = legacy.measure(output, plan, task, process, mailbox,
                                     perf_counter() + cfg['request_watchdog_seconds'])
                row = receipt(output, task, row)
                progress(output / 'requests' / task['id'], row)
            process.stdin.write(json.dumps({'event': 'shutdown'}) + '\n')
            process.stdin.flush()
            bye = legacy.receive(mailbox, perf_counter() + 3)
            cost.check(bye['event'] == 'bye', 'worker shutdown not acknowledged')
            process.wait(timeout=3)
            cost.check(process.returncode == 0, 'worker failed on shutdown')
    except BaseException:
        durable(output / 'error.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if process is not None:
            cleanup = reap(process)
            durable(output / 'cleanup.json', cleanup)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            if 'thread' in locals():
                thread.join(timeout=1)
    return ended


def snapshot_pair(output, phase):
    from experiments import server_resource_snapshot as resources
    from experiments import server_storage_snapshot as storage
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, function in [('resources', resources.snapshot), ('storage', storage.snapshot)]:
        try:
            reports[name] = function(phase=phase)
        except Exception as error:
            reports[name] = {'status': 'UNKNOWN', 'reason': f'{type(error).__name__}: {error}'}
        durable(output / f'{name}.json', reports[name])
    return reports


def admission(reports, cfg):
    s, r = reports['storage'], reports['resources']
    personal = s.get('personal_data2', {}).get('allocated_bytes')
    free = s.get('filesystem', {}).get('free_bytes')
    if personal is None or free is None:
        return 'wait_storage_unknown'
    if personal >= 250_000_000_000 or free < cfg['free_disk_reserve_bytes']:
        return 'storage_limit'
    available = r.get('host_memory', {}).get('MemAvailable_bytes')
    if available is None or available < cfg['minimum_host_available_bytes']:
        return 'wait_host_memory'
    return None


def prepare_case(bundle, run, row, cfg):
    from baseline import solve_ect
    from fjsp import parse_instance, validate_schedule
    root = Path(run) / 'cases' / row['instance']
    root.mkdir(parents=True, exist_ok=True)
    source = (Path(bundle) / row['input']).resolve()
    instance = parse_instance(cost.pinned(source, row['sha256']))
    anchor_path = root / 'anchor.json'
    if anchor_path.exists():
        anchor = cost.read(anchor_path)
    else:
        result = solve_ect(instance, perf_counter() + 10)
        cost.check(result.schedule is not None, 'ECT construction did not finish')
        anchor = result.schedule
        validate_schedule(instance, anchor)
        durable(anchor_path, anchor)
    cost.check(validate_schedule(instance, anchor) == anchor['makespan'], 'invalid ECT')
    case = {'instance': row['instance'], 'input': str(source), 'sha256': row['sha256'],
        'anchor': str(anchor_path), 'anchor_sha256': cost.digest(anchor_path),
        'before': anchor['makespan'], 'operations': len(instance.options)}
    unit = make_unit(case, sorted(instance.options), cfg)
    if (root / 'unit.json').exists():
        cost.check(cost.read(root / 'unit.json') == unit, 'frozen acquisition unit differs')
    else:
        durable(root / 'unit.json', unit)
    return root, unit, instance, anchor


def summary(run, cfg, rows, state, reason):
    from fjsp import parse_instance
    from collections import Counter, defaultdict
    from statistics import median
    cases, total, planned, positive = [], 0, 0, 0
    cleanup_reports = [cost.read(p) for p in (Path(run) / 'cases').glob('*/attempts/*/cleanup.json')]
    times, gains, native, sizes = [], [], Counter(), defaultdict(lambda: {'completed': 0, 'positive': 0})
    for row in rows:
        root = Path(run) / 'cases' / row['instance']
        if not (root / 'unit.json').exists():
            cases.append({'instance': row['instance'], 'status': 'NOT_STARTED', 'completed': 0,
                          'planned': row.get('planned_CP_observations')})
            continue
        unit = cost.read(root / 'unit.json')
        done, attempts = history(root, unit['tasks'], parse_instance(cost.pinned(
            unit['case']['input'], unit['case']['sha256'])), cost.pinned(unit['case']['anchor'], unit['case']['anchor_sha256']))
        counts = {}
        for _, value in done.values():
            key = value['task']['group_key']
            counts[key] = counts.get(key, 0) + 1
            times.append(value['request_seconds'])
            gains.append(value['G_pct'])
            native[value['native_status']] += 1
            size = sizes[str(value['task']['k'])]
            size['completed'] += 1
            size['positive'] += value['gain'] > 0
        total += len(done)
        planned += len(unit['tasks'])
        positive += sum(value['gain'] > 0 for _, value in done.values())
        cases.append({'instance': row['instance'], 'status': 'COMPLETED' if len(done) == len(unit['tasks']) else 'PARTIAL',
            'completed': len(done), 'planned': len(unit['tasks']), 'groups': len(unit['groups']),
            'groups_with_all_three_seeds': sum(n == 3 for n in counts.values()),
            'attempts_including_incomplete': sum(map(len, attempts.values()))})
    result = {'protocol': cfg['protocol'], 'state': state, 'reason': reason,
        'completed_CP_observations': total, 'planned_CP_in_prepared_cases': planned,
        'planned_CP_total': sum(r['planned_CP_observations'] for r in rows)
            if all('planned_CP_observations' in r for r in rows) else None,
        'completed_input_problems': sum(c['status'] == 'COMPLETED' for c in cases),
        'planned_input_problems': len(rows), 'cases': cases, 'positive_gain_observations': positive,
        'budget': cost.read(Path(run) / 'budget.json') if (Path(run) / 'budget.json').exists() else None,
        'native_status_counts': dict(native), 'group_size_coverage': dict(sizes),
        'owned_workers_recorded': len(cleanup_reports),
        'recorded_owned_workers_all_reaped': all(r['reaped'] for r in cleanup_reports),
        'attempts_missing_cleanup': [str(p.relative_to(run)) for p in (Path(run) / 'cases').glob('*/attempts/*')
                                     if not (p / 'cleanup.json').exists()],
        'sessions_missing_final_snapshot': [str(p.relative_to(run)) for p in (Path(run) / 'sessions').glob('*')
            if not (p / 'after' / 'storage.json').exists() or not (p / 'after' / 'resources.json').exists()],
        'CP_request_seconds': {'count': len(times), 'mean': sum(times) / len(times) if times else None,
            'median': median(times) if times else None, 'max': max(times) if times else None},
        'G_pct': {'count': len(gains), 'mean': sum(gains) / len(gains) if gains else None,
            'max': max(gains) if gains else None},
        'GPU_computation': False, 'training': False, 'validation_or_test_opened': False,
        'interpretation': 'CP outcome/coverage evidence only; no preference-label or learning-quality acceptance.'}
    durable(Path(run) / 'summary.json', result)
    return result


def paths(bundle, run):
    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    cost.check(platform.system() == 'Linux' and bundle.is_relative_to(ROOT / 'code')
               and run.is_relative_to(ROOT / 'runs'), 'use the personal Linux server directories')
    cost.check(Path(sys.prefix).resolve() == ROOT / 'envs/fjsp-train', 'wrong server interpreter')
    return bundle, run


def collect(bundle, run):
    bundle, run = paths(bundle, run)
    verify_bundle(bundle)
    cfg = cost.read(bundle / 'acquisition-plan.json')
    actual = {k: platform.python_version() if k == 'python' else version(k)
              for k in cfg['expected_server_versions']}
    cost.check(actual == cfg['expected_server_versions'], f'environment differs: {actual}')
    cost.check(cost.read(run / 'identity.json')['manifest_sha256'] == cost.digest(bundle / 'manifest.json'),
               'resume bundle differs')
    from experiments.resched_qualification import configure_cpu
    configure_cpu()
    os.nice(5)
    rows = cost.read(bundle / 'inputs.json')
    session = run / 'sessions' / uuid4().hex
    session.mkdir(parents=True)
    budget, reason, state = None, 'not_started', 'ERROR'
    with exclusive(ROOT / 'locks' / 'random-acquisition.lock') as lock:
        try:
            reports = snapshot_pair(session / 'before', 'before')
            budget = Budget(run / 'budget.json', cfg['batch_wall_seconds'])
            durable(session / 'ready.json', {'pid': os.getpid(), 'session': str(session),
                'time_utc': datetime.now(timezone.utc).isoformat(), 'zero_GPU_contexts': True})
            def stopped():
                if (run / 'STOP').exists():
                    return 'stop_requested'
                if budget.left < cfg['request_watchdog_seconds'] + 8:
                    return 'budget_exhausted'
                return None
            next_resource, n = perf_counter() + cfg['resource_interval_seconds'], 0
            def check_resources():
                nonlocal next_resource, n, reports
                if perf_counter() >= next_resource:
                    if not budget.reserve(90, 'resource_snapshot'):
                        return 'budget_exhausted'
                    reports = snapshot_pair(session / f'periodic-{n:04d}', 'manual')
                    n += 1
                    next_resource = perf_counter() + cfg['resource_interval_seconds']
                return admission(reports, cfg)
            def progress(directory, value):
                durable(run / 'status.json', {'state': 'RUNNING', 'session': str(session),
                    'last_completed': str(directory), 'G_pct': value['G_pct'],
                    'active_wall_seconds': budget.spent, 'time_utc': datetime.now(timezone.utc).isoformat()})
            for row in rows:
                while True:
                    reason = stopped() or check_resources()
                    if reason is None or not reason.startswith('wait_'):
                        break
                    durable(run / 'status.json', {'state': 'WAITING', 'reason': reason,
                                                  'active_wall_seconds': budget.spent})
                    if not budget.reserve(cfg['resource_wait_seconds'] + 90, 'resource_wait'):
                        reason = 'budget_exhausted'
                        break
                    sleep(cfg['resource_wait_seconds'])
                    next_resource = 0
                if reason:
                    break
                if not budget.reserve(30, 'prepare_and_audit_case'):
                    reason = 'budget_exhausted'
                    break
                root, unit, instance, anchor = prepare_case(bundle, run, row, cfg)
                next_size_check = 0
                def checks():
                    nonlocal next_size_check, next_resource
                    if perf_counter() >= next_size_check:
                        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) >= cfg['output_limit_bytes'] - 50_000_000:
                            return 'output_limit'
                        next_size_check = perf_counter() + 30
                    reason = stopped() or check_resources()
                    if reason is None and os.name != 'nt':
                        # Cheap live memory check between requests; full paired
                        # telemetry is kept at its slower interval.
                        from experiments.server_gpu_probe import memory_snapshot
                        try:
                            available = memory_snapshot().get('MemAvailable_bytes')
                        except (OSError, ValueError):
                            available = None
                        if available is None or available < cfg['minimum_host_available_bytes']:
                            next_resource = 0
                            reason = 'wait_host_memory'
                    return reason
                while True:
                    reason = stopped() or check_resources()
                    if reason and reason.startswith('wait_'):
                        durable(run / 'status.json', {'state': 'WAITING', 'reason': reason,
                                                      'active_wall_seconds': budget.spent})
                        if not budget.reserve(cfg['resource_wait_seconds'] + 90, 'resource_wait'):
                            reason = 'budget_exhausted'
                            break
                        sleep(cfg['resource_wait_seconds'])
                        next_resource = 0
                        continue
                    if reason:
                        break
                    done, attempts = history(root, unit['tasks'], instance, anchor)
                    pending = [t for t in unit['tasks'] if t['id'] not in done
                               and len(attempts[t['id']]) < cfg['max_attempts_per_task']]
                    if any(len(attempts[t['id']]) >= cfg['max_attempts_per_task']
                           for t in unit['tasks'] if t['id'] not in done):
                        reason = 'attempt_limit_requires_review'
                        break
                    if not pending:
                        reason = 'case_completed'
                        break
                    if not budget.reserve(cfg['worker_boot_seconds'] + 8, 'worker_boot'):
                        reason = 'budget_exhausted'
                        break
                    reason = run_case(root, unit, pending, cfg, budget, checks, progress, lock.fileno())
                    if not reason.startswith('wait_'):
                        break
                    # Release the CP process before waiting; never hold a GPU context.
                    durable(run / 'status.json', {'state': 'WAITING', 'reason': reason,
                                                  'active_wall_seconds': budget.spent})
                    if not budget.reserve(cfg['resource_wait_seconds'] + 90, 'resource_wait'):
                        reason = 'budget_exhausted'
                        break
                    sleep(cfg['resource_wait_seconds'])
                    next_resource = 0
                if reason != 'case_completed':
                    break
            else:
                reason = 'all_tasks_completed'
            state = 'COMPLETED' if reason == 'all_tasks_completed' else 'PAUSED'
        except KeyboardInterrupt:
            state, reason = 'PAUSED', 'signal_interrupted'
        except Exception:
            durable(session / 'error.json', {'traceback': traceback.format_exc()})
            state, reason = 'ERROR', 'execution_error_no_automatic_retry'
        finally:
            if budget is not None:
                budget.settle()
            snapshot_pair(session / 'after', 'after')
            try:
                result = summary(run, cfg, rows, state, reason)
                print(json.dumps({k: result[k] for k in ('state', 'reason', 'completed_CP_observations')}), flush=True)
            except Exception:
                state, reason = 'ERROR', 'final_integrity_review_failed'
                durable(session / 'review-error.json', {'traceback': traceback.format_exc()})
                raise
            finally:
                durable(run / 'status.json', {'state': state, 'reason': reason, 'session': str(session),
                    'finished_utc': datetime.now(timezone.utc).isoformat()})
    return 0 if state != 'ERROR' else 2


def launch(bundle, run, *, resume=False):
    bundle, run = paths(bundle, run)
    verify_bundle(bundle)
    with exclusive(ROOT / 'locks' / 'random-acquisition.lock'):
        if resume:
            cost.check(run.is_dir() and cost.read(run / 'identity.json')['manifest_sha256'] == cost.digest(bundle / 'manifest.json'),
                       'resume identity differs')
            if (run / 'STOP').exists():
                (run / 'STOP').rename(run / f'STOP-archived-{uuid4().hex}')
        else:
            run.mkdir(parents=True, exist_ok=False)
            durable(run / 'identity.json', {'bundle': str(bundle),
                'manifest_sha256': cost.digest(bundle / 'manifest.json'), 'created_utc': datetime.now(timezone.utc).isoformat()})
    log_path = run / f'launcher-{uuid4().hex}.log'
    with log_path.open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-m', MODULE, 'run', '--run', str(run)],
            cwd=bundle, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'记录目录：{run}\n后台进程：{process.pid}\n日志：{log_path}', flush=True)
    end = perf_counter() + 15
    while perf_counter() < end:
        if process.poll() is not None:
            cost.check(process.returncode == 0, f'background startup failed; inspect {log_path}')
            print('后台已结束；查看status.json和summary.json确认完成或暂停原因。', flush=True)
            return 0
        if any(cost.read(p).get('pid') == process.pid for p in (run / 'sessions').glob('*/ready.json')):
            print('FJSP_ACQUISITION_DETACHED_READY', flush=True)
            return 0
        sleep(.25)
    print('启动检查尚未完成；查看日志及status.json确认，不要重复启动。', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'resume', 'run', 'status', 'stop', 'worker'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args(argv)
    if args.action == 'worker':
        return worker(args.run, args.parent_pid)
    if args.action in ('start', 'resume'):
        return launch(Path.cwd(), args.run, resume=args.action == 'resume')
    if args.action == 'status':
        for name in ('status.json', 'budget.json', 'summary.json'):
            path = args.run / name
            if path.exists():
                value = cost.read(path)
                if name == 'summary.json':
                    value = {key: val for key, val in value.items() if key != 'cases'}
                print(name + '\n' + json.dumps(value, indent=2, ensure_ascii=False), flush=True)
        return 0
    if args.action == 'stop':
        _, run = paths(Path.cwd(), args.run)
        durable(run / 'STOP', {'time_utc': datetime.now(timezone.utc).isoformat()})
        print('已请求在当前CP观测结束后停止；已完成数据会保留。', flush=True)
        return 0
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return collect(Path.cwd(), args.run)


if __name__ == '__main__':
    raise SystemExit(main())
