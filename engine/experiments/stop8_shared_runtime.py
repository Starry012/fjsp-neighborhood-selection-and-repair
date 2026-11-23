"""Instrument the frozen Stop8 execution and the original independent auditor."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter, thread_time
from unittest.mock import patch

from experiments import stop8_monitor as sm
from experiments import stop8_runtime as frozen
from experiments.stop8_shared_cpu import optional_text


def runqueue_ns():
    raw = optional_text('/proc/thread-self/schedstat')
    try:
        return int(raw.split()[1]) if raw is not None else None
    except (IndexError, ValueError):
        return None


class TimedMonitor(sm.Monitor):
    def solve(self, native, solver, model, callback, deadline):
        def timed_native(*args):
            q0 = runqueue_ns()
            wall0, cpu0 = perf_counter(), thread_time()
            try:
                return native(*args)
            finally:
                cpu, wall = thread_time() - cpu0, perf_counter() - wall0
                q1 = runqueue_ns()
                self.log['shared_cpu_timing'] = {'wall_ms': wall * 1000,
                    'thread_cpu_ms': cpu * 1000,
                    'runqueue_ms': (q1-q0)/1e6 if q0 is not None and q1 is not None and q1 >= q0 else None}
        return super().solve(timed_native, solver, model, callback, deadline)


MATCHED = 'MatchedRandomStop'


def matched_draw(case, seed, call):
    key = sha256(f'stop8-shared-matched/v1/{case}/{seed}/{call}'.encode()).digest()
    return Random(key).randrange(237)


@contextmanager
def policy(method):
    if method != MATCHED:
        yield
        return
    original = sm.stop_decision

    def decide(arm, signals, threshold, draw):
        if arm == MATCHED:
            return draw < 211
        return original(arm, signals, threshold, draw)

    with patch.object(sm, 'stop_decision', decide), patch.object(sm, 'independent_draw', matched_draw):
        yield


def execute(request, publisher, *, controller_factory=None):
    with patch.object(sm, 'Monitor', TimedMonitor), policy(request['spec']['method']):
        return frozen.execute(request, publisher, controller_factory=controller_factory)


def run_trial(bundle, output, case, method, seed, cfg, **kwargs):
    from experiments import stop8_loop as loop
    return loop.run_trial(bundle, output, case, method, seed, cfg,
                          worker_module='experiments.stop8_shared_runtime', **kwargs)


def audit_trial(output, case, method, seed, cfg):
    with policy(method):
        return frozen.audit_trial(output, case, method, seed, cfg)


if __name__ == '__main__':
    from experiments import stop8_loop as loop
    cli = argparse.ArgumentParser()
    cli.add_argument('--worker', type=Path, required=True)
    with patch.object(loop, 'execute_worker', execute):
        raise SystemExit(loop.worker(cli.parse_args().worker))
