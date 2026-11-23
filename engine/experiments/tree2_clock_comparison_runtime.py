"""Compare the two Tree2 weights and the cheap selector with identical Fixed3 CP."""

import argparse
from pathlib import Path
from time import perf_counter, thread_time
from unittest.mock import patch

from experiments import tree8_confirmation_runtime as tree
from experiments.stop8_shared_runtime import runqueue_ns


CLOCKLESS = 'Tree2WithoutClock'


class Controller(tree.Controller):
    def __init__(self, cfg, method, clock=None):
        super().__init__(cfg, method, clock)
        if method == CLOCKLESS:
            self.method = 'Tree2'


def execute(request, publisher, *, controller_factory=None):
    from experiments import stop8_loop as loop
    from experiments.cp_rebuild_observer import observed_repair
    from ortools.sat.python import cp_model

    def repair(instance, anchor, released, deadline, *, seed, stream, publish, call_context):
        native = cp_model.CpSolver.solve
        timing = {}

        def measured(solver, model, solution_callback=None):
            queue = runqueue_ns()
            wall, cpu = perf_counter(), thread_time()
            try:
                return native(solver, model, solution_callback)
            finally:
                wall_ms = (perf_counter() - wall) * 1000
                thread_cpu_ms = (thread_time() - cpu) * 1000
                end = runqueue_ns()
                timing.update(wall_ms=wall_ms, thread_cpu_ms=thread_cpu_ms, native_started=wall,
                              runqueue_ms=(end - queue) / 1e6 if queue is not None and end is not None and end >= queue else None)

        with patch.object(cp_model.CpSolver, 'solve', measured):
            report = observed_repair(instance, anchor, released, deadline, seed=seed, variant='C',
                                     publish=publish, request_started=perf_counter())
        # The shared-core runner consumes the native timing under this field.
        report['stop_monitor'] = {'shared_cpu_timing': timing or None}
        return report

    def make(cfg, method):
        return Controller(cfg, method, clock=lambda: perf_counter() - request['started'])

    return loop.measured_loop(request, publisher, repair_observer=repair, controller_factory=make)


def run_trial(bundle, output, case, method, seed, cfg, **kwargs):
    from experiments import stop8_loop as loop
    return loop.run_trial(bundle, output, case, method, seed, cfg,
                          worker_module='experiments.tree2_clock_comparison_runtime', **kwargs)


def audit_trial(output, case, method, seed, cfg):
    with patch.object(tree, 'Controller', Controller):
        row = tree.audit_trial(output, case, method, seed, cfg)
    row.update(checked=0, stops=0, checkpoints={str(t): min(
        (r['Cmax'] for r in row['anytime_trace'] if r['ms'] <= t * 1000), default=None)
        for t in (10, 30, 50, 60)})
    return row


if __name__ == '__main__':
    from experiments import stop8_loop as loop
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    with patch.object(loop, 'execute_worker', execute):
        raise SystemExit(loop.worker(parser.parse_args().worker))
