"""Run and audit Stall80 and Fixed3 CP on one absolute trial clock."""

import json
from time import get_clock_info, monotonic_ns

from alns_neighborhoods import anchor_hash
from experiments import conditional_selection_cost_pilot as cost
from experiments import glnsa_runtime as glnsa
from experiments import tree2_clock_comparison_runtime as cp
from experiments.offline_job_state import durable


STALL = 'Stall80'
SELECTORS = ('MinOptions2', cp.CLOCKLESS)
SWITCHES = (5, 10, 20, 35)
HYBRIDS = {f'Stall{seconds:02d}-{selector}': (seconds * 1000, selector)
           for seconds in SWITCHES for selector in SELECTORS}
ARMS = (STALL, *SELECTORS, *HYBRIDS)
CHECKPOINTS = (5, 10, 20, 30, 35, 50, 60)


def verify_clocks():
    cost.check(get_clock_info('monotonic').implementation == get_clock_info('perf_counter').implementation,
               'the native and guarded trials require the same monotonic clock')


def run_trial(bundle, trial, case, method, seed, cfg, binary, core, fds):
    from experiments import stop8_loop as loop

    started_ns = monotonic_ns()
    switch_ms, selector = HYBRIDS[method] if method in HYBRIDS else (None, method)
    prefix = None
    if method == STALL or switch_ms is not None:
        prefix = glnsa.run_trial(binary, case, bundle/case['input'], trial/'stall', seed,
                                 total_ms=cfg['total_ms'], pass_fds=fds, policy=STALL,
                                 common_start_ns=started_ns, handoff_ms=switch_ms, expected_cpu=core)
        durable(trial/'stall-result.json', prefix)
    cp_result = None
    if method != STALL:
        incumbent = cost.pinned(trial/'stall'/'delivery.json', prefix['delivery_sha256']) if prefix else None
        cp_result = loop.run_trial(bundle, trial/'cp', case, selector, seed, cfg,
                                   worker_module='experiments.tree2_clock_comparison_runtime', worker_pass_fds=fds,
                                   common_started=started_ns/1e9, handoff_schedule=incumbent)
    wall_ms = (monotonic_ns()-started_ns)/1e6
    return {'started_ns': started_ns, 'wall_ms': wall_ms, 'prefix': prefix, 'cp': cp_result}


def audit_trial(bundle, trial, case, method, seed, cfg, run, core):
    switch_ms, selector = HYBRIDS[method] if method in HYBRIDS else (None, method)
    prefix, guarded = run['prefix'], run['cp']
    trace, cp_calls, queue = [], [], []
    row = {'input_sha256': case['sha256'], 'switch_ms': switch_ms, 'wall_ms': run['wall_ms'],
           'engineering_passed': run['wall_ms'] <= cfg['total_ms'], 'initial_sha256': None,
           'handoff_sha256': None, 'handoff_Cmax': None, 'handoff_written_ms': None, 'prefix_return_ms': None,
           'first_cp_request_ms': None, 'first_native_cp_ms': None, 'repair_calls': 0, 'selection_seconds': 0.}
    if prefix:
        glnsa.audit(case, bundle/case['input'], trial/'stall', prefix)
        cost.check(prefix['started_monotonic_ns'] == run['started_ns'] and
                   prefix['caller_observed_ms'] <= run['wall_ms'], 'native trial clock differs')
        cost.check(prefix['kernel_affinity'] == [core] if core is not None else True, 'native affinity differs')
        trace = [{**entry, 'phase':'Stall80'} for entry in prefix['anytime_trace']]
        row['prefix_return_ms'] = prefix['caller_observed_ms']
        row['native_generations'] = prefix['generations']

    if guarded:
        cost.check(guarded['started_perf_counter'] == run['started_ns']/1e9 and
                   guarded['caller_observed_ms'] <= run['wall_ms'], 'CP reset the outer clock')
        audited = cp.audit_trial(trial/'cp', case, selector, seed, cfg)
        row['engineering_passed'] &= audited['engineering_passed']
        row.update(initial_sha256=audited['initial_sha256'], repair_calls=audited['repair_calls'],
                   selection_seconds=audited['selection_seconds'])
        cp_calls = audited['calls']
        ready = trial/'cp'/'worker_ready.json'
        if core is not None and ready.is_file():
            cost.check(cost.read(ready)['kernel_affinity'] == [core], 'CP affinity differs')
        cp_trace = audited['anytime_trace']
        if prefix:
            handoff = cost.pinned(trial/'stall'/'delivery.json', prefix['delivery_sha256'])
            handoff_digest = anchor_hash(handoff)
            cost.check(prefix['handoff_ms'] == switch_ms and trace[-1]['ms'] <= switch_ms and
                       prefix['credited_Cmax'] == handoff['makespan'] == cp_trace[0]['Cmax'] and
                       audited['initial_source'] == STALL and audited['initial_sha256'] == handoff_digest and
                       cp_trace[0]['ms'] >= max(switch_ms, prefix['caller_observed_ms']),
                       'CP handoff does not match the last timely Stall80 delivery')
            row.update(handoff_sha256=handoff_digest, handoff_Cmax=handoff['makespan'], handoff_written_ms=trace[-1]['ms'])
            cp_trace = cp_trace[1:]
        else:
            cost.check(audited['initial_source'] == 'ECT', 'cold CP must start from ECT')
        trace.extend({**r, 'phase':'CP'} for r in cp_trace)
        starts = [e for line in (trial/'cp'/'calls.jsonl').read_text().splitlines()
                  if (e:=json.loads(line))['event']=='CALL_STARTED'] if cp_calls else []
        if starts:
            row['first_cp_request_ms'] = starts[0]['started_ms']
            if prefix:
                cost.check(starts[0]['started_ms'] >= guarded['worker_started_ms'] >= switch_ms,
                           'CP started before the handoff')
        for call in cp_calls:
            path = trial/'cp'/'calls'/f"call-{call['call']:06d}"/'report.json'
            if path.is_file():
                native = cost.read(path)['stop_monitor']['shared_cpu_timing']
                if native and native['runqueue_ms'] is not None:
                    queue.append(native['runqueue_ms'])
                if native and row['first_native_cp_ms'] is None:
                    row['first_native_cp_ms'] = (native['native_started']-run['started_ns']/1e9)*1000
        if switch_ms is not None and row['first_native_cp_ms'] is not None:
            cost.check(row['first_native_cp_ms'] >= row['first_cp_request_ms'] >= switch_ms, 'native CP clock predates handoff')
        if switch_ms is not None:
            row['engineering_passed'] &= row['first_native_cp_ms'] is not None

    cost.check(trace and all(a['ms'] <= b['ms'] and a['Cmax'] > b['Cmax'] for a,b in zip(trace,trace[1:])) and
               trace[-1]['ms'] <= run['wall_ms'], 'merged delivery trace is invalid')
    row['credited_Cmax'] = trace[-1]['Cmax'] if row['engineering_passed'] else None
    row['checkpoints'] = {str(t): min((r['Cmax'] for r in trace if r['ms'] <= t*1000), default=None) for t in CHECKPOINTS}
    row['post_handoff_gain'] = row['handoff_Cmax']-trace[-1]['Cmax'] if switch_ms is not None else None
    if row['engineering_passed']:
        cost.check(row['checkpoints']['60'] == row['credited_Cmax'],
                   '60 second checkpoint differs from the credited delivery')
    return row, trace, max(queue, default=None)
