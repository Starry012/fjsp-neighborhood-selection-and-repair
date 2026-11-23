"""Shared CPU checks and saved block verification for the benchmark."""

from pathlib import Path
from experiments import conditional_selection_cost_pilot as cost
from experiments import stop8_shared_cpu as cpu


def check_stop(root):
    if (root / 'STOP').exists():
        raise InterruptedError('stop requested')


def cgroup_cpu():
    try:
        group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        base = Path('/sys/fs/cgroup')
        location = (base/group.lstrip('/')).resolve()
        if not location.is_relative_to(base):
            return None
        stats = dict(line.split() for line in (location/'cpu.stat').read_text().splitlines())
        return {'cpu_max': (location/'cpu.max').read_text().strip(),
                'nr_throttled': int(stats['nr_throttled']), 'throttled_usec': int(stats['throttled_usec'])}
    except (OSError, StopIteration, ValueError, KeyError):
        return None


def previous(block_path, block):
    results = []
    for attempt in sorted(block_path.glob('attempt-*')):
        trail = [p for p in sorted(attempt.glob('trial-*/record.json'))]
        proof = attempt/'finished.json'
        if proof.exists():
            entry = cost.read(proof)
            cost.check(entry['block'] == block['id'], 'record belongs to a different block')
            cost.check(entry['sha256'] == {p.relative_to(attempt).as_posix(): cost.digest(p) for p in trail},
                       'saved trial evidence changed')
        else:
            entry = {'status': 'INCOMPLETE', 'block': block['id']}
        rows = [cost.read(p) for p in trail]
        cost.check([r['method'] for r in rows] == block['arms'][:len(rows)] and
                   all(r['instance'] == block['case']['instance'] and r['seed'] == block['seed'] for r in rows),
                   'saved order or instance/seed differs')
        if entry['status'] == 'QUALIFIED':
            cost.check(len(rows) == len(block['arms']) and len({r['input_sha256'] for r in rows}) == 1 and
                       len({r['cpu']['selected_cpu'] for r in rows}) == 1 and all(r['engineering_passed'] and
                       r['cpu']['coverage_passed'] and not r['cpu']['contention_reasons'] for r in rows), 'invalid accepted block')
        results.append({**entry, 'attempt': attempt.name, 'rows': rows})
    return results
