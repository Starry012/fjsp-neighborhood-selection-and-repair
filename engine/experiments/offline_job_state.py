"""Small atomic heartbeat/progress files; terminal state is published after cleanup."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event, Lock, Thread
from time import time, perf_counter, sleep
from uuid import uuid4


def atomic_publish(source, target, *, retry_windows=None):
    """Windows readers can briefly deny replacement; retain one complete temp."""
    retry = os.name == 'nt' if retry_windows is None else retry_windows
    deadline = perf_counter() + 1
    while True:
        try:
            os.replace(source, target)
            break
        except PermissionError:
            if not retry or perf_counter() >= deadline:
                raise
            sleep(.01)
    if os.name != 'nt':
        fd = os.open(Path(target).parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    atomic_publish(temporary, path)


def process_identity(pid=None, proc=Path('/proc')):
    pid = os.getpid() if pid is None else pid
    try:
        raw = (proc / str(pid) / 'stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
                'boot_id': (proc / 'sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return {'pid': pid, 'start_ticks': None, 'boot_id': None, 'state': None}


def effective_status(saved, *, now=None, identity_reader=process_identity):
    result = dict(saved)
    if result.get('state') in ('COMPLETED', 'FAILED', 'PAUSED'):
        return result
    owner = result.get('owner') or {}
    observed = identity_reader(owner.get('pid', -1))
    confirmed = (owner.get('start_ticks') is not None and owner.get('boot_id') is not None
                 and all(observed.get(k) == owner.get(k) for k in ('pid', 'start_ticks', 'boot_id'))
                 and observed.get('state') not in (None, 'Z', 'X'))
    if not confirmed:
        result['effective_state'] = 'INTERRUPTED_OR_UNCONFIRMED'
    elif (time() if now is None else now) - result.get('heartbeat_epoch', 0) > 90:
        result['effective_state'] = 'STALE_HEARTBEAT_CHECK_REQUIRED'
    else:
        result['effective_state'] = result.get('state')
    return result


class JobState:
    def __init__(self, root, *, total, interval=30):
        self.root = Path(root)
        self.lock, self.event = Lock(), Event()
        self.data = {'state': 'STARTING', 'phase': 'startup', 'completed': 0, 'total': total,
                     'owner': process_identity(), 'exit_code': None}
        self.interval = interval
        self.last_logged_event = None
        self.thread = Thread(target=self._heartbeat, daemon=True)
        self.update()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.data.update(heartbeat_epoch=time(), heartbeat_utc=datetime.now(timezone.utc).isoformat())
            durable(self.root / 'status.json', self.data)
            event = (self.data.get('state'), self.data.get('phase'), self.data.get('completed'), self.data.get('current_case'))
            if event != self.last_logged_event:
                print(f"[{self.data['state']}] {self.data['completed']}/{self.data['total']} "
                      f"{self.data['phase']} {self.data.get('current_case', '')}", flush=True)
                self.last_logged_event = event

    def _heartbeat(self):
        while not self.event.wait(self.interval):
            self.update()

    def finish(self, state, exit_code, **values):
        self.event.set()
        self.thread.join(timeout=5)
        self.update(state=state, exit_code=exit_code, **values)
        durable(self.root / ('SUCCESS.json' if state == 'COMPLETED' else 'PAUSED.json' if state == 'PAUSED' else 'FAILED.json'),
                dict(self.data))
