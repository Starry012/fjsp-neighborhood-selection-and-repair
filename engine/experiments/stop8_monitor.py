"""One timed decision from verified parent deliveries and same-neighborhood CP bounds."""
from hashlib import sha256
import json
import math
from threading import Event, Thread
from time import perf_counter

ARMS = ("Fixed3", "Fixed2", "Late2", "RandomStop", "QuietStop", "GainStop", "GapStop", "RelativeStop")
CAL = {"Cal05": 5, "Cal10": 10, "Cal20": 20}
FIELDS = ("check_state", "threshold_pct", "checkpoint_ms", "decision_ms", "native_enter_ms", "native_exit_ms",
          "requested_stop", "stop_latency_ms", "draw", "published_gain", "upper0", "upper1", "lower0", "lower1",
          "gap0", "gap1", "shrink_pct", "static_lower", "bound_events", "receipt_count")


def append_receipt(root, at, source, value):
    # One append after the parent has validated and replaced the public delivery.
    with (root/"stop-receipts.jsonl").open("ab", buffering=0) as out:
        out.write((json.dumps([at, source, value], separators=(",", ":"))+"\n").encode("ascii"))


def read_receipts(path):
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    return [json.loads(line) for line in data[:end].splitlines()]


def snapshot(receipts, bounds, call, anchor, lower, t, lookback):
    prefix = f"alns:{call}:"
    def at(point):
        known = [v for ms,src,v in receipts if ms<=point]
        if not known or min(known)>anchor:
            return None
        upper = min([anchor]+[v for ms,src,v in receipts if ms<=point and src.startswith(prefix)])
        lb = max([lower]+[v for ms,v in bounds if ms<=point])
        if not 0 <= lb <= upper:
            raise ValueError("local lower bound contradicts a feasible schedule")
        return upper, lb
    first, last = at(t-lookback), at(t)
    if first is None or last is None:
        return None
    u0,l0 = first
    u1,l1 = last
    g0,g1 = u0-l0,u1-l1
    denominator = g0*u1
    numerator = denominator-g1*u0
    return dict(upper0=u0,upper1=u1,lower0=l0,lower1=l1,gap0=g0,gap1=g1,
                published_gain=u0-u1,shrink_pct=100*numerator/denominator if denominator else None)


def stop_decision(method, s, threshold, draw):
    if method in ("Fixed2","Fixed3"):
        return False
    if method=="Late2":
        return True
    if method=="RandomStop":
        return draw < 146
    if method=="QuietStop":
        return s["published_gain"]==0
    if method=="GainStop":
        return s["published_gain"]<=1
    if method=="GapStop":
        return s["gap1"]<=2
    if method not in ("RelativeStop", *CAL) or threshold not in CAL.values():
        raise ValueError("unregistered stopping rule")
    if s["gap1"]<=2:
        return True
    denominator = s["gap0"]*s["upper1"]
    numerator = denominator-s["gap1"]*s["upper0"]
    return 100*numerator <= threshold*denominator


def independent_draw(case, seed, call):
    from random import Random
    key = sha256(f"stop8-v1/{case}/{seed}/{call}".encode()).digest()
    return Random(key).randrange(336)


class Monitor:
    def __init__(self, request, call, method, threshold):
        from pathlib import Path
        selection = call["selection"]
        self.lower = selection["pool"][selection["pool_index"]]["features"]["relaxed_lower_bound"]
        self.origin = request["started"]
        self.root = Path(request["output"])
        self.method, self.threshold = method,threshold
        self.anchor,self.number = call["anchor_Cmax"],call["call"]
        self.draw = independent_draw(request["case"]["instance"],request["spec"]["seed"],self.number)
        cfg = request["spec"]["config"]
        self.t = call["started_ms"]+cfg["decision_offset_ms"]
        self.lookback = cfg["lookback_ms"]
        self.bounds = []
        self.eligible = (self.t>=cfg["late_phase_ms"] and
                         call["local_target_ms"]-call["started_ms"]+0.001>=cfg["variants"][method]["fixed_seconds"]*1000)
        self.log = dict(check_state="no_native", requested_stop=False,threshold_pct=threshold,draw=self.draw,
                        checkpoint_ms=self.t,static_lower=self.lower,receipt_count=0)

    def clock(self):
        return (perf_counter()-self.origin)*1000

    def record_bound(self, value):
        if not math.isfinite(value):
            raise ValueError("nonfinite CP lower bound")
        valid = math.floor(value)
        if not self.bounds or valid>self.bounds[-1][1]:
            self.bounds.append([self.clock(),valid])

    def solve(self, native, solver, model, callback, deadline):
        solver.best_bound_callback = self.record_bound
        done = Event()
        errors = []
        def check():
            try:
                if done.wait(max(0,(self.t-self.clock())/1000)):
                    return
                receipts = read_receipts(self.root/"stop-receipts.jsonl")
                info = snapshot(receipts, self.bounds.copy(), self.number,self.anchor,self.lower,self.t,self.lookback)
                self.log["receipt_count"] = len(receipts)
                if done.is_set():
                    self.log["check_state"]="native_finished_during_read"
                    return
                if info is None:
                    self.log["check_state"]="anchor_delivery_pending"
                    return
                self.log.update(info,check_state="checked",decision_ms=self.clock())
                self.log["requested_stop"] = stop_decision(self.method,info,self.threshold,self.draw)
                if self.log["requested_stop"]:
                    solver.stop_search()
            except Exception as error:
                errors.append(error)
        enter = self.clock()
        self.log["native_enter_ms"] = enter
        thread = None
        if self.eligible and enter<self.t:
            self.log["check_state"]="native_finished_before_timer"
            thread = Thread(target=check, name="stop8-clock")
            thread.start()
        else:
            self.log["check_state"]="outside_gate" if not self.eligible else "native_started_after_checkpoint"
        try:
            # The observer, timer setup and bound registration share this request's cap.
            solver.parameters.max_time_in_seconds = max(1e-9, deadline-perf_counter())
            return native(solver,model,callback)
        finally:
            self.log["native_exit_ms"]=self.clock()
            done.set()
            if thread is not None:
                thread.join()
            if self.log["requested_stop"]:
                self.log["stop_latency_ms"]=self.log["native_exit_ms"]-self.log["decision_ms"]
            if errors:
                raise RuntimeError("checkpoint observation failed") from errors[0]
