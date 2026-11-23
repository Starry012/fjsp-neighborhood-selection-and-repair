"""Run and independently audit the frozen Tree2 selector with timed CP stops."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from experiments import tree8_confirmation_runtime as tree
from experiments import conditional_selection_cost_pilot as cost
from experiments import stop8_monitor as sm


class Controller(tree.Controller):
    def __init__(self,cfg,method,clock=None):
        super().__init__(cfg,method,clock)
        self.method = "Tree2"
        self.target = self.theta
        # The Tree2 scoring feature is fixed for every arm; only the actual CP budget varies.
        self.theta = 3

    @property
    def target_seconds(self):
        return self.target


def execute(request,publisher,*,controller_factory=None):
    from experiments import stop8_loop as loop
    from experiments.cp_rebuild_observer import observed_repair
    from ortools.sat.python import cp_model
    cfg = request["spec"]["config"]
    method = request["spec"]["method"]
    def repair(instance,anchor,released,deadline,*,seed,stream,publish,call_context):
        monitor=sm.Monitor(request,call_context,method,cfg.get("threshold_pct"))
        native = cp_model.CpSolver.solve
        def measured(solver,model,solution_callback=None):
            return monitor.solve(native,solver,model,solution_callback,deadline)
        def publish_with_bound(schedule,kind,details):
            bound=details.get("lower_bound")
            if bound is not None:
                monitor.record_bound(bound)
            return publish(schedule,kind,details)
        with patch.object(cp_model.CpSolver,"solve",measured):
            report=observed_repair(instance,anchor,released,deadline,seed=seed,variant="C",
                                   publish=publish_with_bound,request_started=perf_counter())
        report["stop_monitor"]={**monitor.log,"bound_events":len(monitor.bounds),"bounds":monitor.bounds}
        return report
    make = lambda cfg,method: Controller(cfg,method,clock=lambda:perf_counter()-request["started"])
    return loop.measured_loop(request,publisher,repair_observer=repair,controller_factory=make)


def run_trial(bundle,output,case,method,seed,cfg,**kwargs):
    from experiments import stop8_loop as loop
    return loop.run_trial(bundle,output,case,method,seed,cfg,worker_module="experiments.stop8_runtime",**kwargs)


def audit_trial(output,case,method,seed,cfg):
    root=Path(output)
    with patch.object(tree,"Controller",Controller):
        row=tree.audit_trial(root,case,method,seed,cfg)
    result=cost.read(root/"result.json")
    ledger=sm.read_receipts(root/"stop-receipts.jsonl")
    cost.check(ledger==[[e["delivery_written_ms"],e["source"],e["makespan"]] for e in result["receipts"] if e["eligible"]],
               "parent delivery ledger differs from verified receipts")
    starts={e["call"]:e for line in (root/"calls.jsonl").read_text().splitlines()
            if (e:=json.loads(line))["event"]=="CALL_STARTED"}
    for c in row["calls"]:
        start=starts[c["call"]]
        path=root/"calls"/f"call-{c['call']:06d}"/"report.json"
        if not path.exists():
            cost.check(not row["engineering_passed"],"missing successful repair")
            continue
        info=cost.read(path)["stop_monitor"]
        bounds=info["bounds"]
        cost.check(info["static_lower"]==c["selected_lower_bound"] and
                   all(v<=c["after_Cmax"] for at,v in bounds) and
                   all(a[0]<=b[0] and a[1]<b[1] for a,b in zip(bounds,bounds[1:])),
                   "CP bounds invalid against the audited feasible result")
        check_at=start["started_ms"]+cfg["decision_offset_ms"]
        cost.check(info["draw"]==sm.independent_draw(case["instance"],seed,c["call"]) and
                   info["checkpoint_ms"]==check_at and info["threshold_pct"]==cfg.get("threshold_pct"),
                   "checkpoint identity differs")
        if info["check_state"]=="checked":
            cost.check(info["native_enter_ms"]<=check_at<=info["decision_ms"]<=info["native_exit_ms"] and
                       check_at>=cfg["late_phase_ms"] and start["local_target_ms"]-start["started_ms"]+0.001>=
                       cfg["variants"][method]["fixed_seconds"]*1000 and 0<info["receipt_count"]<=len(ledger),
                       "decision outside eligible budget or native search")
            actual=sm.snapshot(ledger[:info["receipt_count"]],bounds,c["call"],c["anchor_Cmax"],
                               c["selected_lower_bound"],check_at,cfg["lookback_ms"])
            cost.check(actual is not None and all(info[k]==v for k,v in actual.items()) and
                       info["requested_stop"]==sm.stop_decision(method,actual,cfg.get("threshold_pct"),info["draw"]),
                       "stop decision differs from time-censored replay")
        else:
            cost.check(not info["requested_stop"],"unobserved stop")
        c.update({k:info.get(k) for k in sm.FIELDS})
        c["target_seconds"]=cfg["variants"][method]["fixed_seconds"]
    row["threshold_pct"]=cfg.get("threshold_pct")
    row["checkpoints"]={str(t):min((e["Cmax"] for e in row["anytime_trace"] if e["ms"]<=1000*t),default=None)
                        for t in (10,30,50,60)}
    row["checked"]=sum(c.get("check_state")=="checked" for c in row["calls"])
    row["stops"]=sum(bool(c.get("requested_stop")) for c in row["calls"])
    checked=[c for c in row["calls"] if c.get("check_state")=="checked"]
    row["absolute_gap_le2"]=sum(c["gap1"]<=2 for c in checked)
    row["threshold_disagreements"]=sum(sm.stop_decision("RelativeStop",c,5,0)!=sm.stop_decision("RelativeStop",c,20,0)
                                       for c in checked)
    return row


if __name__=="__main__":
    from experiments import stop8_loop as loop
    cli=argparse.ArgumentParser()
    cli.add_argument("--worker",type=Path,required=True)
    with patch.object(loop,"execute_worker",execute):
        raise SystemExit(loop.worker(cli.parse_args().worker))
