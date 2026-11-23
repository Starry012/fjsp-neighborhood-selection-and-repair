"""Alternative exact formulations of the same bounded static chain FJSP."""
from fjsp import validate_schedule
from experiments.fjsp_mip_adapter import environment, pairs

FORMULATIONS = ('machine-time', 'compact', 'compact-fast', 'indicator')


def release(bundle):
    """Release Python variable handles while their native model is still alive."""
    import gc
    model, env = bundle['model'], bundle['env']
    bundle.clear()
    gc.collect()
    model.dispose()
    env.dispose()


def time_windows(instance, horizon):
    """Necessary job-chain bounds for every schedule with Cmax <= horizon."""
    minimum = {op:min(opts.values()) for op,opts in instance.options.items()}
    head, tail = {}, {}
    for chain in instance.jobs.values():
        total = 0
        for op in chain:
            head[op] = total
            total += minimum[op]
        total = 0
        for op in reversed(chain):
            tail[op] = total
            total += minimum[op]
    return {op:{'start_lb':head[op], 'start_ub':horizon-tail[op]-minimum[op],
                'end_lb':head[op]+minimum[op], 'end_ub':horizon-tail[op]}
            for op in instance.options}


def build(instance, initial, formulation, threads=1):
    if formulation not in FORMULATIONS:
        raise ValueError('unknown COPT formulation')
    horizon = validate_schedule(instance, initial)
    cp, C, env, model = environment('copt')
    model.setParam('Threads',threads)
    if formulation=='compact-fast':
        model.setParam('Presolve',1)
    if formulation=='indicator':
        model.setParam('LinearizeIndicators',0)
    rows = {r['operation']:r for r in initial['assignments']}
    windows = time_windows(instance,horizon)
    starts, ends, choices, variables, hints = {}, {}, {}, {}, {}
    def variable(name,lb,ub,kind,value):
        var = model.addVar(lb=lb,ub=ub,vtype=kind,name=name)
        variables[name],hints[name] = var,value
        return var
    for op,opts in sorted(instance.options.items()):
        w = windows[op]
        starts[op] = variable(f's_{op}',w['start_lb'],w['start_ub'],C.INTEGER,rows[op]['start'])
        ends[op] = variable(f'e_{op}',w['end_lb'],w['end_ub'],C.CONTINUOUS,rows[op]['end'])
        for m in sorted(opts):
            choices[op,m] = variable(f'x_{op}_{m}',0,1,C.BINARY,int(rows[op]['machine']==m))
    lower = max(sum(min(instance.options[op].values()) for op in chain) for chain in instance.jobs.values())
    makespan = variable('Cmax',lower,horizon,C.INTEGER,horizon)
    for op,opts in sorted(instance.options.items()):
        model.addConstr(cp.quicksum(choices[op,m] for m in opts)==1,name=f'one_{op}')
        model.addConstr(ends[op]==starts[op]+cp.quicksum(p*choices[op,m] for m,p in opts.items()),name=f'end_{op}')
        model.addConstr(ends[op]<=makespan,name=f'cap_{op}')
    for job,chain in instance.jobs.items():
        for a,b in zip(chain,chain[1:]):
            model.addConstr(ends[a]<=starts[b],name=f'job_{job}_{a}_{b}')
    machines = sorted({m for opts in instance.options.values() for m in opts})
    for m in machines:
        model.addConstr(cp.quicksum(opts[m]*choices[op,m] for op,opts in instance.options.items() if m in opts)
                        <=makespan,name=f'load_{m}')
    pair_count, common_count = 0,0
    for a,b,common in pairs(instance):
        pair_count += 1
        common_count += len(common)
        before = int(rows[a]['start']<=rows[b]['start'])
        same = int(rows[a]['machine']==rows[b]['machine'])
        if formulation=='machine-time':
            y = variable(f'y_{a}_{b}',0,1,C.BINARY,before)
            for m in common:
                xa,xb = choices[a,m],choices[b,m]
                mab = max(0,windows[a]['start_ub']+instance.options[a][m]-windows[b]['start_lb'])
                mba = max(0,windows[b]['start_ub']+instance.options[b][m]-windows[a]['start_lb'])
                model.addConstr(starts[a]+instance.options[a][m]<=starts[b]+mab*(3-xa-xb-y))
                model.addConstr(starts[b]+instance.options[b][m]<=starts[a]+mba*(2-xa-xb+y))
        elif formulation.startswith('compact'):
            y = variable(f'y_{a}_{b}',0,1,C.BINARY,before)
            z = variable(f'z_{a}_{b}',0,1,C.CONTINUOUS,same)
            for m in common:
                model.addConstr(z>=choices[a,m]+choices[b,m]-1)
            mab = max(0,windows[a]['end_ub']-windows[b]['start_lb'])
            mba = max(0,windows[b]['end_ub']-windows[a]['start_lb'])
            model.addConstr(ends[a]<=starts[b]+mab*(2-z-y))
            model.addConstr(ends[b]<=starts[a]+mba*(1-z+y))
        else:
            u = variable(f'u_{a}_{b}',0,1,C.BINARY,same*before)
            v = variable(f'v_{a}_{b}',0,1,C.BINARY,same*(1-before))
            model.addConstr(u+v<=1)
            for m in common:
                model.addConstr(u+v>=choices[a,m]+choices[b,m]-1)
            model.addGenConstrIndicator(u,1,ends[a]<=starts[b],type=C.INDICATOR_IF,name=f'if_ab_{a}_{b}')
            model.addGenConstrIndicator(v,1,ends[b]<=starts[a],type=C.INDICATOR_IF,name=f'if_ba_{a}_{b}')
    model.setObjective(makespan,C.MINIMIZE)
    model.setMipStart(list(variables.values()),list(hints.values()))
    model.loadMipStart()
    return {'solver':'copt','env':env,'model':model,'variables':variables,'hints':hints,
            'starts':starts,'ends':ends,'choices':choices,'windows':windows,'C':C,
            'size':{'formulation':formulation,'operations':len(instance.options),'variables':model.cols,
                    'linear_constraints':model.rows,'linear_nonzeros':model.elems,'indicators':model.indicators,
                    'binaries':model.bins,'integers':model.ints,'pairs':pair_count,'common_machine_pairs':common_count,
                    'presolve':1 if formulation=='compact-fast' else 'vendor-default'}}
