"""Common disjunctive integer-time FJSP MILP for Gurobi and COPT."""

from itertools import combinations
from time import perf_counter

from fjsp import validate_schedule


def pairs(instance):
    job_of = {op:j for j,chain in instance.jobs.items() for op in chain}
    for a,b in combinations(sorted(instance.options),2):
        common = sorted(instance.options[a].keys() & instance.options[b].keys())
        if job_of[a] != job_of[b] and common:
            yield a,b,common


def size(instance):
    pair_count, common_count = 0,0
    for _,_,common in pairs(instance):
        pair_count += 1
        common_count += len(common)
    n, modes = len(instance.options),sum(len(opts) for opts in instance.options.values())
    return {'operations':n,'machine_assignment_binaries':modes,'order_binaries':pair_count,
            'integer_time_variables':n+1,'variables':modes+pair_count+n+1,
            'constraints':n+n+sum(len(chain)-1 for chain in instance.jobs.values())+2*common_count,
            'common_machine_pairs':common_count}


def environment(solver):
    if solver == 'gurobi':
        import gurobipy as lib
        env = lib.Env(empty=True)
        env.setParam('OutputFlag',0)
        env.start()
        return lib,lib.GRB,env,lib.Model(env=env)
    import coptpy as lib
    cfg = lib.EnvrConfig()
    cfg.set('nobanner','1')
    env = lib.Envr(cfg)
    model = env.createModel('fjsp-mip')
    model.setParam('Logging',0)
    return lib,lib.COPT,env,model


def build(solver,instance,initial,*,threads=None):
    horizon = validate_schedule(instance,initial)
    lib,C,env,model = environment(solver)
    if threads is not None:
        model.setParam('Threads',threads)
    rows = {r['operation']:r for r in initial['assignments']}
    starts, choices, hints, variables = {},{},{},{}
    def variable(name,lo,hi,kind,value):
        var = model.addVar(lb=lo,ub=hi,vtype=kind,name=name)
        variables[name],hints[name] = var,value
        return var
    for op,opts in sorted(instance.options.items()):
        starts[op] = variable(f's_{op}',0,horizon,C.INTEGER,rows[op]['start'])
        for m in sorted(opts):
            choices[op,m] = variable(f'x_{op}_{m}',0,1,C.BINARY,int(rows[op]['machine']==m))
    makespan = variable('Cmax',0,horizon,C.INTEGER,horizon)
    duration = {op:lib.quicksum(d*choices[op,m] for m,d in opts.items()) for op,opts in instance.options.items()}
    for op,opts in instance.options.items():
        model.addConstr(lib.quicksum(choices[op,m] for m in opts)==1,name=f'one_{op}')
        model.addConstr(starts[op]+duration[op]<=makespan,name=f'cap_{op}')
    for j,chain in instance.jobs.items():
        for a,b in zip(chain,chain[1:]):
            model.addConstr(starts[a]+duration[a]<=starts[b],name=f'job_{j}_{a}_{b}')
    for a,b,common in pairs(instance):
        order = variable(f'y_{a}_{b}',0,1,C.BINARY,int(rows[a]['start']<=rows[b]['start']))
        for m in common:
            xa,xb = choices[a,m],choices[b,m]
            model.addConstr(starts[a]+duration[a]<=starts[b]+horizon*(3-xa-xb-order),name=f'nonoverlap_ab_{a}_{b}_{m}')
            model.addConstr(starts[b]+duration[b]<=starts[a]+horizon*(2-xa-xb+order),name=f'nonoverlap_ba_{a}_{b}_{m}')
    model.setObjective(makespan,C.MINIMIZE)
    if solver == 'gurobi':
        model.update()
        for name,var in variables.items():
            var.Start = hints[name]
    else:
        model.setMipStart(list(variables.values()),list(hints.values()))
        model.loadMipStart()
    return {'solver':solver,'env':env,'model':model,'variables':variables,'hints':hints,
            'starts':starts,'choices':choices,'size':size(instance),'C':C}


def solve(bundle,instance,seconds=2,fixed=False):
    model,solver,C = bundle['model'],bundle['solver'],bundle['C']
    if fixed:
        for name,var in bundle['variables'].items():
            model.addConstr(var==bundle['hints'][name],name=f'fix_{name}')
    model.setParam('Threads',1)
    model.setParam('TimeLimit',seconds)
    model.setParam('MIPGap',0) if solver == 'gurobi' else model.setParam('RelGap',0)
    model.setParam('MIPGapAbs',0) if solver == 'gurobi' else model.setParam('AbsGap',0)
    model.setParam('Seed',42) if solver == 'gurobi' else model.setParam('RandSeed',42)
    started = perf_counter()
    model.optimize() if solver == 'gurobi' else model.solve()
    if solver == 'gurobi':
        status,has_solution = model.Status,model.SolCount>0
        optimal = status==C.OPTIMAL
        bound = model.ObjBound
    else:
        status,has_solution = model.status,bool(model.hasmipsol)
        optimal = status==C.OPTIMAL
        bound = model.bestbnd
    schedule = None
    if has_solution:
        def integer(var):
            value = var.X if solver=='gurobi' else var.x
            rounded = round(value)
            if abs(value-rounded)>1e-5:
                raise ValueError('nonintegral schedule/assignment')
            return rounded
        rows = []
        for op,opts in sorted(instance.options.items()):
            selected = [m for m in opts if integer(bundle['choices'][op,m])]
            if len(selected)!=1:
                raise ValueError('machine choice is not unique')
            m,start = selected[0],integer(bundle['starts'][op])
            rows.append({'operation':op,'machine':m,'start':start,'end':start+opts[m]})
        schedule = {'instance':instance.name,'assignments':rows,'makespan':max(r['end'] for r in rows)}
        validate_schedule(instance,schedule)
    return {'status':status,'optimal':optimal,'schedule':schedule,'bound':bound,'seconds':perf_counter()-started}
