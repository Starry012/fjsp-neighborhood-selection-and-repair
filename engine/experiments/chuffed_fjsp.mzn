include "cumulative.mzn";

int: operations;
int: machines;
int: modes;
int: precedences;
int: horizon;
set of int: Operation = 1..operations;
set of int: Mode = 1..modes;

array[Mode] of Operation: mode_operation;
array[Mode] of 1..machines: mode_machine;
array[Mode] of int: mode_duration;
array[1..precedences] of Operation: predecessor;
array[1..precedences] of Operation: successor;
array[Operation] of 0..horizon: ect_start;
array[Mode] of bool: ect_present;

array[Operation] of var 0..horizon: starts;
array[Operation] of var min(mode_duration)..max(mode_duration): durations;
array[Mode] of var bool: present;
var 0..horizon: makespan = max([starts[o] + durations[o] | o in Operation]);

constraint forall(o in Operation)(
    sum(m in Mode where mode_operation[m] = o)(bool2int(present[m])) = 1 /\
    durations[o] = sum(m in Mode where mode_operation[m] = o)(mode_duration[m] * bool2int(present[m]))
);
constraint forall(k in 1..precedences)(
    starts[predecessor[k]] + durations[predecessor[k]] <= starts[successor[k]]
);
constraint forall(machine in 1..machines)(
    cumulative([starts[mode_operation[m]] | m in Mode where mode_machine[m] = machine],
               [mode_duration[m] | m in Mode where mode_machine[m] = machine],
               [bool2int(present[m]) | m in Mode where mode_machine[m] = machine], 1)
);
constraint redundant_constraint(cumulative(starts, durations, [1 | o in Operation], machines));

solve :: seq_search([warm_start(present, ect_present), warm_start(starts, ect_start)]) minimize makespan;

output ["{\"makespan\":", show(makespan), ",\"starts\":", show(starts),
        ",\"modes\":", show([m | m in Mode where fix(present[m])]), "}"];
