include "cumulative.mzn";

int: operations;
int: machines;
int: modes;
int: precedences;
int: horizon;
int: objective_lb;
bool: structured_search;
set of int: Operation = 1..operations;
set of int: Mode = 1..modes;

array[Mode] of Operation: mode_operation;
array[Mode] of 1..machines: mode_machine;
array[Mode] of int: mode_duration;
array[1..precedences] of Operation: predecessor;
array[1..precedences] of Operation: successor;
array[Operation] of int: earliest;
array[Operation] of int: latest;
array[Operation] of int: tail;
array[int] of Operation: job_last;
array[Operation] of 0..horizon: ect_start;
array[Mode] of bool: ect_present;

array[Operation] of var 0..horizon: starts =
    [let { var earliest[o]..latest[o]: s } in s | o in Operation];
array[Operation] of var int: durations =
    [let { set of int: lengths = {mode_duration[m] | m in Mode where mode_operation[m] = o};
           var lengths: d } in d | o in Operation];
array[Mode] of var bool: present;
var objective_lb..horizon: makespan = max([starts[o] + durations[o] | o in job_last]);

constraint forall(o in Operation)(
    sum(m in Mode where mode_operation[m] = o)(bool2int(present[m])) = 1 /\
    starts[o] + durations[o] + tail[o] <= makespan
);
constraint forall(m in Mode)(present[m] -> durations[mode_operation[m]] = mode_duration[m]);
constraint forall(k in 1..precedences)(
    starts[predecessor[k]] + durations[predecessor[k]] <= starts[successor[k]]
);
constraint forall(machine in 1..machines)(
    cumulative([starts[mode_operation[m]] | m in Mode where mode_machine[m] = machine],
               [mode_duration[m] | m in Mode where mode_machine[m] = machine],
               [bool2int(present[m]) | m in Mode where mode_machine[m] = machine], 1) /\
    sum(m in Mode where mode_machine[m] = machine)(mode_duration[m] * bool2int(present[m])) <= makespan
);
constraint redundant_constraint(cumulative(starts, durations, [1 | o in Operation], machines));

ann: ect_hint = seq_search([
    warm_start(present, ect_present), warm_start(starts, ect_start),
    bool_search(present, input_order, indomain_max, complete),
    int_search(starts, input_order, indomain_min, complete)
]);
ann: duration_mode_start = seq_search([
    warm_start(present, ect_present), warm_start(starts, ect_start),
    int_search(durations, smallest, indomain_min, complete),
    bool_search(present, input_order, indomain_max, complete),
    int_search(starts, smallest, indomain_min, complete)
]);
solve :: (if structured_search then duration_mode_start else ect_hint endif) minimize makespan;

output ["{\"makespan\":", show(makespan), ",\"starts\":", show(starts),
        ",\"modes\":", show([m | m in Mode where fix(present[m])]), "}"];
