// Native translation of J. C. Seck Tuoh Mora's GLNSA v2.0 (PeerJ CS 574, supp-1).
// GLNSA_DEPTH60_WEBSSH.md documents the source and the budget adaptations.
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/prctl.h>
#include <unistd.h>
#endif

using Vec = std::vector<int>;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
constexpr int INF = std::numeric_limits<int>::max();

struct Problem {
    int jobs, machines, n;
    Vec first, owner;
    std::vector<Vec> duration, eligible;
    explicit Problem(std::istream& in) {
        if (!(in >> jobs >> machines) || jobs < 1 || machines < 1) throw std::runtime_error("input dimensions");
        first.push_back(0);
        for (int j = 0; j < jobs; ++j) {
            int count;
            if (!(in >> count) || count < 1) throw std::runtime_error("empty job");
            for (int i = 0; i < count; ++i) {
                Vec row(machines), choices;
                for (int m = 0; m < machines; ++m) {
                    if (!(in >> row[m]) || row[m] < 0) throw std::runtime_error("duration");
                    if (row[m]) choices.push_back(m);
                }
                if (choices.empty()) throw std::runtime_error("empty options");
                duration.push_back(row);
                eligible.push_back(choices);
                owner.push_back(j);
            }
            first.push_back(static_cast<int>(owner.size()));
        }
        n = static_cast<int>(owner.size());
    }
    bool has_next(int op) const { return op + 1 < first[owner[op] + 1]; }
};

struct Cell { Vec os, ms; int value = INF; };

struct Tables {
    Vec end, tail, order, index, prev_machine, next_machine, next_job_op;
    std::vector<Vec> by_machine;
    int value = 0, last = -1;
    Tables(const Problem& p, const Cell& cell) :
        end(p.n), tail(p.n), order(p.n), index(p.n), prev_machine(p.n,-1),
        next_machine(p.n,-1), by_machine(p.machines) { rebuild(p,cell); }
    void rebuild(const Problem& p,const Cell& cell) {
        value=0; last=-1;
        next_job_op=p.first;
        for (auto& queue:by_machine) queue.clear();
        for (int i=0; i<p.n; ++i) {
            int j=cell.os[i], op=next_job_op[j]++, m=cell.ms[op];
            auto& queue=by_machine[m];
            int ready=op>p.first[j] ? end[op-1] : 0;
            prev_machine[op]=-1;
            next_machine[op]=-1;
            if (!queue.empty()) {
                int prev=queue.back();
                prev_machine[op]=prev;
                next_machine[prev]=op;
                ready=std::max(ready,end[prev]);
            }
            queue.push_back(op);
            index[op]=i;
            order[i]=op;
            end[op]=ready+p.duration[op][m];
            if (end[op]>value) { value=end[op]; last=op; }
        }
        for (auto it=order.rbegin(); it!=order.rend(); ++it) {
            int op=*it, succ=next_machine[op];
            tail[op]=succ>=0 ? p.duration[succ][cell.ms[succ]]+tail[succ] : 0;
            if (p.has_next(op)) tail[op]=std::max(tail[op],p.duration[op+1][cell.ms[op+1]]+tail[op+1]);
        }
    }
    Vec path(const Problem& p, const Cell& cell) const {
        Vec critical;
        int op=last;
        while (true) {
            critical.push_back(op);
            int start=end[op]-p.duration[op][cell.ms[op]];
            if (!start) break;
            // The supplement tests the preceding row of the flattened job table.
            op=op>0 && end[op-1]==start ? op-1 : prev_machine[op];
            if (op<0 || end[op]!=start) throw std::runtime_error("critical path predecessor");
        }
        std::reverse(critical.begin(),critical.end());
        return critical;
    }
    int estimate(const Problem& p,const Cell& cell,int op,int m) const {
        const auto& queue=by_machine[m];
        auto it=std::lower_bound(queue.begin(),queue.end(),index[op],
            [&](int q,int pos){return index[q]<pos;});
        int head=op>p.first[p.owner[op]] ? end[op-1] : 0;
        if (it!=queue.begin()) head=std::max(head,end[*std::prev(it)]);
        int suffix=p.has_next(op) ? p.duration[op+1][cell.ms[op+1]]+tail[op+1] : 0;
        if (it!=queue.end()) suffix=std::max(suffix,p.duration[*it][m]+tail[*it]);
        return head+p.duration[op][m]+suffix;
    }
};

struct Move { int flag=0, op=-1, machine=-1, mark=-1, score=INF; };
Move choose(const Problem& p,const Cell& cell,const Tables& t,const Vec& path,
            const Vec& tabu,int iteration) {
    Move best;
    int oldest=INF, last_machine=-1;
    for (int op:path) for (int m:p.eligible[op]) {
        if (m==cell.ms[op]) continue;
        last_machine=m;
        int score=t.estimate(p,cell,op,m), expiry=tabu[op*p.machines+m];
        if (score<t.value && (best.flag!=3 || score<best.score)) {
            best={3,op,m,m,score};
        } else if (best.flag<3 && expiry<iteration && (best.flag<2 || score<best.score)) {
            best={2,op,m,m,score};
        } else if (best.flag<=1 && expiry>=iteration && expiry<oldest) {
            best={1,op,m,m,score}; oldest=expiry;
        }
    }
    // In flags 1/2 the original stores the loop's final new_machine in the tabu table.
    if (best.flag==1 || best.flag==2) best.mark=last_machine;
    return best;
}

struct Timeout {};
struct Policy {
    int cap=250, stall=40, divisor=2;
    bool anytime=false;
};
struct Search {
    const Problem& p;
    std::mt19937 rng;
    fs::path out;
    Clock::time_point until;
    int generations=0, sequence=0, best=INF, tabu_steps=0, global_neighbors=0;
    int published_best=INF;
    Policy policy;
    Vec next_op, job_end, machine_end;
    int current_gen=0, seen=INF, last_gain=0, stagnant=1;
    std::ofstream* milestones=nullptr;
    Search(const Problem& problem,unsigned seed,const fs::path& dir,Clock::time_point deadline):
        p(problem),rng(seed),out(dir),until(deadline) {}
    int draw(int n) { return std::uniform_int_distribution<int>(0,n-1)(rng); }
    double probability() { return std::generate_canonical<double,53>(rng); }
    void check_time() const { if (Clock::now()>=until) throw Timeout{}; }
    int makespan(const Cell& cell) {
        next_op=p.first; job_end.assign(p.jobs,0); machine_end.assign(p.machines,0);
        int maximum=0;
        for (int job:cell.os) {
            const int op=next_op[job]++, m=cell.ms[op];
            const int end=std::max(job_end[job],machine_end[m])+p.duration[op][m];
            job_end[job]=machine_end[m]=end;
            maximum=std::max(maximum,end);
        }
        return maximum;
    }
    Cell initial() {
        Cell cell{p.owner,Vec(p.n)};
        std::shuffle(cell.os.begin(),cell.os.end(),rng);
        for (int i=0;i<p.n;++i) cell.ms[i]=p.eligible[i][draw(static_cast<int>(p.eligible[i].size()))];
        cell.value=makespan(cell);
        return cell;
    }
    void observe(const Cell& cell) {
        seen=std::min(seen,cell.value);
        if (cell.value>=published_best) return;
        Tables t(p,cell);
        if (cell.value!=t.value) throw std::runtime_error("stale objective");
        auto file=out/("frame-"+std::to_string(++sequence));
        std::ofstream f(file.string()+".tmp"); f.exceptions(std::ios::badbit|std::ios::failbit);
        f<<current_gen<<' '<<cell.value<<'\n';
        for (int op=0;op<p.n;++op) f<<cell.ms[op]<<' '<<t.end[op]-p.duration[op][cell.ms[op]]<<' '<<t.end[op]<<'\n';
        f.close(); fs::rename(file.string()+".tmp",file); published_best=cell.value;
    }
    void publish(const Cell& cell) {
        if (cell.value>=best) return;
        best=cell.value;  // generation incumbent alone governs the search
        last_gain=generations;
        observe(cell);
    }
    void fingerprint(const std::vector<Cell>& pop) {
        if (!milestones) return;
        std::uint64_t hash=14695981039346656037ull;
        auto mix=[&](int v){hash^=static_cast<unsigned>(v); hash*=1099511628211ull;};
        for (const auto& c:pop) {
            mix(c.value);
            for (int v:c.os) mix(v);
            for (int v:c.ms) mix(v);
        }
        *milestones<<generations<<' '<<hash<<' '<<best<<' '<<global_neighbors<<' '<<tabu_steps<<'\n';
    }
    Cell neighbor(const Cell& parent,const std::vector<Cell>& pop) {
        Cell c=parent;
        double r=probability();
        if (r<=.5) {
            int a=draw(p.n), b=draw(p.n);
            while (a==b) b=draw(p.n);
            int job=c.os[a];
            c.os.erase(c.os.begin()+a); c.os.insert(c.os.begin()+b,job);
        } else if (r<=.75) {
            for (int k=0;k<8;++k) {
                int a=draw(p.n), b=draw(p.n);
                while (c.os[b]==c.os[a]) b=draw(p.n);
                std::swap(c.os[a],c.os[b]);
            }
        } else {
            c.os=pop[draw(static_cast<int>(pop.size()))].os;
            int count=1+draw(p.n-4);
            for (int i=0;i<count;++i) if (c.os[i]!=parent.os[i]) {
                // The MATLAB loop never examines the last element.
                auto pos=std::find(c.os.begin()+i+1,c.os.end()-1,parent.os[i]);
                if (pos!=c.os.end()-1) std::swap(c.os[i],*pos);
            }
        }
        if (probability()<=.1) {
            Vec ids(p.n); std::iota(ids.begin(),ids.end(),0);
            std::shuffle(ids.begin(),ids.end(),rng);
            for (int i=0;i<p.n/2;++i) {
                int op=ids[i]; c.ms[op]=p.eligible[op][draw(static_cast<int>(p.eligible[op].size()))];
            }
        }
        c.value=makespan(c);
        if (policy.anytime) observe(c);
        ++global_neighbors;
        return c;
    }
    Cell local(Cell current,int steps) {
        Cell saved=current;
        Vec tabu(p.n*p.machines,0);
        Tables t(p,current);
        for (int i=1;i<=steps;++i) {
            check_time();
            auto path=t.path(p,current);
            auto move=choose(p,current,t,path,tabu,i);
            if (!move.flag) break;
            current.ms[move.op]=move.machine;
            tabu[move.op*p.machines+move.mark]=i+static_cast<int>(path.size()+p.eligible[move.op].size());
            t.rebuild(p,current); current.value=t.value;
            if (policy.anytime) observe(current);
            if (move.flag==3) saved=current;  // last improving estimate, even if true Cmax worsens
            ++tabu_steps;
        }
        return saved;
    }
    std::string run() {
        if (p.n<5 || p.jobs<2) throw std::runtime_error("author search requires >=5 operations and >=2 jobs");
        std::vector<Cell> pop;
        for (int i=0;i<40;++i) {
            check_time(); pop.push_back(initial());
            if (policy.anytime) observe(pop.back());
        }
        auto smaller=[](const Cell& a,const Cell& b){return a.value<b.value;};
        publish(*std::min_element(pop.begin(),pop.end(),smaller));
        fingerprint(pop);
        for (int gen=1;!policy.cap || gen<=policy.cap;++gen) {
            current_gen=gen;
            Vec order(40); std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return smaller(pop[a],pop[b]);});
            std::vector<Cell> next{pop[order[0]],pop[order[1]]};
            while (next.size()<40) {
                int a=draw(40),b=draw(40);
                while (a==b) b=draw(40);
                next.push_back(smaller(pop[b],pop[a])?pop[b]:pop[a]);
            }
            for (int i=2;i<40;++i) {
                check_time();
                Cell a=neighbor(next[i],next), b=neighbor(next[i],next);
                next[i]=smaller(b,a)?std::move(b):std::move(a);
            }
            for (int i=2;i<40;++i) { check_time(); next[i]=local(std::move(next[i]),gen/policy.divisor); }
            pop=std::move(next);
            generations=gen;
            const auto& candidate=*std::min_element(pop.begin(),pop.end(),smaller);
            stagnant=candidate.value<best?1:stagnant+1;
            publish(candidate);
            if (gen==1 || gen==20 || gen==100 || gen==250 || gen==251 || gen==500 || gen==501) fingerprint(pop);
            if (policy.stall && stagnant>=policy.stall) return "stagnation";
        }
        return "generation_limit";
    }
};

#ifdef __linux__
std::int64_t monotonic_ns() {
    timespec stamp{};
    if (clock_gettime(CLOCK_MONOTONIC,&stamp)!=0) throw std::runtime_error("monotonic clock unavailable");
    return std::int64_t(stamp.tv_sec)*1000000000+stamp.tv_nsec;
}

int budget_search(const Problem& p,const fs::path& out,std::uint32_t seed,std::int64_t until_ns,Policy policy={}) {
    // Python supplies an absolute CLOCK_MONOTONIC cutoff, which includes child startup.
    const auto sampled=Clock::now();
    const auto deadline=sampled+std::chrono::nanoseconds(until_ns-monotonic_ns());
    std::ofstream rounds(out/"rounds.txt");
    rounds.exceptions(std::ios::badbit|std::ios::failbit);
    std::ofstream checks, decisions;
    if (policy.anytime) {
        checks.exceptions(std::ios::badbit|std::ios::failbit);
        decisions.exceptions(std::ios::badbit|std::ios::failbit);
        checks.open(out/"milestones.txt"); decisions.open(out/"decisions.txt");
        decisions<<policy.cap<<' '<<policy.stall<<' '<<policy.divisor<<'\n';
    }
    int published_best=INF, frames=0;
    for (std::uint32_t round=0; monotonic_ns()<until_ns; ++round) {
        const auto begin=monotonic_ns();
        const std::uint32_t run_seed=seed+round*0x9e3779b9u;
        Search search(p,run_seed,out,deadline);
        search.policy=policy;
        if (policy.anytime && round==0) search.milestones=&checks;
        search.sequence=frames;
        search.published_best=published_best;
        std::string reason;
        try { reason=search.run(); } catch (const Timeout&) { reason="wall_clock"; }
        const auto end=monotonic_ns();
        const int local_best=policy.anytime?search.seen:search.best;
        rounds<<round+1<<' '<<run_seed<<' '<<begin<<' '<<end<<' '<<reason<<' '
              <<search.generations<<' '<<search.global_neighbors<<' '<<search.tabu_steps<<' '
              <<(local_best==INF ? -1 : local_best)<<' '<<search.sequence-frames<<'\n';
        if (policy.anytime)
            decisions<<round+1<<' '<<search.current_gen<<' '<<search.last_gain<<' '<<search.stagnant
                     <<' '<<(search.best==INF ? -1 : search.best)<<'\n';
        published_best=search.published_best;
        frames=search.sequence;
        if (reason=="wall_clock") break;
    }
    rounds.close();
    std::ofstream info(out/"budget-end.txt");
    info<<monotonic_ns()<<'\n';
    return info.good()?0:2;
}
#endif

int main(int argc,char** argv) {
    try {
        if (argc!=7 && argc!=10) throw std::runtime_error("usage: mode input output seed deadline parent_pid [generation_cap stall tabu_divisor]");
#ifdef __linux__
        if (prctl(PR_SET_PDEATHSIG,SIGTERM)!=0 || getppid()!=std::stoi(argv[6])) throw std::runtime_error("parent changed");
#endif
        std::ifstream input(argv[2]); Problem p(input);
        const auto seed=static_cast<unsigned>(std::stoul(argv[4]));
#ifdef __linux__
        if (std::string(argv[1])=="deep60" && argc==10) {
            Policy policy{std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),true};
            if (policy.cap<0 || policy.stall<0 || policy.divisor<1) throw std::runtime_error("policy");
            return budget_search(p,argv[3],seed,std::stoll(argv[5]),policy);
        }
        if (std::string(argv[1])=="budget60") return budget_search(p,argv[3],seed,std::stoll(argv[5]));
#endif
        Search search(p,seed,argv[3],Clock::now()+std::chrono::milliseconds(std::stoi(argv[5])));
        std::string reason,mode=argv[1];
        if (mode=="search") {
            try { reason=search.run(); } catch (const Timeout&) { reason="wall_clock"; }
        } else if (mode=="inspect") {
            Cell c{Vec(p.n),Vec(p.n)};
            for (int& j:c.os) if (!(input>>j)) throw std::runtime_error("OS");
            for (int& m:c.ms) if (!(input>>m)) throw std::runtime_error("MS");
            Tables t(p,c); c.value=t.value; search.publish(c);
            std::ofstream report(fs::path(argv[3])/"tables.txt");
            for (int op:t.path(p,c)) report<<op<<' ';
            report<<'\n';
            for (int op=0;op<p.n;++op) {
                report<<t.tail[op];
                for (int m:p.eligible[op]) if (m!=c.ms[op]) report<<' '<<t.estimate(p,c,op,m);
                report<<'\n';
            }
            reason="inspect";
        } else throw std::runtime_error("mode");
        std::ofstream info(fs::path(argv[3])/"stats.txt");
        info<<reason<<' '<<search.generations<<' '<<search.global_neighbors<<' '<<search.tabu_steps<<' '<<search.sequence<<'\n';
        return info.good()?0:2;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
