"""Stage deterministic posthoc bootstrap compute memo; no statistics changed."""
from experiment import *
from atomic_write import write
import ast,subprocess,difflib


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    cpp=HERE/'bootstrap_fp_control_20261022.cpp';assert not cpp.exists()
    cpp.write_text('''#include <cstdint>
#include <cfenv>
#include <xmmintrin.h>
extern "C" void bootstrap_fp_control(uint32_t* out){
 out[0]=uint32_t(std::fegetround());out[1]=_mm_getcsr()&0xffc0;
 unsigned short control;asm volatile("fnstcw %0":"=m"(control));out[2]=control;
}
''')
    so=cpp.with_suffix('.so');subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
    pairs={'statistics_header_reader_20261023.py':'statistics_bootstrap_memo_reader_20261022.py','statistics_training_selected_reader_20261022.py':'statistics_training_selected_memo_reader_20261022.py'};rows=[]
    for old,new in pairs.items():
        before=(HERE/old).read_text();assert before.count('def bootstrap(M,types,iterations=5000):')==1
        after=before.replace('def bootstrap(M,types,iterations=5000):','def bootstrap_uncached(M,types,iterations=5000):',1)
        needle='def contrast(point,draws,arms,coefficients):'
        after=after.replace(needle,"def bootstrap(M,types,iterations=5000):\n    from bootstrap_exact_memo_20261022 import memoized_bootstrap\n    return memoized_bootstrap(M,types,iterations,bootstrap_uncached)\n\n"+needle,1)
        original=next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap');candidate=next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap_uncached');candidate.name='bootstrap';assert ast.dump(original)==ast.dump(candidate)
        p=HERE/new;assert not p.exists();p.write_text(after);diff=p.with_suffix('.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),uncached_kernel_AST_identical=True,diff=str(diff)))
    write(HERE/'BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.json',dict(utc=utc(),history_checked=['CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json','ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json','statistics_correction_supplement_20261022.py'],gap='Class scopes can have byte-identical paired matrices,but previously repeat5000 deterministic local-RNG draws;retain all requested scope outputs and memoize only identical pure kernel inputs',static_copies=rows,key='Function object,iteration count,complete matrix/type dtype+shape+strides+bytes,IEEE rounding/SSE/x87 controls,numpy error mode',returned_arrays_copied=True,local_RNG_unmodified=True,capacity_entries=16,capacity_payload_bytes=67108864,activated=False,new_rollouts=0,new_training=0))
    seal('BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json',[Path(__file__),cpp,so,HERE/'bootstrap_exact_memo_20261022.py',HERE/'BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.json']+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],candidate_only=True,mathematical_kernel_AST_unchanged=True,production_activated=False)
    print('Staged deterministic bootstrap memo candidate;not activated',flush=True)


if __name__=='__main__':main()
