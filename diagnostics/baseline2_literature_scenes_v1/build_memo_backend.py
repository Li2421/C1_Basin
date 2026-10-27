"""Memoize the exact static-wall predicate; do not change search or geometry."""
from experiment import *
import subprocess
import difflib
HELPER='''
#include <array>
#include <cstring>
struct EdgeKey { std::array<uint64_t,4> b; bool operator==(const EdgeKey&o)const{return b==o.b;} };
struct EdgeHash {size_t operator()(const EdgeKey&k)const{size_t h=0;for(auto b:k.b)h^=std::hash<uint64_t>{}(b)+0x9e3779b9+(h<<6)+(h>>2);return h;}};
static std::vector<double> memo_walls;
static double memo_margin=0;
static std::unordered_map<EdgeKey,bool,EdgeHash> memo;
static bool ready=false;
static void prepare_memo(int nw,const double* walls,double margin){
 if(!ready||memo_walls.size()!=size_t(4*nw)||std::memcmp(&margin,&memo_margin,sizeof(double))!=0||(nw&&std::memcmp(memo_walls.data(),walls,4*nw*sizeof(double))!=0)){
  memo_walls.assign(walls,walls+4*nw);memo_margin=margin;memo.clear();ready=true;
 }
}
static bool memo_geometry(double ax,double ay,double bx,double by,int nw,const double* walls,double margin){
 if(nw==0)return true;
 EdgeKey key;double values[4]={ax,ay,bx,by};std::memcpy(key.b.data(),values,sizeof(values));
 auto found=memo.find(key);if(found!=memo.end())return found->second;
 bool valid=true;
 for(int w=0;w<nw;++w){const double* z=walls+4*w;if(segseg(ax,ay,bx,by,z[0],z[1],z[2],z[3])<margin-1e-12){valid=false;break;}}
 if(memo.size()>1000000)memo.clear();
 memo.emplace(key,valid);return valid;
}
'''
def main():
    inventory=[]
    for name in ('astar_kernel','cbs_kernel'):
        source=HERE/f'{name}.cpp';old=source.read_text();new=old.replace('extern "C"',HELPER+'\nextern "C"',1)
        new=new.replace(' std::vector<int> offset(nr);',' prepare_memo(nw,walls,margin);\n std::vector<int> offset(nr);',1)
        predicate='for(int w=0;w<nw;++w){const double* z=walls+4*w;if(segseg(ax,ay,bx,by,z[0],z[1],z[2],z[3])<margin-1e-12)return false;}'
        assert predicate in new;new=new.replace(predicate,'if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;')
        target=HERE/f'{name}_memo.cpp';target.write_text(new);patch=HERE/f'{name}_MEMO_ONLY.diff';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(target))))
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(target),'-o',str(HERE/f'{name}_memo.so')],check=True)
        inventory.append(dict(canonical_source=str(source),canonical_sha256=sha(source),memo_source=str(target),memo_source_sha256=sha(target),memo_binary_sha256=sha(HERE/f'{name}_memo.so'),patch_sha256=sha(patch)))
    # Exact local wrappers: change only the library and new class names.
    old=(HERE/'planning.py').read_text();new=old.replace("'astar_kernel.so'","'astar_kernel_memo.so'").replace("'cbs_kernel.so'","'cbs_kernel_memo.so'")
    (HERE/'memo_planning.py').write_text(new+'\nfrom controllers import Nominal as CanonicalNominal\nclass MemoNominal(CanonicalNominal):\n    def action(self,env,key):\n        if self.kind in (\'prioritized_astar\',\'conflict_based_search\'):\n            if not hasattr(self,\'planner\'):self.planner=Planner(self.row)\n            return self.planner.action(env)\n        return super().action(env,key)\n')
    oldrun=(HERE/'flow_runtime.py').read_text();newrun=oldrun.replace('from controllers import Nominal,bound','from controllers import bound\nfrom memo_planning import MemoNominal as Nominal')
    assert newrun!=oldrun;(HERE/'flow_runtime_memo.py').write_text(newrun)
    worker=(HERE/'worker_flow.py').read_text().replace('from flow_runtime import run','from flow_runtime_memo import run')
    worker=worker.replace("result,arrays=run(c);save(stage,c,result,arrays)","result,arrays=run(c);result['execution_backend']=dict(canonical='flow_runtime.py',actual='flow_runtime_memo.py',proof='MEMO_EQUIVALENCE.receipt.json',source_sha256=sha(HERE/'flow_runtime_memo.py'));save(stage,c,result,arrays)")
    worker=worker.replace("prereg();cases=", "prereg();proof=read(HERE/'MEMO_EQUIVALENCE.receipt.json');assert all(sha(p)==h for p,h in proof['files'].items());cases=")
    (HERE/'worker_flow_memo.py').write_text(worker)
    (HERE/'flow_memo.sbatch').write_text((HERE/'flow.sbatch').read_text().replace('worker_flow.py','worker_flow_memo.py'))
    write(HERE/'MEMO_BACKEND_BUILD.json',dict(utc=utc(),kernels=inventory,semantic_proof='Each cached boolean is obtained by the byte-identical original segment predicate on the exact four endpoint float64 bit strings and identical complete wall/margin bytes. All temporal reservation checks, negative constraints, search order, budgets and tracking remain unchanged. Forgetting the cache changes time only.',single_thread_per_process=True,canonical_execution_identity_preserved=True))
if __name__=='__main__':main()
