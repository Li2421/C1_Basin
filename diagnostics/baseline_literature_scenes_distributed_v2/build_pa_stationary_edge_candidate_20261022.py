"""Candidate only: exact reservation-edge memo after every path becomes stationary."""
from experiment import *
from atomic_write import write
import ast,difflib,subprocess


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix='PA_STATIONARY_EDGE_CANDIDATE_20261022';assert not (HERE/(prefix+'.receipt.json')).exists()
    base=HERE/'astar_kernel_tiled.cpp';source=base.read_text()
    needle='extern "C" int astar('
    insert='''#include <new>
static uint64_t tail_stats[8]={};
extern "C" void pa_stationary_edge_stats(uint64_t* out){std::copy(tail_stats,tail_stats+8,out);}

'''
    assert source.count(needle)==1;source=source.replace(needle,insert+needle)
    first=''' auto edge=[&](double ax,double ay,double bx,double by,int t){
  if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;
  for(int i=0;i<nr;++i){int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);if(segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12)return false;}return true;
 };'''
    second=''' ++tail_stats[0];
 int stationary_after=0;for(int i=0;i<nr;++i)stationary_after=std::max(stationary_after,lens[i]-1);
 std::unordered_map<EdgeKey,bool,EdgeHash> stationary_edges;
 auto edge=[&](double ax,double ay,double bx,double by,int t){
  if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;
  const bool cacheable=nr>0&&t>=stationary_after;
  EdgeKey key;
  if(cacheable){
   ++tail_stats[1];double values[4]={ax,ay,bx,by};std::memcpy(key.b.data(),values,sizeof(values));
   auto found=stationary_edges.find(key);if(found!=stationary_edges.end()){++tail_stats[2];return found->second;}
  }
  bool valid=true;
  for(int i=0;i<nr;++i){int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);++tail_stats[3];if(segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12){valid=false;break;}}
  if(cacheable){
   if(stationary_edges.size()<65536){try{stationary_edges.emplace(key,valid);++tail_stats[4];tail_stats[7]=std::max(tail_stats[7],uint64_t(stationary_edges.size()));}catch(const std::bad_alloc&){++tail_stats[6];}}
   else ++tail_stats[5];
  }
  return valid;
 };'''
    assert source.count(first)==1;source=source.replace(first,second)
    cpp=HERE/'astar_kernel_stationary_edge_candidate_20261022.cpp';assert not cpp.exists();cpp.write_text(source)
    so=cpp.with_suffix('.so');subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(cpp),'-o',str(so)],check=True)
    pairs={}
    for family in ('pa_memo_candidate','pa_memo_query_disk_candidate'):
        for stem in ('tiled_planning','reservation_certificate','reservation_certificate_v2','runtime_correction'):
            old=f'{stem}_{family}_20261022.py';new=f'{stem}_{family}_stationary_edge_20261022.py';pairs[old]=new
    rows=[]
    for old,new in pairs.items():
        before=(HERE/old).read_text();after=before
        for a,b in sorted(pairs.items(),key=lambda p:-len(p[0])):after=after.replace(Path(a).stem,Path(b).stem)
        after=after.replace('astar_kernel_tiled.so','astar_kernel_stationary_edge_candidate_20261022.so')
        ast.parse(after);p=HERE/new;assert not p.exists();p.write_text(after)
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p)))
    delta=HERE/'astar_kernel_stationary_edge_candidate_20261022.cpp.diff';delta.write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),source.splitlines(True),fromfile=base.name,tofile=cpp.name)))
    plan=dict(utc=utc(),history_checked=['PA_SAVED_QUERY_PROFILE_20261022.receipt.json','PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json','RESERVATION_V2_EQUIVALENCE.receipt.json','astar_kernel_tiled.cpp','reservation_certificate_v2_pa_memo_candidate_20261022.py'],existing_evidence_reused='Saved PA queries and full episodes;no new physics rollout,labels,journals or training',remaining_gap='No previous stationary-reservation-tail edge memo found;measure native duplicate edge counts and verify exact outputs before any deployment',native_source_sha256=sha(base),native_library_sha256=sha(HERE/'astar_kernel_tiled.so'),candidate_cpp_sha256=sha(cpp),candidate_so_sha256=sha(so),static_copies=rows,entry_cap_per_native_query=65536,activated=False,theorem='At t>=max(lens-1),every reserved before/after point equals its final point. For the same four IEEE coordinate bit strings and fixed query arguments,the original segment comparisons therefore have identical booleans. Cache those booleans only;miss/full/allocation fallback compute original expressions. Parent records,heap order,serials,all horizon and projection budgets remain untouched.',stats_fields=['queries','eligible_edges','edge_hits','reserved_segment_tests','stored_edges','capacity_fallbacks','allocation_fallbacks','maximum_entries'],requires_before_deployment=['Finite path byte equivalence','Four-scene real saved queries','PA0/PA1 full2000-step saved episode byte equivalence','Storage bound/fallback review'],new_research_rollouts=0)
    write(HERE/(prefix+'.json'),plan)
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'.json'),cpp,so,delta]+[HERE/v for v in pairs.values()],candidate_only=True,production_activated=False,new_research_rollouts=0)
    print('Staged exact stationary-edge native candidate;not deployed',flush=True)


if __name__=='__main__':main()
