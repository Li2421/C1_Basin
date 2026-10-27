"""Candidate only: per-query exact cell/move masks for static geometry."""
from experiment import *
from atomic_write import write
import ast,subprocess,difflib


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix='PA_SPATIAL_MASK_CANDIDATE_20261022';assert not (HERE/(prefix+'.receipt.json')).exists()
    base=HERE/'astar_kernel_tiled.cpp';source=base.read_text()
    source=source.replace('extern "C" int astar(','''#include <new>
static uint64_t mask_stats[8]={};
extern "C" void pa_stationary_edge_stats(uint64_t* out){std::copy(mask_stats,mask_stats+8,out);}
struct CellMasks {uint16_t wall_known=0,wall_forbidden=0;std::array<uint16_t,16> known{},forbidden{};};
extern "C" int astar(''',1)
    old=''' auto edge=[&](double ax,double ay,double bx,double by,int t){
  if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;
  for(int i=0;i<nr;++i){int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);if(segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12)return false;}return true;
 };'''
    new=''' ++mask_stats[0];std::unordered_map<uint64_t,CellMasks> spatial_cells;
 auto edge=[&](double ax,double ay,double bx,double by,int t,bool grid_edge=false,uint64_t cell=0,int move=0){
  CellMasks* cache=nullptr;const uint16_t bit=uint16_t(1u<<move);
  if(grid_edge&&t>0&&nr<=16){
   ++mask_stats[1];auto found=spatial_cells.find(cell);
   if(found!=spatial_cells.end())cache=&found->second;
   else if(spatial_cells.size()<65536){try{cache=&spatial_cells.emplace(cell,CellMasks{}).first->second;++mask_stats[4];mask_stats[7]=std::max(mask_stats[7],uint64_t(spatial_cells.size()));}catch(const std::bad_alloc&){++mask_stats[6];}}
   else ++mask_stats[5];
  }
  bool wall_valid;
  if(cache&&(cache->wall_known&bit)){++mask_stats[2];wall_valid=!(cache->wall_forbidden&bit);}
  else{wall_valid=memo_geometry(ax,ay,bx,by,nw,walls,margin);if(cache){cache->wall_known|=bit;if(!wall_valid)cache->wall_forbidden|=bit;}}
  if(!wall_valid)return false;
  for(int i=0;i<nr;++i){
   const bool stationary=cache&&t>=lens[i]-1;
   if(stationary&&(cache->known[i]&bit)){++mask_stats[2];if(cache->forbidden[i]&bit)return false;continue;}
   int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);++mask_stats[3];
   const bool forbidden=segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12;
   if(stationary){cache->known[i]|=bit;if(forbidden)cache->forbidden[i]|=bit;}
   if(forbidden)return false;
  }
  return true;
 };'''
    assert source.count(old)==1;source=source.replace(old,new)
    assert source.count('for(auto&m:moves)')==1;source=source.replace('for(auto&m:moves)','for(int mi=0;mi<9;++mi)',1).replace('for(int mi=0;mi<9;++mi){int x=', 'for(int mi=0;mi<9;++mi){const auto&m=moves[mi];int x=',1)
    assert source.count('edge(ax,ay,x*res,y*res,u.t)')==1;source=source.replace('edge(ax,ay,x*res,y*res,u.t)','edge(ax,ay,x*res,y*res,u.t,true,code(u.x,u.y,0),mi)',1)
    cpp=HERE/'astar_kernel_spatial_mask_candidate_20261022.cpp';assert not cpp.exists();cpp.write_text(source);so=cpp.with_suffix('.so')
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(cpp),'-o',str(so)],check=True)
    pairs={}
    for family in ('pa_memo_candidate','pa_memo_query_disk_candidate'):
        for stem in ('tiled_planning','reservation_certificate','reservation_certificate_v2','runtime_correction'):pairs[f'{stem}_{family}_20261022.py']=f'{stem}_{family}_spatial_mask_20261022.py'
    rows=[]
    for old,new in pairs.items():
        text=(HERE/old).read_text()
        for a,b in sorted(pairs.items(),key=lambda p:-len(p[0])):text=text.replace(Path(a).stem,Path(b).stem)
        text=text.replace('astar_kernel_tiled.so',so.name);ast.parse(text);p=HERE/new;assert not p.exists();p.write_text(text)
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p)))
    diff=cpp.with_suffix('.cpp.diff');diff.write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),source.splitlines(True),fromfile=base.name,tofile=cpp.name)))
    write(HERE/(prefix+'.json'),dict(utc=utc(),history=['PA_SAVED_QUERY_PROFILE_20261022.receipt.json','PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json','PA_STATIONARY_EDGE_CANDIDATE_DECISION_20261022.receipt.json'],gap='Tail-only four-double-bit hash did not improve20-query matrix;remaining99M reservation tests often include individually stationary references and repeated cell/move wall lookups. Measure cheaper one-cell lookup and per-reference masks.',theorem='Within one native query,the same grid cell and move at t>0 produces exactly the same four float64 coordinates. Wall predicates are time-independent;reference i is time-independent once t>=lens[i]-1. Cache individual unchanged booleans, preserving reference order.Continuous-start initial edges and all goal/hold edges bypass masks;nr>16 also bypasses. Horizon,heap,parent traversal,moves,serials and every comparison unchanged.',static_copies=rows,per_query_cell_cap=65536,per_cell_value_bytes=68,original_native_source_sha256=sha(base),native_candidate_source_sha256=sha(cpp),native_candidate_library_sha256=sha(so),activated=False,requires='Finite byte tests with actual capacity fallback;four-scene saved cold queries;fullPA0/PA1 saved episodes;material measured improvement',new_rollouts=0,new_labels=0,new_journals=0))
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'.json'),cpp,so,diff]+[HERE/v for v in pairs.values()],candidate_only=True,production_activated=False)
    print('Staged exact spatial-mask candidate;not deployed',flush=True)


if __name__=='__main__':main()
