"""Engineering candidate: exact repeated heuristic memo, no search changes."""
from experiment import *
from atomic_write import write
import ast,difflib,subprocess
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
history=dict(utc=utc(),reviewed=['CBS_SAVED_QUERY_PROFILE.json','CBS_FLAT_CACHE_SOURCE_CHECK.json','CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json','CBS_TIME_INDEX_REVIEW.json'],compatible='Same current native CBS search and scalar heuristic; prior geometry dictionary acceleration retained',gap='No prior exact (x,y) heuristic value memo benchmark; source text search found no candidate',reason='Pure identical-input computation reuse may reduce repeated sqrt calculations across time states; no new training or rollout',production_unchanged=True)
write(HERE/'CBS_HEURISTIC_CACHE_HISTORY.json',history)
original=HERE/'cbs_kernel_flat_cache.cpp';source=original.read_text();needle=' auto heur=[&](int x,int y){return norm(double(x-gx),double(y-gy))/std::sqrt(2.);};'
replacement=''' // Memo stores the ORIGINAL scalar expression for the exact integer (x,y).
 // The bounded array is memory layout only; all other keys use the same expression.
 constexpr int heuristic_extent=128,heuristic_width=2*heuristic_extent+1;
 std::vector<double> heuristic_values(heuristic_width*heuristic_width);
 std::vector<uint8_t> heuristic_seen(heuristic_width*heuristic_width,0);
 auto heur=[&](int x,int y){
  if(x < -heuristic_extent || x > heuristic_extent || y < -heuristic_extent || y > heuristic_extent)
   return norm(double(x-gx),double(y-gy))/std::sqrt(2.);
  const size_t slot=size_t(x+heuristic_extent)*heuristic_width+size_t(y+heuristic_extent);
  if(!heuristic_seen[slot]){
   heuristic_values[slot]=norm(double(x-gx),double(y-gy))/std::sqrt(2.);
   heuristic_seen[slot]=1;
  }
  return heuristic_values[slot];
 };'''
assert source.count(needle)==1;updated=source.replace(needle,replacement);assert updated.replace(replacement,needle)==source
candidate=HERE/'cbs_kernel_heuristic_cache.cpp';assert not candidate.exists();candidate.write_text(updated)
(HERE/'CBS_HEURISTIC_CACHE_SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile=original.name,tofile=candidate.name)))
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(candidate),'-o',str(HERE/'cbs_kernel_heuristic_cache.so')],check=True)
changes=[]
for original,new,replacements in [
 ('tiled_planning_flat_static.py','tiled_planning_heuristic_candidate.py',[('cbs_kernel_flat_cache.so','cbs_kernel_heuristic_cache.so')]),
 ('reservation_certificate_flat_static.py','reservation_certificate_heuristic_candidate.py',[('tiled_planning_flat_static','tiled_planning_heuristic_candidate')]),
 ('reservation_certificate_v2_flat_static.py','reservation_certificate_v2_heuristic_candidate.py',[('reservation_certificate_flat_static','reservation_certificate_heuristic_candidate')])]:
 src=(HERE/original).read_text();out=src
 for a,b in replacements:assert a in out;out=out.replace(a,b)
 ast.parse(out);dest=HERE/new;assert not dest.exists();dest.write_text(out)
 changes.append(dict(original=original,original_sha256=sha(HERE/original),new=new,new_sha256=sha(dest)))
write(HERE/'CBS_HEURISTIC_CACHE_SOURCE_CHECK.json',dict(utc=utc(),passed=True,cpp_outside_heuristic_memo_byte_identical=True,original_scalar_expression_identical=True,key='Exact integer x,y within each native search; same goal, no cross-goal reuse',fallback='Original expression outside memory window; no state or node removed',static_imports=changes,new_rollouts=0,deployment=False))
seal('CBS_HEURISTIC_CACHE_CANDIDATE_CODE.receipt.json',[HERE/n for n in ('prepare_cbs_heuristic_cache.py','CBS_HEURISTIC_CACHE_HISTORY.json','CBS_HEURISTIC_CACHE_SOURCE_DIFF.patch','CBS_HEURISTIC_CACHE_SOURCE_CHECK.json','cbs_kernel_heuristic_cache.cpp','cbs_kernel_heuristic_cache.so','tiled_parent_store.hpp')]+[HERE/r['new'] for r in changes],engineering_candidate_only=True,new_rollouts=0,no_production_change=True)
print('Prepared isolated exact heuristic memo candidate')
