"""Replace only A* parent dictionary representation, preserving every operation."""
from experiment import *
import subprocess
import difflib


def main():
    sources=[]
    for name in ('astar_kernel','cbs_kernel'):
        source=HERE/f'{name}_certified.cpp';old=source.read_text()
        assert old.count('std::unordered_map<uint64_t,uint64_t> parent;')==1
        assert old.count('parent.find(id)!=parent.end()')==1
        new='#include "tiled_parent_store.hpp"\n'+old.replace(
            'std::unordered_map<uint64_t,uint64_t> parent;',
            'TiledParentStore parent(res,max_time);').replace(
            'parent.find(id)!=parent.end()','parent.contains(id)')
        target=HERE/f'{name}_tiled.cpp';assert not target.exists();target.write_text(new)
        (HERE/f'{name}_TILED_PARENT.diff').write_text(''.join(difflib.unified_diff(
            old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(target))))
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(target),'-o',str(HERE/f'{name}_tiled.so')],check=True)
        sources.append(dict(source=str(source),source_sha256=sha(source),copy=str(target),copy_sha256=sha(target)))
    subprocess.run(['g++','-std=c++17','-O3',str(HERE/'test_tiled_parent_store.cpp'),'-o',str(HERE/'test_tiled_parent_store')],check=True)
    unit=subprocess.check_output([str(HERE/'test_tiled_parent_store')],text=True)
    write(HERE/'TILED_PARENT_DICTIONARY_TEST.json',json.loads(unit))
    p=HERE/'certified_planning.py';old=p.read_text();new=old.replace('astar_kernel_certified.so','astar_kernel_tiled.so').replace('cbs_kernel_certified.so','cbs_kernel_tiled.so')
    (HERE/'tiled_planning.py').write_text(new)
    old=(HERE/'interior_nominal.py').read_text();new=old.replace('from certified_planning import Planner','from tiled_planning import Planner').replace('class InteriorNominal','class TiledNominal');(HERE/'tiled_nominal.py').write_text(new)
    old=(HERE/'flow_runtime_interior.py').read_text();new=old.replace('from interior_nominal import InteriorNominal as Nominal','from tiled_nominal import TiledNominal as Nominal');(HERE/'flow_runtime_tiled.py').write_text(new)
    write(HERE/'TILED_PARENT_PROOF.json',dict(
        utc=utc(),sources=sources,header_sha256=sha(HERE/'tiled_parent_store.hpp'),
        proof='The 64-bit key is uniquely decomposed into bits[63:40], bits[39:20], bits[19:0]. Interior tiles index exactly these encoded bits. Membership and first-insert parent values equal unordered_map, including duplicate inserts and every uint64 value. Other keys use the original unordered_map. The search makes no parent-map iteration; contains/emplace/at are its only operations. Thus queue entries, insertion serials, accepted edges, visited states, goal tests, backtracking and all returned path bytes are unchanged by induction over queue operations. The ten-metre tile window is a storage choice only, with exact hash fallback, not a search bound.',
        algorithm_or_budget_changes=False,old_sources_unchanged=True,
        dictionary_unit=json.loads(unit),compiler_flags=['-std=c++17','-O3','-shared','-fPIC']))


if __name__=='__main__':main()
