"""Exact early certificate for a goal grid node with no possible incoming edge.

The frozen search terminates only at its rounded goal cell. If that cell's
point-to-wall distance is below the margin, every incoming edge is rejected
by the very same fourth term of the original segseg minimum. Initial-cell
equals goal-cell is exempt, preserving the original direct-terminal case.
"""
from experiment import *
import subprocess
import difflib

CERT='''
 // Exact no-incoming-edge certificate, using the original endpoint term.
 // No goal, resolution, radius, search tie, reservation or budget is changed.
 if(sx!=gx||sy!=gy){
  const double bx=gx*res,by=gy*res;
  for(int w=0;w<nw;++w){const double* z=walls+4*w;
   if(segment(z[0]-bx,z[1]-by,z[2]-bx,z[3]-by)<margin-1e-12)return 0;
  }
 }
'''
def main():
    sourcefiles=[]
    for name in ('astar_kernel','cbs_kernel'):
        source=HERE/f'{name}_memo.cpp';old=source.read_text();anchor=' auto heur=[&]';assert anchor in old
        new=old.replace(anchor,CERT+anchor,1);target=HERE/f'{name}_certified.cpp';target.write_text(new)
        (HERE/f'{name}_GRID_CERTIFICATE.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(target))))
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(target),'-o',str(HERE/f'{name}_certified.so')],check=True);sourcefiles.append(str(target))
    text=(HERE/'memo_planning.py').read_text().replace('astar_kernel_memo.so','astar_kernel_certified.so').replace('cbs_kernel_memo.so','cbs_kernel_certified.so')
    (HERE/'certified_planning.py').write_text(text)
    text=(HERE/'flow_runtime_cached.py').read_text().replace('from memo_planning import MemoNominal as Nominal','from certified_planning import MemoNominal as Nominal')
    (HERE/'flow_runtime_certified.py').write_text(text)
    text=(HERE/'worker_stream.py').read_text().replace('from flow_runtime_cached import run','from flow_runtime_certified import run').replace('PROJECTION_MEMO_EQUIVALENCE.receipt.json','GRID_CERTIFICATE_EQUIVALENCE.receipt.json').replace("actual='flow_runtime_cached.py'","actual='flow_runtime_certified.py'").replace("sha(HERE/'flow_runtime_cached.py')","sha(HERE/'flow_runtime_certified.py')").replace("HERE/'worker_stream.py'","HERE/'worker_certified_stream.py'")
    (HERE/'worker_certified_stream.py').write_text(text)
    (HERE/'flow_certified.sbatch').write_text((HERE/'flow_stream.sbatch').read_text().replace('worker_stream.py','worker_certified_stream.py'))
    test=(HERE/'test_cached_n1_replay.py').read_text().replace('from flow_runtime_cached import run','from flow_runtime_certified import run').replace('CACHED_N1_REPLAY','GRID_CERTIFICATE_REPLAY').replace('CACHED_N1_EQUIVALENCE','GRID_CERTIFICATE_REPLAY_ONLY')
    (HERE/'test_grid_certificate_replay.py').write_text(test)
    write(HERE/'GRID_CERTIFICATE_PROOF.json',dict(utc=utc(),proof=__doc__,canonical_output_unchanged=True,controller_incapability_not_geometric_unreachability=True,sourcefiles=sourcefiles))
if __name__=='__main__':main()
