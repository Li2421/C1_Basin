"""Add reproducibility and full-word method cost tables after analysis."""
from common import *
import subprocess,csv
metadata=dict(utc=utc(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],architecture=subprocess.check_output(['uname','-m'],text=True).strip(),command='g++ -O3 -std=c++17 -shared -fPIC -ffp-contract=off diagnostics/blackbox_families_v1/astar_kernel.cpp -o diagnostics/blackbox_families_v1/astar_kernel.so',binary_sha256=sha(HERE/'astar_kernel.so'),no_package_installation=True)
write(HERE/'FAST_ASTAR_BUILD.json',metadata)
if (HERE/'results.json').exists():
 r=read(HERE/'results.json');out=[]
 for name,unit in list(r['families'].items())+[('combined',r['combined'])]:
  for k in [2,4,8,16]:
   m=unit['methods'].get('pilot_'+str(k))
   if m:out.append(dict(unit=unit.get('unit',name),pilot_states=k,closed_loop_labels_per_controller=2*k,mean_regret_lower_pp=m['mean_regret_pp'][0],mean_regret_upper_pp=m['mean_regret_pp'][1],max_regret_lower_pp=m['max_regret_pp'][0],max_regret_upper_pp=m['max_regret_pp'][1]))
 with (HERE/'PILOT_LABEL_COST_CURVE.csv').open('w') as file:
  writer=csv.DictWriter(file,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
 write(HERE/'PILOT_LABEL_COST_CURVE.json',out)
