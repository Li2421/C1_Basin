"""After pilot: all eight archived development states, no outcome-based sampling."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:18])
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import json
from scipy.stats import qmc
from oracle import inputs,DEV_SCENES,b3_pool,b5_pool,execute,make_case,OUT,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);a=ap.parse_args()
    scenes,ctls,_,_=inputs();cohort=[(s,ci,si)for s in DEV_SCENES for ci in range(2)for si in range(8,16)]
    dump(OUT/'EXPANSION_PROTOCOL.json',dict(cohort=cohort,search_seed=2026102703,
        b3_budget=168,b5_extra_per_recipe=96,stages=['A40','A168','ADD','BLEND'],
        selection='All states 8..15 in the same four development scenes; independent test unchanged',
        evidence_gap='Archived fresh contains menu16, not new Sobol/B5; existing pilot reused.'))
    cases=[make_case(scenes[s],ctls[ci],si,sp,0)for s,ci,si in cohort for sp in b3_pool()]
    rows=execute(cases,'expanded_A40',a.workers)
    def unresolved(rr,cohort):
        return [(s,ci,si)for s,ci,si in cohort if not any(r['success']for r in rr if r['scene']==s and r['controller']==ctls[ci]['type_code']and r['state_index']==si)]
    fail40=unresolved(rows,cohort);dump(OUT/'expanded_A40_failures.json',fail40)
    points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102703).random_base2(7),[.5,-.5,0.],[1.25,.5,.75])
    specs=[dict(name='b3_deep_'+str(k),mode='b3',eta=e.tolist()+[0.,0.])for k,e in enumerate(points)]
    cases=[make_case(scenes[s],ctls[ci],si,sp,0)for s,ci,si in fail40 for sp in specs]
    deep=execute(cases,'expanded_A168',a.workers)
    fail168=unresolved(deep,fail40);dump(OUT/'expanded_A168_failures.json',fail168)
    for mode in ('add','blend'):
        cases=[make_case(scenes[s],ctls[ci],si,sp,0)for s,ci,si in fail168 for recipe in ('goal','cluster','wall')for sp in b5_pool(scenes[s],si,mode,recipe)]
        execute(cases,'expanded_'+mode,a.workers)
    dump(OUT/'EXPANSION_COMPLETE.json',dict(cohorts=len(cohort),A40_failures=len(fail40),A168_failures=len(fail168)))


if __name__=='__main__':main()
