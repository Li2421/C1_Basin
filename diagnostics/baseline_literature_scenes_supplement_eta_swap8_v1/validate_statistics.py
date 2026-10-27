"""Synthetic analysis check only; no simulation or new research evidence."""
import bootstrap
from experiment import *
from statistics_supplement import infer
import numpy as np
rows=[]
for type_code,ctls in [('A',['A0']),('B',['B0','B1'])]:
    for ctl in ctls:
        for state in range(3):
            for seed in (0,1):
                for arm in ('baseline','treatment'):
                    rows.append(dict(controller=ctl,type=type_code,N=8,state=state,seed=seed,action=arm,success=int(arm=='treatment'),upper=int(arm=='treatment')))
r=infer(rows,['baseline','treatment'],(0,1));d=r['methods']['treatment']['versus']['baseline'];assert d['delta']==1 and d['ci95']==[1.,1.]
assert r['methods']['baseline']['versus']['baseline']['ci95']==[0.,0.]
# Unequal controller multiplicity must not change equal type weight.
for row in rows:row['success']=row['upper']=int(row['type']=='A')
r2=infer(rows,['baseline','treatment'],(0,1));assert r2['methods']['baseline']['success_lower']==.5
write(HERE/'ANALYSIS_HELPER_VALIDATION.json',dict(passed=True,synthetic_only=True,new_rollouts=0,paired_constant_effect=d,type_equal_weighted_success=.5))
print('Analysis helper validation passed',flush=True)
