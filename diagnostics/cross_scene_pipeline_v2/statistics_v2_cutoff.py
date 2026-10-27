"""Unchanged preregistered core statistics plus requested cutoff accounting."""
from common import *
import statistics_v2 as original
import numpy as np

def main():
    original.main();s=read(HERE/'STATISTICS.json');tables=read(HERE/'EVALUATION_LABEL_TABLES.json');a=read(HERE/'USER_CUTOFF_AMENDMENT.json')
    gap=read(HERE/'UNCOMPLETED_CELLS.json');assert gap['count']==1 and gap['cells'][0]['rollout_uid']==a['rollout_uid']and gap['cells'][0]['method']=='generator_only'
    for group,rows in tables.items():
        for name,row in rows.items():
            done=np.asarray(row['completed']);methods=[m for m in original.CORE_METHODS if m in row['methods']]
            assert done[:,:,[row['methods'].index(m)for m in methods]].all(),(group,name,'CORE_METHOD_INCOMPLETE')
    per_scene={};values=[]
    for name,row in tables['loso'].items():
        i=row['methods'].index('generator_only');y=np.asarray(row['y'])[:,:,i];done=np.asarray(row['completed'])[:,:,i];app=np.asarray(row['applicable'])[:,:,i]
        if not app.any():continue
        mask=done&app;within=[float(y[c][mask[c]].mean())for c in range(len(y))if mask[c].any()];value=float(np.mean(within))if within else None
        per_scene[name]=dict(rate=value,completed_cells=int(mask.sum()),applicable_cells=int(app.sum()))
        if value is not None:values.append(value)
    s['generator_only']['completed_only']=dict(rate=float(np.mean(values)),per_scene=per_scene,controller_types_then_scenes_equally_weighted=True)
    s['user_cutoff']=dict(rollout_uid=a['rollout_uid'],reported_steps=4000,original_horizon=a['original_horizon'],only_affected_method='generator_only',
        original_horizon_outcome=None,imputed_failure_not_observed_timeout=True,all_core_methods_completed=True,primary_and_G1_G2_G3_unaffected=True)
    write(HERE/'STATISTICS.json',s)

if __name__=='__main__':main()
