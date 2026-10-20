"""Five exact CPU TT interventions at cached A3 entry states; no rollout."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import json
import numpy as np
import reproduce as original


def main():
    here=original.OUT
    coverage=list(csv.DictReader((here/'formal_state_coverage.csv').open()))
    traces={x['id']:x for x in csv.DictReader((here/'trajectory_audit.csv').open())}
    cases={x['id']:x for x in json.loads((original.BASE/'protocol.json').read_text())['cases']}
    selected=[]
    for term,n in [('success',3),('deadlock',1),('timeout',1)]:
        selected.extend([r for r in coverage if r['termination']==term and r['A3_local']=='True'][:n])
    rt=original.OtherRuntime('toy_give_way')
    outputs=[]
    for row in selected:
        case=cases[row['id']];t=int(row['first_A3_step'])
        with np.load(traces[row['id']]['trace']) as trace:
            p=trace['positions'];u=trace['actions'];keys=trace['noise_keys']
            env=rt.make_env(case['state']);env.positions=p[t].copy()
            env.velocities=u[t-1].copy() if t else np.zeros((2,2));env.step_count=t
            key=original.jax.numpy.asarray(keys[t]);eta=np.asarray(case['eta'])
            assert np.array_equal(np.asarray(original.key_for(case,t)),keys[t])
            _,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
            B=rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values
            eta0=eta.copy();eta0[1]=0
            a0,cert0=safety.project(safe+sum(e*b for e,b in zip(eta0,B)))
            a1,cert1=safety.project(safe+delta)
            assert np.max(np.abs(a1-u[t]))<1e-10
            sign=1 if row['role']=='R' else -1
            def progress(a):return [float(a[0,0]-a[1,0]),float(sign*(a[1,1]-a[0,1]))]
            residual=safety.A@a1.ravel()-safety.lower
            active=['pair' if i==0 else f'agent{(i-1)//8}_wall{(i-1)%8}'
                    for i,r in enumerate(residual) if r<=1e-7]
            active += [f'speed{i}' for i in range(2) if abs(np.linalg.norm(a1[i])-.5)<=1e-7]
            def formal_coordinates(a):
                return a if row['role']=='R' else a[::-1]*np.array([-1.,1.])
            outputs.append(dict(case_id=row['id'],source_state_uid=row['state_uid'],local_uid=f"{row['state_uid']}:{row['id']}:t{t}:role{row['role']}",
                                timestep=t,role=row['role'],positions=p[t].tolist(),noise_key=keys[t].tolist(),
                                positions_formal=formal_coordinates(p[t]).tolist(),
                                B_formal=[formal_coordinates(b).tolist() for b in B],
                                eta=eta.tolist(),B=[b.tolist() for b in B],raw_ds_dz=[progress(b) for b in B],
                                u_perp0=a0.tolist(),u_full=a1.tolist(),ds_dz_perp0=progress(a0),ds_dz_full=progress(a1),
                                perp_delta_dz=progress(a1)[1]-progress(a0)[1],active=active,
                                certificate_full=safety.certificate(a1),certificate_perp0=safety.certificate(a0),
                                A3_record=row,controller_uid=original.uid(rt,'TT'),
                                projector_status=[cert0['status'],cert1['status']]))
    (here/'formal_local_probes.json').write_text(json.dumps(outputs,indent=2,sort_keys=True)+'\n')
    print(json.dumps([dict(id=x['case_id'],step=x['timestep'],outcome=x['A3_record']['termination'],
                           dz0=x['ds_dz_perp0'][1],dz=x['ds_dz_full'][1],delta=x['perp_delta_dz'],active=x['active']) for x in outputs],indent=2))


if __name__=='__main__':main()
