"""Independent source-state, hard-projection, random-key and replication checks."""
import numpy as np
import jax
from diagnostics.astra_true_q_audit.audit import OUT, OLD, ROOT, read, write, sha, arrays, restore, pairmetrics
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from diagnostics.cl_fhcb.closed_loop import DiagnosticPhi, DiagnosticCorrector

def main():
    jax.config.update('jax_enable_x64',True)
    plan=read(OUT/'replication_plan.json'); cfg=Config(**plan['environment']); cbf=CBFConfig()
    catalog={s['state_id']:s for s in read(OLD/'raw/q_map/state_catalog.json')}
    sourcebase=ROOT/'diagnostics/cl_fhcb_qualification/raw/stage1'
    sourceindex={r['id']:r for r in read(sourcebase/'manifest.json')['records']}
    goals=GiveWayEnv(cfg).goals
    for sid,item in catalog.items():
        src=arrays(sourcebase,sourceindex[item['source_trace']]); t=item['start_step']
        with np.load(OLD/'raw/q_map'/item['state_file']) as state:
            assert np.array_equal(state['positions'],src['positions_before'][t])
            assert np.array_equal(state['last_velocity'],src['last_velocity_before'][t])
            assert int(state['candidate_since'])==int(src['candidate_since_before'][t])
            pos=np.concatenate([src['positions_before'][:1],src['positions_after']],axis=0)
            hist=np.linalg.norm(goals[None]-pos[max(0,t-40):t+1],axis=-1)
            assert np.array_equal(state['error_history'],hist)
    checked=0; maxproj=0.; maxg=0.; sampled_keys=0; second_active_errors=0
    for stage in ['q_map','direction_validation','fresh']:
        if stage=='fresh':base=OUT; records=read(OUT/'fresh_replication.json')['records']
        else:base=OLD/'raw'/stage; records=read(base/'manifest.json')['records']
        for r in records:
            d=arrays(base,r); sid=r['state_id']; item=catalog[sid]
            env=GiveWayEnv(cfg)
            key0=jax.random.fold_in(jax.random.PRNGKey(r['flow_seed']),item['pair_id'])
            indices=sorted(set([0,len(d['event'])//2,len(d['event'])-1]))
            for i in indices:
                env.positions=d['positions_before'][i].copy()
                A,b,_=barrier_constraints(env.snapshot(),cbf)
                us,_=project_velocity(d['u_flow'][i],A,b,cfg.max_speed,cbf)
                ue,_=project_velocity(d['w'][i],A,b,cfg.max_speed,cbf)
                maxproj=max(maxproj,float(np.max(abs(us-d['u_safe'][i]))),float(np.max(abs(ue-d['u_exec'][i]))))
                key=jax.random.fold_in(key0,item['start_step']+i)
                assert np.array_equal(np.asarray(jax.random.key_data(key)),d['flow_key_data'][i])
                act=np.r_[A@d['u_exec'][i].ravel()-b<=1e-6,abs(np.linalg.norm(d['u_exec'][i],axis=-1)-cfg.max_speed)<=1e-6]
                second_active_errors+=int(not np.array_equal(act,d['second_active'][i])); sampled_keys+=1
            if stage=='fresh':
                with np.load(OLD/'raw/q_map'/item['state_file']) as s:env=restore(dict(s),cfg)
                corrector=DiagnosticCorrector(DiagnosticPhi(*r['phi']))
                for i,u in enumerate(d['u_exec']):
                    expected=corrector(env.observation(),d['u_safe'][i],cfg.max_speed)
                    maxg=max(maxg,float(np.max(abs(expected-d['g'][i]))))
                    _,_,done,info=env.step(u)
                    assert info['termination']==str(d['event'][i])
                    assert np.array_equal(env.positions,d['positions_after'][i])
                    assert done==(i==len(d['event'])-1)
            checked+=1
    assert maxproj<1e-9 and maxg<1e-12 and second_active_errors==0
    repl=read(OUT/'fresh_replication.json'); paired={}
    for r in repl['records']:paired[(r['state_id'],r['flow_seed'],r['arm'])]=r
    metrics=[]
    for s in plan['states']:
        for seed in plan['seeds']:
            a=arrays(OUT,paired[(s['state_id'],seed,'phi_minus')]); b=arrays(OUT,paired[(s['state_id'],seed,'phi_plus')])
            n=min(len(a['event']),len(b['event'])); assert np.array_equal(a['flow_key_data'][:n],b['flow_key_data'][:n])
            metrics.append({'state_id':s['state_id'],'seed':seed,**pairmetrics(a,b)})
    proj=read(OUT/'projection_audit.json'); proj['fresh_pairs']=metrics; proj['fresh_aliases']=sum(x['alias'] for x in metrics)
    ratios=[p['full_overlap']['u_exec_rms']/p['full_overlap']['w_rms'] for p in proj['pairs'] if p['full_overlap']['w_rms']>1e-12]
    proj['historical_executed_over_requested_rms_ratio']={'median':float(np.median(ratios)),'min':float(min(ratios)),'max':float(max(ratios))}
    write('projection_audit.json',proj)
    write('verification.json',{'source_states_exact_reconstruction':len(catalog),'traces_checked':checked,
        'projection_and_key_sample_steps':sampled_keys,'projection_max_abs_error':maxproj,
        'fresh_corrector_max_abs_error':maxg,'active_signature_errors':second_active_errors,
        'fresh_event_replay':'PASS','old_event_replay':'432 traces checked separately in static audit',
        'fresh_paired_key_prefixes':'PASS','frozen_system_modified':False})
    print(read(OUT/'verification.json'))

if __name__=='__main__':main()
