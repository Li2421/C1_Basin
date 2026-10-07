"""Pre-registered legal waiting witness; full-Q evidence, no model adaptation."""
import argparse,json
import numpy as np
import jax
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev,rotation_confirmation_b as rot
from new_benchmark_common.basin_dataset_v1 import CONDITIONING_FLOW_ROOT
from new_benchmark_common.safety_eta3 import state_token
from shared_rollout_db.src.rollout_db import uid,connect,canonical

OUT=a.ROOT/'diagnostics/orthoflow3_closed_loop_contract_v1/waiting_probe'
a.bd.OUT=OUT/'registry';a.bd.WORK=OUT

def dump(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

class Runtime(rot.Runtime):
    def _training_controller(self,chain,parent):
        d=super()._training_controller(chain,parent)
        d['payload']['conditioning']='contract_legal_waiting_probe_v1'
        d['uid']=uid('ctl',d['payload'])
        return d

def freeze():
    if (OUT/'states.json').exists():raise RuntimeError('Already frozen')
    models=ev.load_models();records=[];panels=[];witnesses=[]
    source=a.load(a.OUT/'initial_expansion/states.json')
    for sc in ('four_way_intersection','ring_exchange'):
        r=Runtime(sc,[])
        rows=sorted([x for x in source if x['scenario']==sc and x['split']=='train'],key=lambda x:a.digest('contract-wait|'+x['state_uid']))[:2]
        for row in rows:
            phy=a.decode(row['structured_state']);anchor=uid('contract_anchor',{'source':row['state_uid']})
            initial={**phy,'rng_anchor':anchor}
            env=r.make_env();r.reset(env,{'physical':initial})
            key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(anchor))
            hh=[];pair=[];eta=None
            for target in (1,r.config.max_steps-1):
                while env.step_count<target:
                    env.step(np.zeros((4,2)))
                    assert not env.done,'Waiting must remain a valid live trajectory'
                physical={'positions':env.positions.tolist(),'velocities':np.zeros((4,2)).tolist(),
                    'goals':env.goals.tolist(),'timestep':target,'normalized_episode_time':target/r.config.max_steps,'rng_anchor':anchor}
                flow=r.flow_world(env,key)
                cond={'flat':np.r_[r.observation(env).ravel(),flow.ravel()].tolist()}
                ch=a.digest(physical);sid=uid('state',{'scenario':r.scenario_uid,'content':ch})
                rr={**row,'state_uid':sid,'structured_state':physical,'conditioning':cond}
                tr=b.transformed(rr);h,_=b.inputs(tr);hh.append(h)
                if eta is None:
                    p=ev.proposals(tr,models,seed_identity=anchor);eta=p['etas'][p['selected_index']]
                records.append({'scenario':sc,'uid':sid,'alias':f'contract_wait_{sc}_{row["state_id"]}_{target}',
                    'physical':physical,'content_hash':ch,'index':len(records),'provenance':{'parent':row['state_uid'],
                    'split':'train','source':'legal_zero_action_wait','target_step':target},'metadata':{}})
                panels.append({'state_uid':sid,'etas':[[0.,0.,0.],eta]});pair.append(sid)
            dist=np.linalg.norm(env.goals-env.positions,axis=-1)
            assert np.any(dist>r.config.max_speed*r.config.dt+r.config.goal_tolerance)
            witnesses.append({'scenario':sc,'parent':row['state_uid'],'state_pair':pair,
                'h_hashes':[a.digest(x.tolist()) for x in hh],'h_max_error':float(np.max(np.abs(hh[0]-hh[1]))),
                'late_remaining_steps':1,'goal_distances':dist.tolist(),
                'one_step_reach_bound':r.config.max_speed*r.config.dt+r.config.goal_tolerance,
                'late_success_analytically_impossible':True,'same_eta_panel':True,'matched_future_rng':anchor})
    dump('states.json',records);dump('proposals.json',{'states':panels,'frozen_before_outcomes':True})
    dump('preregistration.json',{'witnesses':witnesses,'canonical_seeds':list(a.SEEDS),
        'new_continuation_budget':256,'pass_rule':'Exact h alias and large paired Q difference; late infeasibility independently bounded',
        'not_threshold_flip_test':True,'no_test_states':True})
    return witnesses

def runtimes():
    for sc in ('four_way_intersection','ring_exchange'):
        r=Runtime(sc,[s for s in a.load(OUT/'states.json') if s['scenario']==sc]);r.output=OUT/sc;r.output.mkdir(parents=True,exist_ok=True)
        path=OUT/'execution';r.experiment_uid=uid('exp',{'path':str(path)})
        with connect() as con:
            con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                (r.experiment_uid,'closed_loop_contract_waiting',str(path),a.sha(OUT/'proposals.json'),a.sha(__file__),canonical({'train_only':True})))
            con.commit()
        yield sc,r

def jobs(r):
    pp={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i}
        for s in r.states for i,eta in enumerate(pp[s['uid']]['etas'])]

def collect():
    rows=[]
    for sc,r in runtimes():
        for j in jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(missing)
            rows.append({'scenario':sc,'state_uid':j['state']['uid'],'eta_index':j['index'],
                'successes':sum(x['success'] for x in valid.values()),'valid':len(valid),'numerical':len(num),
                'collisions':sum(x['collision'] for x in valid.values()),'terminal_reasons':[x['outcome'] for x in valid.values()],
                'rollout_uids':[x['rollout_uid'] for x in valid.values()]})
    dump('results.json',rows);return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    ev.OUT=OUT;ev.runtimes=runtimes;ev.jobs=jobs
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
