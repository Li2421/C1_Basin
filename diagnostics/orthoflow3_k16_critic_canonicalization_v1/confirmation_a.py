"""One-shot fresh Phase-A confirmation, gated on the frozen critic decision.

No invocation here trains a model or creates train/dev states. Cohort creation
is refused until Phase-A development has been explicitly sealed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
from new_benchmark_common.safety_eta3 import ScenarioRuntime,DatabaseSink,request,preflight
from four_way_intersection.scenario import sample_initial_state as sample_four
from ring_exchange.environment import sample_initial_state as sample_ring
from shared_rollout_db.src.rollout_db import canonical,connect,eta_identity,uid

OUT=a.OUT/'phase_a/confirmation'


def dump(name,value):
    a.dump('phase_a/confirmation/'+name,value)


def decision():
    d=a.load(a.OUT/'phase_a/critic_frozen.json')
    assert d['development_frozen'] is True
    assert a.sha(Path(d['checkpoint']))==d['checkpoint_sha256']
    assert a.sha(a.frozen.GENERATOR_CKPT)==d['generator_sha256']
    return d


def generate_double_pool():
    decision()
    target=OUT/'double_pool'
    if target.exists():return
    from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _generate_split
    from double_bottleneck.expert_dataset import save_dataset
    specs,records,rejections=_generate_split('k16_critic_canonicalization_A_confirmation_20261002',8,2)
    save_dataset(target,records,generation_metadata={'scientific_split':'phase_A_fresh_confirmation',
                 'storage_split_label':'val','adaptive_sampling':False,'after_critic_freeze':True})
    dump('double_generation_audit.json',{'states':len(specs),'records':len(records),'rejections':rejections})


class NewRuntime(a.bd.TrainingRuntime):
    # Same execution and RNG as frozen OrthoFlow3; no old test archive is opened.
    def _training_controller(self,chain,parent):
        return ScenarioRuntime._controller(self,chain)

    conditioning=a.frozen.FrozenNewRuntime.conditioning


class DoubleRuntime(a.frozen.DoubleFrozenRuntime):
    def _states(self):return []


def register(runtime):
    runtime.output=OUT/runtime.name;runtime.output.mkdir(parents=True,exist_ok=True)
    # The confirmation root was previously registered by the historical DB
    # importer under another experiment UID. experiment.path is UNIQUE, so an
    # INSERT OR IGNORE at that path silently omitted this experiment and made
    # every source_file insert fail its foreign key. Use a dedicated provenance
    # path; neither the frozen state/proposals nor controller identity changes.
    provenance_path=OUT/'run_provenance'
    runtime.experiment_uid=uid('exp',{'path':str(provenance_path),'phase':'A_confirmation'})
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                    (runtime.experiment_uid,'k16_critic_A_fresh_confirmation',str(provenance_path),a.digest(decision()),a.sha(__file__),canonical({'fresh':True,'adaptation_after_outcomes':False})))
        assert con.execute('SELECT 1 FROM experiment WHERE experiment_uid=?',(runtime.experiment_uid,)).fetchone(), 'confirmation experiment registration failed'
        for s in runtime.states:
            con.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                        (s['uid'],runtime.scenario_uid,'phase_A_fresh_confirmation',s['content_hash'],canonical(s['physical']),canonical({'goals':s['physical'].get('goals')}),canonical(s['metadata']),'CONTENT_EXACT'))
            con.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(runtime.scenario_uid,s['alias'],s['uid'],runtime.experiment_uid))
        con.commit()


def runtimes():
    decision()
    a.bd.OUT=OUT;a.bd.WORK=OUT
    a.frozen.OUT=OUT;a.frozen.DOUBLE_POOLS={'phase_A':OUT/'double_pool'}
    r=DoubleRuntime();states=[];dataset=r.datasets['phase_A']
    for i,family in enumerate(dataset.family_names):
        ep=dataset.by_family[family][0]
        physical={'positions':ep.initial_positions.tolist(),'velocities':ep.initial_velocities.tolist()};ch=a.digest(physical)
        env=a.frozen.db_old._initialize_env(a.frozen.db_old.Config(**dataset.config),ep)
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(92841),i),0)
        raw=np.asarray(r.policy.sample_actions(env.observation()[None],key)[0],float)
        states.append({'uid':uid('state',{'scenario':r.scenario_uid,'content':ch}), 'alias':f'K16A_DB_CONFIRM_{i:03d}',
                       'index':i,'content_hash':ch,'physical':physical,
                       'conditioning_flat':np.r_[env.observation().ravel(),raw.ravel()].tolist(),'environment_descriptor':dataset.config,
                       'metadata':{'set':'phase_A','family_id':family,'historical_seed':92841,'rollout_id':i,'fresh_confirmation':True}})
    r.states=states;register(r);yield r.name,r
    for sc,count,offset in [('four_way_intersection',24,8_000_000),('ring_exchange',60,9_000_000)]:
        r=NewRuntime(sc,[],parent=True);states=[]
        r.controllers['hard_safety_q16']=ScenarioRuntime._controller(r,'hard_safety_q16')
        r._register_training_identities()
        for i in range(count):
            if sc=='four_way_intersection':
                pos,vel,meta=sample_four(r.config,'test',offset+i)
                physical={'positions':pos.tolist(),'velocities':vel.tolist(),'goals':None}
            else:
                init=sample_ring('test',offset+i,r.config)
                physical={'positions':init.positions.tolist(),'velocities':init.velocities.tolist(),'goals':init.goals.tolist()}
            ch=a.digest(physical)
            states.append({'uid':uid('state',{'scenario':r.scenario_uid,'content':ch}),'alias':f'K16A_{sc}_CONFIRM_{i:03d}',
                           'index':i,'content_hash':ch,'physical':physical,'metadata':{'draw_index':offset+i,'distribution':'original_test','fresh_confirmation':True}})
        r.states=states;register(r);yield sc,r


def freeze():
    d=decision()
    if (OUT/'proposals.json').exists():raise RuntimeError('Confirmation already frozen; no resampling')
    generate_double_pool()
    norm=a.load(a.frozen.NORMALIZATION);gm,gp,cm,cp=a.fresh.load_models(norm)
    old_cp=cp
    cp=serialization.from_bytes(cp,Path(d['checkpoint']).read_bytes())
    rows=[];manifest=[]
    for sc,r in runtimes():
        for s in r.states:
            flat,env=r.conditioning(s);n=norm['scenarios'][sc]
            h=(np.asarray(flat,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
            c=a.fresh.context_vector(env,norm,sc)
            raw=gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,a.learn.METHOD[sc]))[0]
            rng=np.random.default_rng(a.learn.stable_int('generator-v1-proposals',41,sc,s['uid']))
            samples=np.asarray(a.learn.eta_from_noise(raw[None].repeat(16,0),jnp.asarray(rng.standard_normal((16,3)),jnp.float32)),float)
            mean=np.asarray(a.learn.eta_mean(raw),float);etas=np.asarray([mean,*samples])
            logits=np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                   jnp.asarray((etas-a.learn.CENTER)/a.learn.RADIUS),method=getattr(cm,a.learn.METHOD[sc])))
            old_logits=np.asarray(cm.apply(old_cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                   jnp.asarray((etas-a.learn.CENTER)/a.learn.RADIUS),method=getattr(cm,a.learn.METHOD[sc])))
            rows.append({'scenario':sc,'state_uid':s['uid'],'state_id':s['alias'],'etas':etas.tolist(),
                         'scores':(1/(1+np.exp(-logits))).tolist(),'critic_index':int(np.argmax(logits)),
                         'old_scores':(1/(1+np.exp(-old_logits))).tolist(),'old_critic_index':int(np.argmax(old_logits)),
                         'proposal_hash':a.digest(etas.tolist()),'controller_uid':r.controllers['orthoflow3']['uid']})
            manifest.append({'scenario':sc,'state_uid':s['uid'],'physical':s['physical'],'metadata':s['metadata']})
    dump('state_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'after_development_freeze':d,'states':manifest})
    dump('proposals.json',{'frozen_before_outcomes':True,'states':rows,'K_stochastic':16})
    return {'states':len(rows),'seed_slots':len(rows)*18*16}


def jobs(r,manifest):
    by={s['state_uid']:s for s in manifest['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3' if j<17 else 'hard_safety_q16','seeds':list(a.SEEDS),
             'metadata':{'proposal_index':j if j<17 else None,'B0_reference':j==17,'critic_index':by[s['uid']]['critic_index'],
                         'proposal_hash':by[s['uid']]['proposal_hash'],'fresh_confirmation':True}}
            for s in r.states for j,eta in enumerate(by[s['uid']]['etas']+[[0.,0.,0.]])]


def run(shard,shards):
    manifest=a.load(OUT/'proposals.json');totals=Counter();index=0
    for sc,r in runtimes():
        selected=[]
        for j in jobs(r,manifest):
            if index%shards==shard:selected.append(j)
            index+=1
        if not selected:continue
        name=f'cache/{sc}_{shard}of{shards}.json'
        dump(name,{'requests':[request(r,j['state'],j['eta'],j['chain'],a.SEEDS) for j in selected]})
        dump(name.replace('.json','_preflight.json'),preflight(OUT/name))
        sink=DatabaseSink(r,f'confirmation_{shard}of{shards}')
        try:
            # A failed DB insert may have left an already-computed, flushed raw
            # rollout behind. Recover it into the cache without appending it a
            # second time or executing a new physical rollout.
            if sink.path.exists():
                with sink.path.open() as raw:
                    for line in raw:
                        row=json.loads(line)
                        eta_uid=eta_identity(row['eta'])[0]
                        rollout_uid=uid('roll',{'state':row['state_uid'],'eta':eta_uid,
                                                'controller':row['controller_uid'],
                                                'seed':canonical({'future_index':int(row['future_index'])})})
                        with connect(True) as con:
                            present=con.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rollout_uid,)).fetchone()
                        if not present:
                            assert row['scenario']==sc and row['provenance_experiment']=='k16_critic_A_fresh_confirmation'
                            sink.insert(row,append_raw=False)
                            totals['recovered_raw']+=1
            for j in selected:
                found,num,missing=a.evidence(r,j)
                totals['reused']+=len(found);totals['existing_numerical']+=len(num)
                for seed in missing:
                    for attempt in range(4):
                        result=r.rollout(j['state'],np.asarray(j['eta'],float),seed,j['chain'])
                        result.update({**j['metadata'],'execution_attempt':attempt,'provenance_experiment':'k16_critic_A_fresh_confirmation'})
                        sink.insert(result);totals['physical_attempts']+=1
                        if not result['numerical_failure']:break
                    totals['completed_seeds']+=1;totals['numerical_unresolved']+=bool(result['numerical_failure'])
                    if totals['physical_attempts']%25==0:print(json.dumps({'shard':shard,**totals}),flush=True)
        finally:sink.finalize()
    dump(f'run_shard{shard}of{shards}.json',dict(totals));return dict(totals)


def collect():
    manifest=a.load(OUT/'proposals.json');truth={};details=[];missing=[]
    for sc,r in runtimes():
        for j in jobs(r,manifest):
            valid,num,absent=a.evidence(r,j)
            success=sum(bool(x['success']) for x in valid.values());fail=len(valid)-success
            e={'scenario':sc,'state_uid':j['state']['uid'],'candidate':j['metadata']['proposal_index'],
               'controller_uid':r.controllers[j['chain']]['uid'],'eta':j['eta'],'valid_seeds':len(valid),
               'successes':success,'failures':fail,'numerical_seeds':list(num),'missing':absent,
               'robust':True if success>=15 else False if fail>=2 else None,
               'Q16':success/16 if len(valid)==16 else None,'Q_lower':success/16,'Q_upper':(16-fail)/16,
               'collisions':sum(bool(x['collision']) for x in valid.values()),
               'rollout_uids':[x['rollout_uid'] for x in valid.values()],
               'terminal_summary':dict(Counter(x['outcome'] for x in valid.values()))}
            details.append(e);truth[(e['state_uid'],e['candidate'])]=e
            if absent:missing.append({'state_uid':e['state_uid'],'candidate':e['candidate'],'seeds':absent})
    dump('seed_evidence.json',{'rows':details,'missing':missing})
    if missing:raise RuntimeError(f'Confirmation incomplete: {len(missing)} tuples')
    rows=[]
    for p in manifest['states']:
        sid=p['state_uid'];ee=[truth[(sid,i)] for i in range(17)];b0=truth[(sid,None)]
        idx=p['critic_index'];chosen=ee[idx];old=ee[p['old_critic_index']]
        oracle=(True if any(e['robust'] is True for e in ee) else None if any(e['robust'] is None for e in ee) else False)
        exact=all(e['Q16'] is not None for e in ee)
        best=int(np.argmax([e['Q_lower'] for e in ee])) if exact else None
        rows.append({'scenario':p['scenario'],'state_uid':sid,'B0_robust':b0['robust'],'oracle_robust':oracle,
                     'old_critic_robust':old['robust'],'old_selected_index':p['old_critic_index'],
                     'old_miss':oracle is True and old['robust'] is False,
                     'old_rescue':b0['robust'] is False and old['robust'] is True,
                     'old_break':b0['robust'] is True and old['robust'] is False,
                     'old_exploitation':p['old_scores'][p['old_critic_index']]>=.9375 and old['Q_upper']<.5,
                     'critic_robust':chosen['robust'],'selected_index':idx,'oracle_index':best,
                     'exact_top1_agreement':idx==best if best is not None else None,
                     'Q_tie_top1_agreement':chosen['Q16']==ee[best]['Q16'] if best is not None else None,
                     'Q_regret_lower':max(0.,max(e['Q_lower'] for e in ee)-chosen['Q_upper']),
                     'Q_regret_upper':max(e['Q_upper'] for e in ee)-chosen['Q_lower'],
                     'rescue':b0['robust'] is False and chosen['robust'] is True,
                     'break':b0['robust'] is True and chosen['robust'] is False,
                     'miss':oracle is True and chosen['robust'] is False,
                     'exploitation':p['scores'][idx]>=.9375 and chosen['Q_upper']<.5,
                     'selected_Q16':chosen['Q16'],'selected_score':p['scores'][idx],
                     'selected_collisions':chosen['collisions'],'selected_numerical_seeds':chosen['numerical_seeds']})
    summary={}
    for sc in a.SCENARIOS:
        rr=[x for x in rows if x['scenario']==sc]
        summary[sc]={'states':len(rr),**{key:sum(r[key] is True for r in rr) for key in
                      ['B0_robust','oracle_robust','critic_robust','old_critic_robust','rescue','break','miss','exploitation',
                       'old_rescue','old_break','old_miss','old_exploitation','exact_top1_agreement','Q_tie_top1_agreement']},
                     'B0_nonrobust':sum(r['B0_robust'] is False for r in rr),
                     'B0_unresolved':sum(r['B0_robust'] is None for r in rr),
                     'critic_unresolved':sum(r['critic_robust'] is None for r in rr),
                     'oracle_unresolved':sum(r['oracle_robust'] is None for r in rr),
                     'mean_Q_regret_lower':float(np.mean([r['Q_regret_lower'] for r in rr])),
                     'mean_Q_regret_upper':float(np.mean([r['Q_regret_upper'] for r in rr])),
                     'selected_collisions':sum(r['selected_collisions'] for r in rr),
                     'selected_numerical_seeds':sum(len(r['selected_numerical_seeds']) for r in rr)}
    dump('results.json',{'summary':summary,'states':rows,'no_adaptation_after_confirmation':True})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    print(json.dumps(freeze() if args.stage=='freeze' else collect() if args.stage=='collect' else run(args.shard,args.shards),indent=2))
