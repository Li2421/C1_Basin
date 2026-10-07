"""Small outcome-blind shared-eta panel, diagnostic evidence only."""
import argparse,json
from collections import Counter
import numpy as np
from scipy.stats import qmc
from .audit import OUT as BASE,SCENES,load,dump,Frozen,folders,ev as existing
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,confirmation_a as ca,evaluate_b as runner
from new_benchmark_common.safety_eta3 import DatabaseSink,request,preflight
from shared_rollout_db.src.rollout_db import uid,connect,canonical
OUT=BASE/'probe'
def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def freeze():
    assert not (OUT/'protocol.json').exists()
    original=load(ca.OUT/'state_manifest.json')['states'];states=[];preds=[]
    domain=load(a.ROOT/'diagnostics/ring_exchange_safety_eta3/frozen_eta_design.json')['domain']
    lo,hi=np.array(domain['lower']),np.array(domain['upper']);eta=(lo+qmc.Sobol(3,scramble=True,seed=730031).random_base2(3)*(hi-lo)).tolist()
    eta+=[[0.,0.,0.]] # explicit reference, not added to deployed generator proposals
    for fold in ('ring','four'):
        ps,x=existing.inputs(fold);selected=sorted(range(len(ps)),key=lambda i:a.digest('interaction-panel-v1|'+ps[i]['state_uid']))[:8]
        by={s['state_uid']:s for s in original}
        for i in selected:
            p=ps[i];s=by[p['state_uid']];physical=s['physical'];ch=a.digest(physical)
            states.append({'uid':p['state_uid'],'alias':'INTERACTION_'+p['state_uid'],'index':i,'scenario':SCENES[fold],
                 'physical':physical,'metadata':s['metadata'],'content_hash':ch,'controller_uid':p['controller_uid']})
        xx={k:v[selected] for k,v in x.items()};ee=np.tile(eta,(8,1,1))
        scores={name:Frozen(folder).score(Frozen(folder).encode(xx),ee).tolist() for name,folder in folders(fold).items()}
        preds.append({'fold':fold,'state_uids':[ps[i]['state_uid'] for i in selected],'scores':scores})
    save('states.json',states);save('proposals.json',{'states':[{'state_uid':s['uid'],'scenario':s['scenario'],'etas':eta} for s in states]})
    save('predictions_frozen.json',preds)
    save('protocol.json',{'states_per_scene':8,'scenes':['ring','four'],'shared_eta':eta,'domain':domain,
       'selection':'hash state identity only; 8 fixed Sobol eta seed730031 + diagnostic eta0; independent of critic and outcomes',
       'seeds':list(a.SEEDS),'full_Q16':True,'requested':2304,'model_changes':False,'new_training_data':False,
       'purpose':'existing VAL shared-eta full-Q evidence has Ring1/Four30 quadruples, zero strong interactions; fill diagnostic cross-state matrix',
       'population':'previous frozen independent confirmation, now diagnostic only; no adaptation',
       'extension_rule':'no adaptive eta search; report failure to find reversals honestly'})

def runtimes():
    a.bd.OUT=OUT/'registry';a.bd.WORK=OUT
    for sc in ('four_way_intersection','ring_exchange'):
        r=ca.NewRuntime(sc,[],parent=True);r.states=[s for s in load(OUT/'states.json') if s['scenario']==sc]
        assert all(s['controller_uid']==r.controllers['orthoflow3']['uid'] for s in r.states),'physical protocol mismatch'
        r.output=OUT/sc;r.output.mkdir(parents=True,exist_ok=True);path=OUT/'execution'
        r.experiment_uid=uid('exp',{'path':str(path)})
        with connect() as con:
            con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                (r.experiment_uid,'state_eta_interaction_diagnostic',str(path),a.sha(OUT/'protocol.json'),a.sha(__file__),canonical({'training':False,'diagnostic':True})))
            assert con.execute('SELECT 1 FROM experiment WHERE experiment_uid=?',(r.experiment_uid,)).fetchone();con.commit()
        yield sc,r
def jobs(r):
    etas=load(OUT/'protocol.json')['shared_eta']
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i} for s in r.states for i,eta in enumerate(etas)]
def cache():
    runner.OUT=OUT;runner.runtimes=runtimes;runner.jobs=jobs
    return runner.cache()
def run(shard,shards):
    c=Counter();idx=0
    for sc,r in runtimes():
        chosen=[]
        for j in jobs(r):
            if idx%shards==shard:chosen.append(j)
            idx+=1
        if not chosen:continue
        name=f'cache/{sc}_{shard}of{shards}'
        save(name+'.json',{'requests':[request(r,j['state'],j['eta'],'orthoflow3',a.SEEDS) for j in chosen]})
        save(name+'_preflight.json',preflight(OUT/(name+'.json')))
        sink=DatabaseSink(r,f'interaction_{shard}of{shards}')
        try:
            for j in chosen:
                found,num,missing=a.evidence(r,j);c['requested']+=16;c['reused']+=len(found);c['existing_numerical']+=len(num)
                for seed in missing:
                    for attempt in range(4):
                        row=r.rollout(j['state'],np.asarray(j['eta']),seed,'orthoflow3')
                        row.update(probe_eta_index=j['index'],execution_attempt=attempt,phase='state_eta_interaction_diagnostic',training_label=False)
                        sink.insert(row);c['physical_attempts']+=1
                        if not row['numerical_failure']:break
                    c['completed_seeds']+=1;c['numerical_unresolved']+=bool(row['numerical_failure'])
                print(json.dumps({'shard':shard,**c}),flush=True)
        finally:sink.finalize()
    save(f'run_{shard}of{shards}.json',dict(c));return dict(c)
def collect():
    out=[]
    for sc,r in runtimes():
        for j in jobs(r):
            found,num,missing=a.evidence(r,j);assert not missing
            s=sum(x['success'] for x in found.values());f=len(found)-s
            out.append({'scenario':sc,'state_uid':j['state']['uid'],'eta_index':j['index'],
                'q':s/16 if len(found)==16 else None,'lower':s/16,'upper':(16-f)/16,
                'b15':True if s>=15 else False if f>=2 else None,'numeric':list(num),
                'seed_success':{str(k):v['success'] for k,v in found.items()},
                'collisions':sum(x['collision'] for x in found.values()),'rollout_uids':[x['rollout_uid'] for x in found.values()]})
    save('evidence.json',out);return {'tuples':len(out),'numeric':sum(len(x['numeric']) for x in out),'collisions':sum(x['collisions'] for x in out)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    print(json.dumps({'freeze':freeze,'cache':cache,'run':lambda:run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
