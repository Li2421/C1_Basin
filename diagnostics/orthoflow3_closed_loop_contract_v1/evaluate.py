"""Matched repair validation, using existing physical evidence wherever exact."""
import argparse,json
import numpy as np
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev
OUT=repair.OUT/'development'

def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def freeze():
    if (OUT/'proposals.json').exists():raise RuntimeError('Already frozen')
    model=ev.load_models();by={r['state_uid']:r for r in b.rows()};pp=[]
    old=a.load(repair.OLD_OUT/'development/proposals.json')['states']
    for sc in ('four_way_intersection','ring_exchange'):
        rows=sorted([p for p in old if p['scenario']==sc],key=lambda p:a.digest('contract-dev|'+p['state_uid']))[:6]
        for p in rows:
            r=by[p['state_uid']];pp.append({'state_uid':r['state_uid'],'scenario':sc,**ev.proposals(r,model),
                                          'old_selected_index':p['selected_index'],'old_proposal_hash':p['proposal_hash']})
    dump('proposals.json',{'states':pp,'frozen_before_outcomes':True,'no_witness_labels_trained':True,
                          'generator':a.load(b.OUT/'generator_frozen.json'),'critic':a.load(b.OUT/'critic_frozen.json')})
    return {'states':len(pp)}

def collect():
    truth={};allrows=[]
    for sc,r in ev.runtimes():
        for j in ev.jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(missing)
            s=sum(x['success'] for x in valid.values());f=len(valid)-s
            e={'state_uid':j['state']['uid'],'scenario':sc,'index':j['index'],'successes':s,'failures':f,
               'numerical':len(num),'Q_lower':s/16,'Q_upper':(16-f)/16,
               'robust':True if s>=15 else False if f>=2 else None,'collision':sum(x['collision'] for x in valid.values())}
            truth[e['state_uid'],e['index']]=e;allrows.append(e)
    old={r['state_uid']:r for r in a.load(repair.OLD_OUT/'development/results.json')['states']}
    rows=[]
    for p in a.load(OUT/'proposals.json')['states']:
        sid=p['state_uid'];ee=[truth[sid,i] for i in range(17)];s=ee[p['selected_index']];z=truth[sid,None]
        rows.append({'state_uid':sid,'scenario':p['scenario'],'oracle':any(e['robust'] is True for e in ee),
                     'selected':s['robust'],'old_selected':old[sid]['selected_robust'],'old_oracle':old[sid]['oracle_robust'],
                     'rescue':z['robust'] is False and s['robust'] is True,'break':z['robust'] is True and s['robust'] is False,
                     'exploitation':p['scores'][p['selected_index']]>=.9375 and s['Q_upper']<.5,
                     'regret_lower':max(0.,max(e['Q_lower'] for e in ee)-s['Q_upper']),
                     'regret_upper':max(e['Q_upper'] for e in ee)-s['Q_lower']})
    summary={sc:{k:sum(r[k] is True for r in rows if r['scenario']==sc) for k in ['oracle','selected','old_selected','old_oracle','rescue','break','exploitation']} for sc in ('four_way_intersection','ring_exchange')}
    dump('results.json',{'summary':summary,'states':rows,'evidence':allrows});return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    repair.install();ev.OUT=OUT
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
