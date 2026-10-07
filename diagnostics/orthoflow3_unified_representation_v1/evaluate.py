"""Frozen matched R_old/R_unified proposal comparisons on train/dev only."""
import argparse,json
from . import build,train,analyze
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev
OUT=build.OUT/'development'

def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def freeze():
    if (OUT/'proposals.json').exists():raise RuntimeError('Already frozen')
    assert a.load(build.OUT/'structural_tests.json')['passed']
    newmodel=train.load_models();repair.install();oldmodel=ev.load_models()
    cohort=a.load(repair.OUT/'development/proposals.json')['states']
    previous=a.load(repair.OLD_OUT/'development/proposals.json')['states']
    cohort+=sorted([r for r in previous if r['scenario']=='double_bottleneck'],key=lambda r:a.digest('unified-dev|'+r['state_uid']))[:6]
    newrows={r['state_uid']:r for r in build.rows()};oldrows={r['state_uid']:r for r in b.rows()};pp=[]
    for c in cohort:
        sid=c['state_uid'];p=analyze.proposals(newrows[sid],newmodel);old=ev.proposals(oldrows[sid],oldmodel)
        pp.append({'state_uid':sid,'scenario':c['scenario'],**p,'old':old})
    dump('proposals.json',{'states':pp,'frozen_before_outcomes':True,'K_stochastic':16,
        'new_generator':a.load(build.OUT/'generator_frozen.json'),'new_critic':a.load(build.OUT/'critic_frozen.json'),
        'old_generator':a.load(repair.OUT/'generator_frozen.json'),'old_critic':a.load(repair.OUT/'critic_frozen.json'),
        'population':'fixed train/dev validation subset; no test states'})
    return {'states':len(pp)}

def jobs(r):
    by={p['state_uid']:p for p in a.load(OUT/'proposals.json')['states']}
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(a.SEEDS),'index':i}
            for s in r.states for i,eta in enumerate(by[s['uid']]['etas']+by[s['uid']]['old']['etas']+[[0,0,0]])]

def collect():
    truth={};evidence=[]
    for sc,r in ev.runtimes():
        for j in jobs(r):
            valid,num,missing=a.evidence(r,j)
            if missing:raise RuntimeError(missing)
            s=sum(x['success'] for x in valid.values());f=len(valid)-s
            e={'state_uid':j['state']['uid'],'scenario':sc,'index':j['index'],'successes':s,'failures':f,'numerical':len(num),
               'Q16':s/16 if len(valid)==16 else None,'Q_lower':s/16,'Q_upper':(16-f)/16,
               'robust':True if s>=15 else False if f>=2 else None,'collisions':sum(x['collision'] for x in valid.values()),
               'rollout_uids':[x['rollout_uid'] for x in valid.values()]}
            truth[e['state_uid'],e['index']]=e;evidence.append(e)
    states=[]
    for p in a.load(OUT/'proposals.json')['states']:
        sid=p['state_uid'];z=truth[sid,34]
        for mode,offset,proposal in [('unified',0,p),('old',17,p['old'])]:
            ee=[truth[sid,i+offset] for i in range(17)];s=ee[proposal['selected_index']];correct=pairs=0
            for j in range(17):
                for k in range(j):
                    if ee[j]['Q16'] is None or ee[k]['Q16'] is None:continue
                    d=ee[j]['Q16']-ee[k]['Q16']
                    if abs(d)<1/16:continue
                    pairs+=1;correct+=(proposal['logits'][j]-proposal['logits'][k])*d>0
            states.append({'scenario':p['scenario'],'state_uid':sid,'representation':mode,'oracle':any(e['robust'] is True for e in ee),
                'selected':s['robust'],'B0':z['robust'],'robust_hit_count':sum(e['robust'] is True for e in ee),
                'rescue':z['robust'] is False and s['robust'] is True,'break':z['robust'] is True and s['robust'] is False,
                'exploitation':proposal['scores'][proposal['selected_index']]>=.9375 and s['Q_upper']<.5,
                'regret_lower':max(0.,max(e['Q_lower'] for e in ee)-s['Q_upper']),
                'regret_upper':max(e['Q_upper'] for e in ee)-s['Q_lower'],'pairwise_correct':int(correct),'pairwise_pairs':pairs})
    summary={}
    for sc in a.SCENARIOS:
        summary[sc]={}
        for mode in ('old','unified'):
            rr=[s for s in states if s['scenario']==sc and s['representation']==mode]
            summary[sc][mode]={**{k:sum(s[k] is True for s in rr) for k in ['oracle','selected','B0','rescue','break','exploitation']},
                'states':len(rr),'mean_hits':sum(s['robust_hit_count'] for s in rr)/len(rr),
                'pairwise_accuracy':sum(s['pairwise_correct'] for s in rr)/max(1,sum(s['pairwise_pairs'] for s in rr)),
                'mean_regret_lower':sum(s['regret_lower'] for s in rr)/len(rr),'mean_regret_upper':sum(s['regret_upper'] for s in rr)/len(rr)}
    dump('results.json',{'summary':summary,'states':states,'evidence':evidence});return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cache','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);x=p.parse_args()
    ev.OUT=OUT;ev.jobs=jobs
    print(json.dumps({'freeze':freeze,'cache':ev.cache,'run':lambda:ev.run(x.shard,x.shards),'collect':collect}[x.stage](),indent=2))
