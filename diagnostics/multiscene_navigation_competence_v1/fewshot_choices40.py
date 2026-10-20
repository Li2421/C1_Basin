"""Pilot-only choices frozen before reading each family's evaluation labels."""
from common import *
from selector_data40 import MENU
from shared_rollout_db.src.rollout_db import connect
from scipy.special import betaln,logsumexp
from functools import lru_cache

@lru_cache(None)
def beta_gt(s,n,t,m):
    a=s+1;b=n-s+1;c=t+1;d=m-t+1;i=np.arange(a)
    return float(np.exp(logsumexp(betaln(c+i,b+d)-np.log(b+i)-betaln(1+i,b)-betaln(c,d))))

def freeze(family):
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');stage=f'pilot_full40_{family}';assert (HERE/f'{stage}_POSTFLIGHT.json').exists()
    cases=read(HERE/f'{stage}_CASES.json');names=read(HERE/f'MULTIAGENT_CAPABILITY_{family}.json')['eligible'];baseline=MENU.index('S13');rng=np.random.default_rng(cfg['fewshot_seed']+'ABCDEFG'.index(family));output={};provenance=[]
    with connect(True) as db:
        for name in names:
            pools=[];first4={}
            for arm in MENU:
                selected=sorted([c for c in cases if c['controller_name']==name and c['arm']==arm],key=lambda c:c['state']['index']);assert len(selected)==16
                rows=[dict(db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()) for c in selected];assert all(not r['conflict_quarantined'] for r in rows)
                pools.append(np.asarray([r['success'] for r in rows if not r['numerical_failure']],int));first4[arm]=dict(success=sum(r['success'] for r in rows[:4]),unknown=sum(r['numerical_failure'] for r in rows[:4]))
                provenance.extend(dict(rollout_uid=c['rollout_uid'],raw_record_hash=r['raw_record_hash'],controller=name,arm=arm,state_index=c['state']['index']) for c,r in zip(selected,rows))
            choices={}
            for k in cfg['ks']:
                if min(map(len,pools))<k:
                    choices[str(k)]=dict(status='GAP',reason='Insufficient known pilot labels per arm; unknown retained without replacement',known_counts=list(map(len,pools)));continue
                wins=np.asarray([[int(pool[rng.permutation(len(pool))[:k]].sum()) for pool in pools] for _ in range(cfg['fewshot_repeats'])]);emp=wins.argmax(1);emp[(wins==wins.max(1)[:,None]).sum(1)>1]=baseline;safe=np.full(cfg['fewshot_repeats'],baseline,int)
                for b in range(cfg['fewshot_repeats']):
                    eligible=[a for a in range(7) if a!=baseline and beta_gt(int(wins[b,a]),k,int(wins[b,baseline]),k)>=cfg['fewshot_bayesian_threshold']]
                    if eligible:safe[b]=max(eligible,key=lambda a:(wins[b,a],-a))
                choices[str(k)]=dict(status='FROZEN',known_labels_per_arm=k,repeats=cfg['fewshot_repeats'],empirical_choice_counts=np.bincount(emp,minlength=7),bayesian_choice_counts=np.bincount(safe,minlength=7),empirical_choices=emp,bayesian_choices=safe,query_cost=f'7*{k} known labels; unknown queried labels are not free')
            simplified='right_hand' if first4['right_hand']['success']>first4['S13']['success'] else 'S13'
            output[name]=dict(curves=choices,simplified_four_pilot_choice=simplified,simplified_counts={arm:first4[arm] for arm in ['S13','right_hand']},simplified_unknown_sensitivity=bool(first4['S13']['unknown'] or first4['right_hand']['unknown']))
    p=HERE/'selector_choices40'/f'{family}_fewshot.json';write(p,dict(utc=utc(),family=family,choices=output,menu=MENU,source='disjoint original pilot states only',evaluation_labels_opened=False,provenance=provenance))
    seal(str(p.with_suffix('.receipt.json').relative_to(HERE)),[p,HERE/'fewshot_choices40.py',HERE/f'{stage}_POSTFLIGHT.json',HERE/'SELECTOR40_CONFIG_FROZEN.receipt.json'],before_target_evaluation_label_analysis=True,evaluation_labels_opened=False)

if __name__=='__main__':freeze(sys.argv[1])
