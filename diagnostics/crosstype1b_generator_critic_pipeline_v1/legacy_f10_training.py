"""Reuse exact old fixed-eta10 labels for the standard training disaster metric."""
from common import *
from statistics_pipeline import sample_indices,point_indices,best


def main():
    old=D/'crosstype1_learned_zoo_selector_v1'
    with np.load(HERE/'TRAINING_BUNDLE.npz') as z:data={k:z[k] for k in ('names','types','state_uids','labels','uids')}
    with np.load(HERE/'DEVELOPMENT_CHOICES_V2B.npz') as z:fixed={k:z[k] for k in z.files if k not in ('names','types')}
    lower=(data['labels'][...,0]==1).astype(float)
    fixed['state_in_sample_oracle']=best(lower)
    fixed['controller_oracle']=np.broadcast_to(best(lower.mean(1))[:,None],(62,48))
    reference=np.full((62,48),-2,np.int8);uids=np.full((62,48),'',dtype='U96');sources=[]
    events={k:np.zeros((62,48),float) for k in ('collision','timeout','unknown')}
    with rollout_db.connect(True) as db:
        for cohort in ('development','locked'):
            for split,offset in (('tuning',0),('evaluation',16)):
                path=old/f'{cohort}_{split}_CASES.json';sources.append(path)
                for case in read(path):
                    if case['arm']!='F10' or case['seed']!=0:continue
                    i=list(data['names']).index(case['base_controller_name']);s=offset+case['state_index']
                    assert case['state_uid']==data['state_uids'][s]
                    row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(case['rollout_uid'],)).fetchone();assert row is not None
                    reference[i,s]=-1 if row['numerical_failure'] else row['success'];uids[i,s]=case['rollout_uid']
                    for e in events:events[e][i,s]=row['numerical_failure'] if e=='unknown' else row[e]
    assert not np.any(reference==-2)
    lo=reference==1;hi=reference!=0;types=data['types'];groups={}
    for group in [*sorted(set(types)),'combined']:
        mask=types==group if group!='combined' else np.ones(62,bool);tt=types[mask];a=lo[mask];b=hi[mask];n,ns=a.shape;reports={}
        for te in ((True,False) if group=='combined' else (True,)):
            cc,ss,w=point_indices(tt,ns,te);point_weight=w[0].copy();low=float(np.sum(a.mean(1)*point_weight));high=float(np.sum(b.mean(1)*point_weight))
            rng=np.random.default_rng(202610180901);samples=[]
            for start in range(0,5000,32):
                cc,ss,w=sample_indices(tt,ns,rng,min(32,5000-start),te)
                samples.append(np.stack([np.sum(x[cc[...,None],ss].mean(-1)*w,axis=1) for x in (a,b)],axis=-1))
            draws=np.concatenate(samples);disasters={}
            for method,choice in fixed.items():
                y=np.take_along_axis(data['labels'][...,0],choice[...,None],axis=2)[...,0]
                selected_uid=np.take_along_axis(data['uids'][...,0],choice[...,None],axis=2)[...,0]
                same=selected_uid==uids
                possible=np.where(same,0,hi.astype(float)-(y==1)).mean(1);certain=np.where(same,0,lo.astype(float)-(y!=0)).mean(1)
                disasters[method]=dict(possible=int(np.sum(possible[mask]>=.1-1e-12)),certain=int(np.sum(certain[mask]>=.1-1e-12)))
            safety=data['labels'][:,:,10,0];oracle=(data['labels'][...,0]==1).max(2)
            harm=((safety!=0).astype(float)-lo).mean(1);certain=((safety==1).astype(float)-hi).mean(1)
            regret=(oracle.astype(float)-lo).mean(1)
            reports['type_equal' if te else 'controller_equal']=dict(success_lower_pct=100*low,success_upper_pct=100*high,
                success_CI95_pct=[100*float(np.quantile(draws[:,0],.025)),100*float(np.quantile(draws[:,1],.975))],
                **{e+'_pct':100*float(np.sum(v[mask].mean(1)*point_weight)) for e,v in events.items()},
                state_crossfit_regret_pp=None,max_controller_crossfit_regret_pp=None,
                state_in_sample_regret_pp=100*float(np.sum(regret[mask]*point_weight)),
                max_controller_in_sample_regret_pp=100*float(regret[mask].max()),
                harm_possible=int(np.sum(harm[mask]>=.1-1e-12)),harm_certain=int(np.sum(certain[mask]>=.1-1e-12)),
                disasters_vs_F10=0,
                disaster_count_by_method=disasters,replicates=5000,evidence_level='DEVELOPMENT_HISTORICAL_EXACT_REUSE_COMMON_SEED0')
        groups[str(group)]=reports
    write(HERE/'LEGACY_F10_TRAINING_REFERENCE.json',dict(utc=utc(),summaries=groups,new_rollouts=0,
        reference='Full-episode historical eta10; outside the thirteen selectable candidates, excluded from every oracle',
        known=int((reference>=0).sum()),unknown=int((reference<0).sum())))
    seal('LEGACY_F10_TRAINING_REFERENCE.receipt.json',[HERE/'LEGACY_F10_TRAINING_REFERENCE.json',HERE/'legacy_f10_training.py',
        HERE/'TRAINING_BUNDLE.receipt.json',HERE/'DEVELOPMENT_CHOICES_V2B.receipt.json',*sources],new_rollouts=0)


if __name__=='__main__':main()
