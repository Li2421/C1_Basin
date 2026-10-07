"""Does crossed supervision repair causal prediction, not merely global priors?"""
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .phase_factorial_train import OUT,SOURCE,ARMS,KINDS,SEEDS,read,write,csvwrite,metrics,causal,folds
from shared_rollout_db.src.rollout_db import connect,canonical


def seed_effect_audit(d,va):
    pairs=read(SOURCE/'pairs.json');native=read(SOURCE/'protocol.json')['profiles'];extra=read(OUT/'protocol.json')['profiles'];rows=[]
    with connect(True) as db:
      for prog,parent in ((0,0),(1,1)):
        for i in va:
          p=pairs[i];records=[]
          for c in (native[parent],extra[prog]):
            rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
            records.append([rr[canonical({'future_index':k})] for k in range(16)])
          valid=[(a,b) for a,b in zip(*records) if not a['numerical_failure'] and not b['numerical_failure']]
          a=np.array([r[0]['success'] for r in valid]);b=np.array([r[1]['success'] for r in valid]);r=int(((a==0)&(b==1)).sum());br=int(((a==1)&(b==0)).sum())
          rows.append(dict(program=prog,pair_index=int(i),state_uid=p['state_uid'],eta_index=p['eta_index'],valid_paired_seeds=len(valid),Q_change=float(np.mean(b-a)),seed_rescue=r,seed_break=br,p=float(binomtest(r,r+br,.5).pvalue) if r+br else 1.))
    pp=np.array([r['p'] for r in rows]);order=np.argsort(pp);adj=np.empty(len(rows));adj[order]=np.minimum(1,np.maximum.accumulate(pp[order]*(len(rows)-np.arange(len(rows)))))
    for r,a in zip(rows,adj):r['Holm_p']=float(a)
    csvwrite(OUT/'source_VAL_causal_seed_audit.csv',rows)
    return dict(cells=len(rows),Holm_significant=int((adj<.05).sum()),large_Q_change=int(sum(abs(r['Q_change'])>=.25 for r in rows)),mean_abs_Q_change=float(np.mean([abs(r['Q_change']) for r in rows])))


def main():
    sel=read(OUT/'source_selection.json')['selection'];d=dict(np.load(OUT/'dataset.npz'));tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='validation');q=d['success']/(d['success']+d['failure']);qq=q[:,va].reshape(14,16,2);good=(d['success'][:,va]>=15).reshape(14,16,2);bad=(d['failure'][:,va]>=2).reshape(14,16,2)
    effect=seed_effect_audit(d,va);rows=[];comp=[];pred={}
    rng=np.random.default_rng(202610048140);boot=rng.integers(16,size=(10000,16));ix=np.arange(16)
    for arm in ARMS:
      for kind in KINDS:
       step=sel[arm][kind]['step']
       for fold in range(3):
        for seed in SEEDS:
            path=OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'
            z=np.load(path/'validation_predictions.npz')[f'step{step}'];pred[(arm,kind,fold,seed)]=z
            for name,zz in (('correct',z),('wrong_goal_parent',np.concatenate([z[:12],z[[0,1]]])),('all_state_context_shuffle',z[:,np.roll(np.arange(32).reshape(-1,2),-1,axis=0).ravel()])):
                mm=metrics(zz,d,va,[12,13]);rows.append(dict(arm=arm,kind=kind,fold=fold,seed=seed,step=step,condition=name,constituents_seen=fold!=2,**mm,**causal(zz,d,va)))
    # Privileged known-controller/eta prior: identifies how much gain can be
    # explained without adapting to individual physicalstates. Not zero-shot.
    prior=q[:,tr].reshape(14,-1,2).mean(1);pz=np.log(np.clip(prior,1e-5,1-1e-5)/(1-np.clip(prior,1e-5,1-1e-5)))
    pz=np.broadcast_to(pz[:,None,:],(14,16,2)).reshape(14,32)
    write(OUT/'known_program_eta_prior.json',dict(phase_metrics=metrics(pz,d,va,[12,13]),phase_causal=causal(pz,d,va),interpretation='PrivilegedprogramID+TRAINetaaverage, no stateadaptation; diagnosticnotunseencontroller model.'))
    for fold in range(3):
      for seed in SEEDS:
        za=pred[('crossed_phase','H20_goal',fold,seed)].reshape(14,16,2)
        refs=dict(native_repeated=pred[('native_repeated','H20_goal',fold,seed)],eta_only=pred[('crossed_phase','eta_only',fold,seed)],H20_only=pred[('crossed_phase','H20_only',fold,seed)],known_program_eta_prior=pz,wrong_goal_parent=np.concatenate([za[:12],za[[0,1]]]).reshape(14,32))
        for name,zb in refs.items():
            zb=zb.reshape(14,16,2);aa=za[[12,13]];bb=zb[[12,13]];true=qq[[12,13]];ga=good[[12,13]];ba=bad[[12,13]]
            la=(true*np.logaddexp(0,-aa)+(1-true)*np.logaddexp(0,aa)).mean(2);lb=(true*np.logaddexp(0,-bb)+(1-true)*np.logaddexp(0,bb)).mean(2)
            ca=aa.argmax(2);cb=bb.argmax(2);gg_a=ga[np.arange(2)[:,None],ix,ca];gg_b=ga[np.arange(2)[:,None],ix,cb];bad_a=ba[np.arange(2)[:,None],ix,ca];bad_b=ba[np.arange(2)[:,None],ix,cb]
            dl=(la-lb).mean(0);dq=(true[np.arange(2)[:,None],ix,ca]-true[np.arange(2)[:,None],ix,cb]).mean(0)
            r=int((gg_a&bad_b).sum());br=int((bad_a&gg_b).sum())
            comp.append(dict(fold=fold,seed=seed,constituents_seen=fold!=2,reference=name,NLL_delta=float(dl.mean()),NLL_CI=np.quantile(dl[boot].mean(1),[.025,.975]).tolist(),selected_Q_delta=float(dq.mean()),selected_Q_CI=np.quantile(dq[boot].mean(1),[.025,.975]).tolist(),rescue=r,breaks=br,net_rescue=r-br,
                paired_state_note='Two controllerprograms per physicalfamily; no independence assumption between programs; bootstrap resamples16families.'))
    csvwrite(OUT/'selected_source_phase_metrics.csv',rows);csvwrite(OUT/'selected_source_phase_comparisons.csv',comp)
    write(OUT/'source_adjudication.json',dict(selection=sel,causal_label_audit=effect,
        phase_results=[r for r in rows if r['kind']=='H20_goal' and r['condition']=='correct'],
        comparisons=comp,target_labels_used=False,new_target_opened=False,
        scope='Source-controller CV plus independent sourcefamilies, not finalunseencontroller confirmation. Fold2 has no phaseintervention in TRAIN by isolationrule.'))
    print(dict(causal_label_audit=effect,selection=sel))


if __name__=='__main__':main()
