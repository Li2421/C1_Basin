"""Score the two frozen physical rules on state-disjoint splits."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TRANSFER=Path('/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/state_eta_chain_features.csv')

def readcsv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def average_precision(y,score):
    y=np.asarray(y,dtype=bool);score=np.asarray(score,float)
    if not y.any():return None
    order=np.argsort(-score,kind='stable');yy=y[order];ss=score[order]
    cuts=np.r_[np.flatnonzero(np.diff(ss)!=0)+1,len(ss)]
    ap=0.;start=0;total=int(y.sum());caught=0
    for stop in cuts:
        npos=int(yy[start:stop].sum());caught+=npos
        ap+=(npos/total)*(caught/stop);start=stop
    return float(ap)

def metrics(rows,flag,score):
    y=np.asarray([r['target'] for r in rows],bool);p=np.asarray([r[flag] for r in rows],bool)
    tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p));tn=int(np.sum(~y&~p))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        specificity=tn/max(tn+fp,1),balanced_accuracy=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),
        prevalence=float(y.mean()),average_precision=average_precision(y,[r[score] for r in rows]))

def boot(rows,flag,score,seed):
    rng=np.random.default_rng(seed);states=sorted(set(r['uid'] for r in rows));by={s:[r for r in rows if r['uid']==s] for s in states}
    values=[]
    for _ in range(2000):
        sampled=[r for s in rng.choice(states,len(states),replace=True) for r in by[s]]
        if not any(r['target'] for r in sampled):continue
        m=metrics(sampled,flag,score);values.append((m['precision'],m['recall'],m['balanced_accuracy'],m['average_precision']))
    a=np.asarray(values)
    return dict(n_valid=len(a),precision_95=np.quantile(a[:,0],[.025,.975]).tolist(),
        recall_95=np.quantile(a[:,1],[.025,.975]).tolist(),balanced_accuracy_95=np.quantile(a[:,2],[.025,.975]).tolist(),
        average_precision_95=np.quantile(a[:,3],[.025,.975]).tolist()) if len(a) else dict(n_valid=0)

def main():
    features={(r['state_uid'],int(r['eta_index']),r['chain']):r for r in readcsv(TRANSFER) if r['phase']=='initial'}
    allrows=[]
    for scene,file,old,new,target in [('toy_give_way','toy_loss_analysis.csv','FT','FF','loss'),
                                       ('ring_exchange','ring_gain_analysis.csv','TT','TF','gain')]:
        for r in readcsv(HERE/file):
            if r['b15_outcome']=='unknown':continue
            a=features[(r['uid'],int(r['eta_index']),old)];b=features[(r['uid'],int(r['eta_index']),new)]
            d_goal=float(b['goal_signed_nominal_motion'])-float(a['goal_signed_nominal_motion'])
            d_tan=float(b['tangential_exec_motion_norm'])-float(a['tangential_exec_motion_norm']) if scene=='ring_exchange' else None
            valid=bool(r['first_action_l2'])
            x=dict(scene=scene,uid=r['uid'],eta_index=int(r['eta_index']),split=r['split'],
                b15_outcome=r['b15_outcome'],target=int(r['b15_outcome']==target),
                old_successes=int(r['old_successes']),new_successes=int(r['new_successes']),
                d_goal=d_goal,d_tan=d_tan,seed0_valid=int(valid),
                initial_goal_error_max=float(r['initial_goal_error_max']),
                first_action_l2=float(r['first_action_l2']) if valid else None)
            if scene=='toy_give_way':
                x['early_progress_difference']=float(r['H20_worst_progress_delta']) if valid else None
                x['condition0']=int(d_goal<-.14)
                x['condition1']=int(valid and x['early_progress_difference']<-.12)
                x['condition1_available']=int(valid)
                x['condition0_score']=-d_goal
                x['condition1_score']=-x['early_progress_difference'] if valid else None
                x['baseline_goal_distance']=int(x['initial_goal_error_max']>1.743316)
                x['baseline_action_magnitude']=int(valid and x['first_action_l2']>0.233135)
            else:
                x['early_progress_difference']=float(r['H20_worst_progress_delta']) if valid else None
                x['condition0']=int(d_tan>0)
                x['condition1']=int(valid and d_tan>0 and x['early_progress_difference']<0)
                x['condition1_available']=int(valid)
                x['condition0_score']=d_tan
                x['condition1_score']=d_tan if valid and x['early_progress_difference']<0 else -abs(d_tan)
                x['baseline_goal_distance']=int(x['initial_goal_error_max']>4.688165)
                x['baseline_action_magnitude']=int(valid and x['first_action_l2']>0.070644)
            x['baseline_negative']=0
            x['baseline_majority']=0
            allrows.append(x)
    fields=list(dict.fromkeys(k for r in allrows for k in r))
    with (HERE/'heldout_prediction.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(allrows)
    summary={}
    for scene in ('toy_give_way','ring_exchange'):
        for split in ('discovery','validation','heldout_test'):
            rows=[r for r in allrows if r['scene']==scene and r['split']==split]
            key=f'{scene}:{split}';summary[key]={'n':len(rows),'outcomes':dict(Counter(r['b15_outcome'] for r in rows))}
            for flag,score in [('condition0','condition0_score'),('baseline_goal_distance','initial_goal_error_max'),
                               ('baseline_action_magnitude','first_action_l2'),('baseline_negative','baseline_negative'),
                               ('baseline_majority','baseline_majority')]:
                subset=[r for r in rows if r[score] is not None]
                summary[key][flag]=metrics(subset,flag,score)
                if split=='heldout_test':summary[key][flag]['state_bootstrap']=boot(subset,flag,score,20261012)
            subset=[r for r in rows if r['condition1_available']]
            summary[key]['condition1']=metrics(subset,'condition1','condition1_score')
            summary[key]['condition1']['n_evaluable']=len(subset)
            if split=='heldout_test':summary[key]['condition1']['state_bootstrap']=boot(subset,'condition1','condition1_score',20261013)
    (HERE/'prediction_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for k,v in summary.items():
        print(k,v['outcomes'],'C0',*[v['condition0'][j] for j in ('tp','fp','fn','tn')],
              'C1',*[v['condition1'][j] for j in ('tp','fp','fn','tn')],flush=True)

if __name__=='__main__':main()
