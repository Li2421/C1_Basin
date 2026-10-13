"""Read-only paired seed-0 action/trajectory/margin explanation ledger."""
from __future__ import annotations

import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from reuse_analysis import HERE,write_csv,write_json


def main():
    rows=list(csv.DictReader((HERE/'gain_loss_audit.csv').open()))
    out=[];stats=defaultdict(list)
    for r in rows:
        if r['seed']!='0' or not r['old_result'] or not r['new_result'] or not r['delta_rho']:continue
        pa=Path(r['old_result']).with_suffix('.npz');pb=Path(r['new_result']).with_suffix('.npz')
        if not pa.exists() or not pb.exists():continue
        with np.load(pa) as z:
            a=z['actions'].copy();p=z['positions'].copy();ka=z['noise_keys'].copy()
        with np.load(pb) as z:
            b=z['actions'].copy();q=z['positions'].copy();kb=z['noise_keys'].copy()
        n=min(len(a),len(b));t=min(80,len(p)-1,len(q)-1)
        first=float(np.linalg.norm(a[0]-b[0])) if n else None
        mean=float(np.mean(np.linalg.norm((a[:n]-b[:n]).reshape(n,-1),axis=1))) if n else None
        maxpos=float(np.max(np.linalg.norm((p[:t+1]-q[:t+1]).reshape(t+1,-1),axis=1)))
        length=max(len(p),len(q));pp=np.concatenate([p,np.repeat(p[-1:],length-len(p),axis=0)])
        qq=np.concatenate([q,np.repeat(q[-1:],length-len(q),axis=0)])
        maxfull=float(np.max(np.linalg.norm(pp-qq,axis=2)))
        noise=bool(np.array_equal(ka[:n],kb[:n]))
        drho=float(r['delta_rho']);oldrho=float(r['old_rho']);newrho=float(r['new_rho'])
        item=dict(scene=r['scene'],uid=r['uid'],eta_index=int(r['eta_index']),seed=0,old=r['old'],new=r['new'],
            outcome=r['outcome'],first_action_l2=first,mean_common_action_l2=mean,
            max_common_position_l2_to80=maxpos,max_full_agent_position_l2=maxfull,
            positive_margin_preservation_certificate=bool(oldrho>maxfull),
            margin_lipschitz_residual=float(abs(newrho-oldrho)-maxfull),
            common_steps=n,noise_keys_match=noise,
            old_rho=oldrho,new_rho=newrho,delta_rho=drho,
            margin_sign_predicts_membership=bool((oldrho>0)==(r['old_success']=='True') and (newrho>0)==(r['new_success']=='True')),
            old_termination=r['old_termination'],new_termination=r['new_termination'])
        out.append(item)
        stats[(r['scene'],r['old'],r['new'],r['outcome'])].append(item)
    fields=list(out[0]);write_csv('gain_loss_mechanism_seed0.csv',out,fields)
    summary={}
    for key,vals in stats.items():
        summary[':'.join(key)]=dict(n=len(vals),median_first_action=float(np.median([x['first_action_l2'] for x in vals])),
            median_mean_action=float(np.median([x['mean_common_action_l2'] for x in vals])),
            median_max_position_to80=float(np.median([x['max_common_position_l2_to80'] for x in vals])),
            median_delta_rho=float(np.median([x['delta_rho'] for x in vals])))
    write_json('gain_loss_mechanism_summary.json',dict(rows=len(out),all_noise_keys_match=all(x['noise_keys_match'] for x in out),
        margin_label_mismatches=sum(not x['margin_sign_predicts_membership'] for x in out),
        margin_lipschitz_violations=sum(x['margin_lipschitz_residual']>1e-9 for x in out),
        positive_margin_certificates=sum(x['positive_margin_preservation_certificate'] for x in out),
        certificate_failures=sum(x['positive_margin_preservation_certificate'] and x['new_rho']<=0 for x in out),groups=summary))
    print('gain/loss mechanism rows',len(out),flush=True)


if __name__=='__main__':main()
