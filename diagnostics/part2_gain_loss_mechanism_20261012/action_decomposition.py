"""Existing 16-key initial action probes: pre/post projection contrasts."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TRANSFER=Path('/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/state_eta_chain_features.csv')

def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    source={(r['state_uid'],r['eta_index'],r['chain']):r for r in readcsv(TRANSFER) if r['phase']=='initial'}
    rows=[]
    for sc,file,old,new,crit in [('toy_give_way','toy_loss_analysis.csv','FT','FF','goal'),
                                  ('ring_exchange','ring_gain_analysis.csv','TT','TF','tangential')]:
        for r in readcsv(HERE/file):
            a=source[(r['uid'],r['eta_index'],old)];b=source[(r['uid'],r['eta_index'],new)]
            fp=lambda x,k:float(x[k])
            item=dict(scene=sc,uid=r['uid'],eta_index=r['eta_index'],split=r['split'],outcome=r['b15_outcome'],
                old=old,new=new,noise_count=int(a['noise_count']),
                old_critical_pre=fp(a,crit+'_pre_signed_nominal_motion'),
                new_critical_pre=fp(b,crit+'_pre_signed_nominal_motion'),
                old_critical_post=fp(a,crit+'_signed_nominal_motion'),
                new_critical_post=fp(b,crit+'_signed_nominal_motion'),
                old_critical_exec_norm=fp(a,crit+'_exec_motion_norm'),
                new_critical_exec_norm=fp(b,crit+'_exec_motion_norm'),
                old_terminal_projection_deformation=fp(a,'terminal_projection_deformation'),
                new_terminal_projection_deformation=fp(b,'terminal_projection_deformation'),
                old_field_projection_deformation=fp(a,'field_projection_deformation_mean'),
                new_field_projection_deformation=fp(b,'field_projection_deformation_mean'),
                old_effective_norm=fp(a,'effective_norm'),new_effective_norm=fp(b,'effective_norm'))
            item['critical_pre_delta']=item['new_critical_pre']-item['old_critical_pre']
            item['critical_post_delta']=item['new_critical_post']-item['old_critical_post']
            item['projection_lift_of_contrast']=item['critical_post_delta']-item['critical_pre_delta']
            rows.append(item)
    with (HERE/'initial_action_decomposition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    z=defaultdict(list)
    for r in rows:z[(r['scene'],r['outcome'])].append(r)
    summ={}
    for k,a in z.items():
        summ[':'.join(k)]={name:float(np.median([r[name] for r in a])) for name in
            ('critical_pre_delta','critical_post_delta','projection_lift_of_contrast',
             'old_terminal_projection_deformation','new_terminal_projection_deformation',
             'old_field_projection_deformation','new_field_projection_deformation',
             'old_effective_norm','new_effective_norm')}
        summ[':'.join(k)]['n']=len(a)
    (HERE/'initial_action_summary.json').write_text(json.dumps(summ,indent=2,sort_keys=True)+'\n')
    print('initial action rows',len(rows))

if __name__=='__main__':main()
