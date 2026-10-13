"""Numerical replay sensitivity, without overwriting any primary prediction."""
from pathlib import Path
import argparse
import json
import numpy as np
from . import metrics,learn
from .checkpoint_tools import dataset
from .factor_analysis import specs
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--recorded',action='store_true');args=ap.parse_args();suffix='_recorded' if args.recorded else ''
 audit=json.loads((HERE/f'CHECKPOINT_REPLAY_gpu{suffix}.json').read_text());assert audit['completed'] and len(audit['checks'])==138
 output={};arrays={};bs=np.random.default_rng(20261011).integers(0,8,(2000,8))
 for kind in ['legacy_diagnostic','factors','normalization']:
  D=dataset(kind);test=list(range(24,32)) if kind=='legacy_diagnostic' else list(range(8,16))
  if kind=='legacy_diagnostic':
   W=json.loads((HERE/'data/legacy_diagnostic_metadata.json').read_text())['W']
  else:tests=specs({'contexts':json.loads((HERE/'data/factor_contexts.json').read_text())})[1]
  for root in sorted((HERE/'models'/kind).iterdir()):
   reg=root.name
   for f in sorted(root.glob('*_17.npz')):
    arm=f.stem[:-3];old=[];new=[]
    for sd in learn.SEEDS:
     name=f'{arm}_{sd}.npz';old.append(np.load(root/name)['pred']);new.append(np.load(HERE/f'replay_predictions/gpu{suffix}'/kind/reg/name)['pred'])
    train=np.load(f);prior=learn.prior(train['tr'],train['y'],D['Y'].shape[2],D['Y'].shape[3])
    if kind=='legacy_diagnostic':groups={'interpolation':[W.index(w) for w in ([.44,.50] if reg=='EXTRAP' else [.44,.50,.56])],'extrapolation':[W.index(w) for w in ([.59,.62] if reg=='EXTRAP' else [.62])]}
    else:groups={'all':tests[reg]}
    for group,gs in groups.items():
     yy=D['Y'][gs][:,test];p0=np.mean(old,0)[gs][:,test];p1=np.mean(new,0)[gs][:,test];before,a0=metrics.stats(yy,p0,prior);after,a1=metrics.stats(yy,p1,prior);d=a1['success']-a0['success'];key=f'{kind}:{arm}_{reg}:{group}'
     top0=np.take_along_axis(p0,p0.argmax(-1)[...,None],-1)[...,0];part=np.partition(p0,-2,axis=-1);margin=part[...,-1]-part[...,-2];changed=p0.argmax(-1)!=p1.argmax(-1)
     output[key]={'before':before,'replay':after,'delta_success':d.mean(),'delta_success_ci95':np.percentile(d[bs].mean(1),[2.5,97.5]),'argmax_changed':int(changed.sum()),'choices':int(changed.size),'changed_with_old_top_margin_lt_1e_4':int((changed&(margin<1e-4)).sum()),'changed_old_exact_ties':int((changed&(margin==0)).sum()),'maximum_Q_change':float(abs(p1-p0).max())};arrays[key]=a1
 # Reassess primary paired claims under replay, same state bootstrap.
 contrasts={}
 pairs=[('legacy_diagnostic:ENT_MANY:interpolation','legacy_diagnostic:ENT_FEW:interpolation')]
 pairs += [(f'factors:ENT_mech_{reg}:all',f'factors:ENT_{reg}:all') for reg in tests]
 for left,right in pairs:
  a,b=arrays[left],arrays[right];s=a['success']-b['success'];q=a['q_mse']-b['q_mse'];contrasts[left+' minus '+right]={'success_delta':s.mean(),'success_ci95':np.percentile(s[bs].mean(1),[2.5,97.5]),'q_mse_delta':q.mean(),'q_mse_ci95':np.percentile(q[bs].mean(1),[2.5,97.5])}
 summary={'max_per_checkpoint_Q_change':max(c['max_abs_error'] for c in audit['checks']),'max_ensemble_success_change':max(abs(v['delta_success']) for v in output.values()),'groups_with_success_change':{k:v['delta_success'] for k,v in output.items() if abs(v['delta_success'])>1e-12},'changed_argmax_total':sum(v['argmax_changed'] for v in output.values()),'changed_with_old_top_margin_lt_1e_4':sum(v['changed_with_old_top_margin_lt_1e_4'] for v in output.values())}
 write(HERE/f'REPLAY_SENSITIVITY{suffix}.json',{'completed':True,'strict_prediction_reproduction':audit['strict_numeric_reproduction'],'interpretation':'Saved predictions are immutable primary evidence. Matching precision replay.' if args.recorded else 'Initial audit omitted original factor x64 setting; this is a mismatched-precision sensitivity, NOT evidence of same-setting recompile instability. See recorded-precision replay.','summary':summary,'results':output,'primary_paired_contrasts_replay':contrasts});print(summary)
if __name__=='__main__':main()
