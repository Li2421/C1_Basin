"""Stage-1 fits and paired evaluations. No final labels used for fitting decisions."""
from pathlib import Path
import json,argparse
import numpy as np
from . import learn,metrics,toy_data
HERE=Path(__file__).resolve().parent

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,np.ndarray):return clean(x.tolist())
 return x

def write(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def setup(broad=True):
 source=HERE/'broad_main' if broad else HERE.parent/'controller_state_transfer_v1';D=toy_data.build(source,broad)
 train=list(range(16)) if broad else list(range(8,24));test=list(range(16,24)) if broad else list(range(24,32));cal=list(range(24,28)) if broad else list(range(8,24))
 tw=[D['W'].index(w) for w in [.41,.47,.53,.59]];ends=[tw[0],tw[-1]]
 specs={'MANY':[(g,[train[(4*i+k)%16] for k in range(8)],[0,1]) for i,g in enumerate(tw)],'FEW':[(g,train,[0,1]) for g in ends],
 'FEW_x2':[(g,train,[0,1,2,3]) for g in ends],'ALL':[(g,train,[0,1]) for g in tw],'EXTRAP':[(g,train,[0,1]) for g in tw[:3]]}
 sets={'interpolation':[D['W'].index(w) for w in [.44,.50,.56]],'seen_geometry':tw,'extrapolation':[D['W'].index(.62)]}
 assert not(set(train)&set(test)) and not(set(cal)&set(test));return D,specs,sets,test,cal

def main(mode,broad):
 D,specs,sets,test,cal=setup(broad);tag='broad_main' if broad else 'legacy_diagnostic';out=HERE/'models'/tag
 plans=[('ENT',r) for r in specs]+[(arm,r) for arm in ['STATE','MLP'] for r in ['MANY','ALL']]+[(arm,'MANY') for arm in ['ENT_raw','ENT_mech','MECH','RANK']]
 if mode=='fit':
  toy_data.save_data(D,tag)
  for arm,regime in plans:
   tr,y=learn.rows(D,specs[regime]);assert not set(tr[:,1])&set(test)
   for seed in learn.SEEDS:learn.fit(D,tr,y,arm,seed,out/regime)
  write(HERE/(tag+'_training_complete.json'),{'completed':True,'plans':plans,'seeds':learn.SEEDS,'outcomes':D['counts'],'rollout_cpu_seconds':D['cpu_seconds']})
 else:
  result={};state_values={}
  for arm,regime in plans:
   tr,y=learn.rows(D,specs[regime]);prior=learn.prior(tr,y,2,8);ps=[np.load(out/regime/f'{arm}_{s}.npz')['pred'] for s in learn.SEEDS];record={}
   regsets=sets if regime!='EXTRAP' else {'interpolation':[D['W'].index(w) for w in [.44,.50]],'seen_geometry':[D['W'].index(w) for w in [.41,.47,.53]],'extrapolation':[D['W'].index(w) for w in [.59,.62]]}
   for name,gs in regsets.items():
    yy=D['Y'][gs][:,test];per=[]
    for pp in ps:per.append(metrics.stats(yy,pp[gs][:,test],prior)[0])
    pp=np.mean(ps,0);ensemble,arrays=metrics.evaluate(yy,pp[gs][:,test],prior)
    record[name]={'per_seed':dict(zip(map(str,learn.SEEDS),per)),'ensemble':ensemble,'n_states':len(test),'n_geometries':len(gs)}
    state_values[f'{arm}_{regime}_{name}']=arrays
   result[f'{arm}_{regime}']=record;print(arm,regime,record['interpolation']['ensemble']['success'],record['interpolation']['ensemble']['state_eta'],flush=True)
  comparisons={};rng=np.random.default_rng(20261011);draws=rng.integers(0,len(test),(2000,len(test)))
  for lhs,rhs in [('ENT_MANY','ENT_FEW'),('ENT_FEW_x2','ENT_FEW'),('ENT_ALL','ENT_MANY'),('MLP_MANY','ENT_MANY'),('ENT_mech_MANY','ENT_MANY'),('ENT_raw_MANY','ENT_MANY'),('RANK_MANY','ENT_MANY'),('STATE_MANY','ENT_MANY')]:
   if lhs not in result:continue
   for name in sets:
    a=state_values[f'{lhs}_{name}']['success']-state_values[f'{rhs}_{name}']['success'];comparisons[f'{lhs}-{rhs}:{name}']={'delta_success':a.mean(),'ci95':np.percentile(a[draws].mean(1),[2.5,97.5])}
    d=state_values[f'{lhs}_{name}']['q_mse']-state_values[f'{rhs}_{name}']['q_mse'];comparisons[f'{lhs}-{rhs}:{name}'].update(delta_q_mse=d.mean(),q_mse_ci95=np.percentile(d[draws].mean(1),[2.5,97.5]))
  write(HERE/(tag+'_results.json'),{'results':result,'paired_comparisons':comparisons,'counts':D['counts'],'strong_gate_required':broad,'state_split':{'test':test,'calibration':cal},'scope':'prospective broad-state panel' if broad else 'exploratory low-headroom legacy diagnostic; not strong-state validation'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','evaluate']);ap.add_argument('--legacy',action='store_true');a=ap.parse_args();main(a.mode,not a.legacy)
