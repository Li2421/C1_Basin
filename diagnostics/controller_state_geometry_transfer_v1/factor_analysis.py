from pathlib import Path
import argparse
import numpy as np
from . import factor_data,learn,metrics
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def specs(D):
 c=D['contexts'];allg=list(range(len(c)));tgt=lambda predicate:[i for i,g in enumerate(c) if predicate(g)]
 tests={'WIDTH':tgt(lambda g:g['width']==.65),'EXTRAP':tgt(lambda g:g['width']==.8),'COMBINATION':tgt(lambda g:g['width']==.65 and g['N']==4 and g['topology']=='parallel'),'FAMILY':tgt(lambda g:g['topology']=='parallel'),'FAMILY_REVERSE':tgt(lambda g:g['topology']=='single'),'COUNT':tgt(lambda g:g['N']==4),'TIME':tgt(lambda g:g['target_horizon']==850)}
 trains={k:[g for g in allg if g not in v] for k,v in tests.items()};return trains,tests

def main(mode):
 D=factor_data.load();trains,tests=specs(D);plans=[(a,r) for a in ['ENT','ENT_raw','ENT_mech','MECH'] for r in trains]+[('ENT_mech_no_margin',r) for r in ['WIDTH','FAMILY']];test=list(range(8,16));outdir=HERE/'models/factors'
 assert D['unknown'][:,test].sum()==0, 'Final test contains unknown; explicit interval evaluation required'
 assert D['unknown'].sum()==2, 'Unexpected unknown pattern requires renewed audit'
 if mode=='fit':
  np.savez_compressed(HERE/'data/factors.npz',Y=D['Y'],unknown=D['unknown'],T=D['T'],raw=D['raw'],mech=D['mech'],**{'entity_'+k:v for k,v in D['X'].items()});write(HERE/'data/factor_index.json',D['index']);write(HERE/'data/factor_contexts.json',D['contexts'])
  for arm,reg in plans:
   tr,y=learn.rows(D,[(g,list(range(8)),[0,1]) for g in trains[reg]],allow_unknown=True)
   for sd in learn.SEEDS:learn.fit(D,tr,y,arm,sd,outdir/reg)
  write(HERE/'factor_training_complete.json',{'completed':True,'counts':D['counts'],'unknown_physical_outcomes':1,'unknown_horizon_labels':2,'plans':plans,'cpu_seconds_rollout':D['cpu_seconds']})
 else:
  result={};vals={};rng=np.random.default_rng(20261011);bs=rng.integers(0,8,(2000,8))
  for arm,reg in plans:
   tr,y=learn.rows(D,[(g,list(range(8)),[0,1]) for g in trains[reg]],allow_unknown=True);prior=learn.prior(tr,y,2,4);ps=[np.load(outdir/reg/f'{arm}_{s}.npz')['pred'] for s in learn.SEEDS];ensemble=np.mean(ps,0);groups={'all':tests[reg],**{str(h):[g for g in tests[reg] if D['contexts'][g]['target_horizon']==h] for h in [400,850]}};rec={}
   for name,gs in groups.items():
    if not gs:continue
    yy=D['Y'][gs][:,test];r,a=metrics.evaluate(yy,ensemble[gs][:,test],prior);rec[name]={'ensemble':r,'per_seed':{str(s):metrics.stats(yy,p[gs][:,test],prior)[0] for s,p in zip(learn.SEEDS,ps)},'test_geometries':gs};vals[f'{arm}_{reg}_{name}']=a
   result[f'{arm}_{reg}']=rec;print(arm,reg,'succ',rec['all']['ensemble']['success'],'state_eta',rec['all']['ensemble']['state_eta'],flush=True)
  comp={}
  for reg in trains:
   for arm in ['ENT_raw','ENT_mech','MECH','ENT_mech_no_margin']:
    if f'{arm}_{reg}' not in result:continue
    for h in ['all','400','850']:
     if f'{arm}_{reg}_{h}' not in vals:continue
     d=vals[f'{arm}_{reg}_{h}']['success']-vals[f'ENT_{reg}_{h}']['success'];comp[f'{arm}-ENT:{reg}:{h}']={'mean':d.mean(),'ci95':np.percentile(d[bs].mean(1),[2.5,97.5])}
     dq=vals[f'{arm}_{reg}_{h}']['q_mse']-vals[f'ENT_{reg}_{h}']['q_mse'];comp[f'{arm}-ENT:{reg}:{h}'].update(delta_q_mse=dq.mean(),q_mse_ci95=np.percentile(dq[bs].mean(1),[2.5,97.5]))
  # Direct, independently crossed factor effects at BOTH seeds; state is the block.
  q=np.nan_to_num(D['Y'],nan=0.).mean(-1).reshape(2,2,3,2,16,2,4) # topology,N,width,H,state,C,eta
  factors={}
  for name,axis in [('topology',0),('N',1),('width',2),('horizon',3)]:
   d=np.take(q,-1,axis)-np.take(q,0,axis);state_axis=3;others=tuple(i for i in range(d.ndim) if i!=state_axis);sv=d.mean(others);factors[name]={'last_minus_first_mean':sv.mean(),'state_ci95':np.percentile(sv[np.random.default_rng(20261011).integers(0,16,(2000,16))].mean(1),[2.5,97.5]),'levels':{'topology':['single','parallel'],'N':[2,4],'width':[.5,.65,.8],'horizon':[400,850]}[name]}
  from .metrics import center,corr
  yy=np.nan_to_num(D['Y'],nan=0.).reshape(2,2,3,2,16,2,4,2);halves=[yy[...,i] for i in [0,1]];interactions={}
  axes_names=['topology','N','width','horizon','state','controller','eta']
  for axes in [(0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(0,1),(0,2),(3,4,6)]:
   other=tuple(i for i in range(7) if i not in axes)
   def pattern(x):return center(x.mean(other,keepdims=True),axes)
   aa,bb=map(pattern,halves);energies=[]
   for ss in np.random.default_rng(20261011).integers(0,16,(2000,16)):
    aaa=pattern(np.take(halves[0],ss,4));bbb=pattern(np.take(halves[1],ss,4));energies.append(np.mean(aaa*bbb))
   interactions['×'.join(axes_names[i] for i in axes)]={'unbiased_energy':float(np.mean(aa*bb)),'state_ci95':np.percentile(energies,[2.5,97.5]),'split_seed_reliability':corr(aa,bb)}
  sensitivities={}
  for lo,hi in [(0,0),(0,1),(1,1)]:
   world=D['Y'].copy();world[16][D['unknown'][16]]=lo;world[17][D['unknown'][17]]=hi;assert np.isfinite(world).all();worldq=world.mean(-1).reshape(2,2,3,2,16,2,4);row={}
   for name,axis in [('topology',0),('N',1),('width',2),('horizon',3)]:row[name]=float((np.take(worldq,-1,axis)-np.take(worldq,0,axis)).mean())
   wh=world.reshape(2,2,3,2,16,2,4,2)
   for axes in [(0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(0,1),(0,2),(3,4,6)]:
    other=tuple(i for i in range(7) if i not in axes);aa=center(wh[...,0].mean(other,keepdims=True),axes);bb=center(wh[...,1].mean(other,keepdims=True),axes);row['×'.join(axes_names[i] for i in axes)]=float(np.mean(aa*bb))
   sensitivities[f'{lo},{hi}']=row
  witnesses=(np.nanmax(D['Y'][1::2],axis=(2,3,4))>0);joint_witnesses={'config_state_success_any_eta_controller_seed':int(witnesses.sum()),'total_config_states':int(witnesses.size),'by_configuration':witnesses.sum(1),'unwitnessed_not_declared_infeasible':True}
  write(HERE/'factor_results.json',{'results':result,'paired_feature_comparisons':comp,'factor_effects':factors,'factor_interactions':interactions,'one_unknown_monotone_completion_sensitivity':sensitivities,'factor_descriptive_point_completion':'unknown as failure; see all three monotone sensitivity worlds','joint_witnesses':joint_witnesses,'outcomes':D['counts'],'unknown':int(D['unknown'].sum()),'n_physical_rollouts':3072,'correlated_horizon_labels':6144,'state_split':{'train':list(range(8)),'test':test},'scope':'two route topologies under one analytic FF navigation program; no held-controller/Flow/general cross-scene claim'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['fit','evaluate']);x=a.parse_args();main(x.mode)
