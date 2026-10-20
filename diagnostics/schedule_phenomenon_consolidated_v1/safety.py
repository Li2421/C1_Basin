"""Read-only raw flags and physical trace summaries; no simulations."""
import os,sys,json,csv,datetime
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig');sys.path.insert(0,str(ROOT/'diagnostics/controller_loco_real_flow_v1/plot_vendor'))
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
inv=json.load(open(HERE/'DATA_INVENTORY.json'));groups={};lookup={};metrics={};collisions={};timegrid={};tables=[];rng=np.random.default_rng(202610152100)
for tag,m in inv.items():
 z=np.load(HERE/(tag+'.npz'));names=z['names'].tolist();states=z['states'].tolist();arms=z['arms'].tolist();lookup[tag]=(names,states,arms);metrics[tag]={a:[] for a in arms};collisions[tag]=np.full(z['success'].shape,np.nan);p=m['folder'];groups.setdefault(p,[]).append(tag);H=1100 if m['scene']=='four_way_intersection' else 700 if m['scene']=='ring_exchange' else 850;ts=[]
 for a in arms:
  if not a.startswith('S'):continue
  if '_t' in a:t=round((3 if a.endswith('t1') else 5)*H/7)
  elif '_' in a:t=int(a.rsplit('_',1)[1])
  else:t=int(a[1:])
  ts.append(t)
 timegrid[tag]=sorted(set(ts))
files={tag:(HERE/(tag+'_SAFETY_RECORDS.jsonl')).open('w') for tag in inv};windows={};seen={tag:set() for tag in inv};status={tag:dict(raw=0,trace_missing=0,coarse_trace=0,trace_available=0) for tag in inv}
for folder,tags in groups.items():
 p=Path(folder);dirs=[d for d in p.iterdir() if d.is_dir() and (d.name in ('main','train','test','stage2','stage3') or d.name.startswith('wave')) and (d/'raw').exists()];paths=[f for d in dirs for f in (d/'raw').glob('*.json')];historical=[]
 if p.name=='controller_eta_switch_time_v1':historical=json.load(open(p/'HISTORICAL_RAW_INDEX.json'));paths+=list({Path(x['path']) for x in historical if x['case']['eta_index'] in (10,13) and x['case']['seed']<4})
 for f in paths:
  r=json.load(open(f));c=r.get('case',{});name=c.get('base_controller_name',c.get('controller_name'));state=c.get('state_uid',r.get('state_uid'));seed=c.get('seed',c.get('future_index',0));assigned=[]
  for tag in tags:
   ns,ss,aa=lookup[tag];arm=c.get('arm')
   if tag=='R9':arm=f"S{c['eta_a_index']}to{c['eta_b_index']}_{c['switch_step']}" if 'eta_a_index' in c and c['switch_step']<700 else 'F'+str(c.get('eta_index'))
   if name in ns and state in ss and arm in aa and seed<collisions[tag].shape[3]:assigned.append((tag,arm,ns.index(name),ss.index(state),aa.index(arm)))
  if not assigned:continue
  pairmin=None;wallmin=None;full=False;pair_series=None;pos=None;z=None;cfg={};trace=r.get('trace_path');swept=r.get('minimum_swept_clearance')
  if trace and Path(trace).exists():
   try:
    with np.load(trace,allow_pickle=False) as zz:
     if 'positions' in zz:pos=zz['positions'];steps=zz['steps'] if 'steps' in zz else np.arange(len(pos));cfg=json.loads(str(zz['metadata_json'].item()))['config'] if 'metadata_json' in zz else {};walls=zz['walls'] if 'walls' in zz else np.empty((0,2,2));full=np.array_equal(steps,np.arange(len(pos))) and len(pos)==r.get('physical_steps',len(pos)-1)+1
    if pos is not None and len(pos):
     pairs=np.array([(i,j) for i in range(pos.shape[1]) for j in range(i+1,pos.shape[1])]);pair_series=np.linalg.norm(pos[:,pairs[:,0]]-pos[:,pairs[:,1]],axis=-1).min(1);pairmin=float(pair_series.min())
     if 'obstacle_radius' in cfg:
      rad=np.linalg.norm(pos,axis=-1);wallmin=float(min((rad-cfg['obstacle_radius']-cfg['agent_radius']).min(),(cfg['outer_radius']-rad-cfg['agent_radius']).min()))
     elif len(walls):
      a=walls[:,0];d=walls[:,1]-a;dp=pos[:,:,None,:]-a[None,None,:,:];t=np.clip(np.sum(dp*d[None,None,:,:],-1)/np.maximum(np.sum(d*d,-1),1e-30),0,1);dist=np.linalg.norm(dp-t[:,:,:,None]*d[None,None,:,:],axis=-1);wallmin=float(dist.min()-cfg.get('agent_radius',0)-cfg.get('wall_radius',0))
   except (ValueError,KeyError) as e:full=False
  for tag,arm,i,s,j in assigned:
   key=(i,s,j,seed)
   if key in seen[tag]:continue
   seen[tag].add(key);collision=r.get('collision');collisions[tag][key]=float(collision) if collision is not None else np.nan;row=dict(controller=name,state_uid=state,seed=seed,arm=arm,collision=collision,unknown=r.get('numerical_failure'),minimum_pair_center_distance=pairmin,minimum_discrete_wall_clearance=wallmin,minimum_swept_combined_clearance=swept,full_resolution=bool(full),trace=trace,local_windows={})
   status[tag]['raw']+=1;status[tag]['trace_available' if pos is not None else 'trace_missing']+=1;status[tag]['coarse_trace']+=bool(pos is not None and not full)
   for t in timegrid[tag]:
    value=float(pair_series[max(0,t-20):t+61].min()) if full and len(pos)>t+60 else None;row['local_windows'][str(t)]=value
   windows[tag,name,state,seed,arm]=row['local_windows'];metrics[tag][arm].append(row);files[tag].write(json.dumps(row)+'\n')
 print('Safety read',p.name,len(paths),flush=True)
for f in files.values():f.close()
results={};local=[]
def dist(values):
 v=np.array([x for x in values if x is not None and np.isfinite(x)]);return dict(n=len(v),median=float(np.median(v)) if len(v) else None,p05=float(np.quantile(v,.05)) if len(v) else None,minimum=float(v.min()) if len(v) else None)
for tag,m in inv.items():
 ns,ss,arms=lookup[tag];z=np.load(HERE/(tag+'.npz'));unknown=z['unknown'];out={};C,S,A,Q=unknown.shape
 for j,a in enumerate(arms):
  rows=metrics[tag][a];co=collisions[tag][:,:,j,:];uc=np.where(z['available'][:,j,None,None],unknown[:,:,j,:],np.nan);validco=np.isfinite(co);den=validco.sum();rate=float(np.nansum(co)/den) if den else None;boot=[];ub=[]
  for _ in range(5000):
   cc=rng.integers(0,C,C);st=rng.integers(0,S,(C,S));values=co[cc[:,None],st];boot.append(float(np.nanmean(values)) if np.isfinite(values).any() else np.nan);ub.append(float(np.nanmean(uc[cc[:,None],st])))
  out[a]=dict(episodes=len(rows),collisions=int(np.nansum(co)),collision_rate=rate,collision_CI=np.nanquantile(boot,[.025,.975]).tolist() if den else None,unknown=int(sum(r['unknown'] or 0 for r in rows)),unknown_rate=float(sum(r['unknown'] or 0 for r in rows)/len(rows)) if rows else None,unknown_CI=np.quantile(ub,[.025,.975]).tolist(),pair_center_distance=dist([r['minimum_pair_center_distance'] for r in rows]),wall_discrete_clearance=dist([r['minimum_discrete_wall_clearance'] for r in rows]),swept_combined_clearance=dist([r['minimum_swept_combined_clearance'] for r in rows]));tables.append(dict(dataset=tag,arm=a,episodes=len(rows),collision_rate=rate,unknown_rate=out[a]['unknown_rate'],pair_median=out[a]['pair_center_distance']['median'],pair_p05=out[a]['pair_center_distance']['p05'],pair_min=out[a]['pair_center_distance']['minimum'],wall_median=out[a]['wall_discrete_clearance']['median'],wall_p05=out[a]['wall_discrete_clearance']['p05'],wall_min=out[a]['wall_discrete_clearance']['minimum']))
 for arm in arms:
  if not arm.startswith('S'):continue
  if 'to' in arm:first,last=arm[1:].split('to');fa='F'+first;fb='F'+last.split('_')[0]
  elif arm.startswith(('S13_','S15_')):fa='F'+arm[1:3];fb='F10'
  else:fa='F13';fb='F10'
  t=round((3 if arm.endswith('t1') else 5)*(1100 if m['scene']=='four_way_intersection' else 700 if m['scene']=='ring_exchange' else 850)/7) if '_t' in arm else int(arm.rsplit('_',1)[1]) if '_' in arm else int(arm[1:])
  for fixed in (fa,fb):
   if fixed not in arms:continue
   vals={};missing=0
   for name in ns:
    rows=[]
    for state in ss:
     dif=[]
     for seed in range(Q):
      one=windows.get((tag,name,state,seed,arm),{}).get(str(t));two=windows.get((tag,name,state,seed,fixed),{}).get(str(t))
      if one is None or two is None:missing+=1
      else:dif.append(one-two)
     if dif:rows.append(float(np.mean(dif)))
    if rows:vals[name]=np.array(rows)
   draws=[];validnames=list(vals)
   if validnames:
    for _ in range(5000):
     picked=rng.integers(0,len(validnames),len(validnames));draws.append(float(np.mean([np.mean(vals[validnames[i]][rng.integers(0,len(vals[validnames[i]]),len(vals[validnames[i]]))]) for i in picked])))
   local.append(dict(dataset=tag,arm=arm,fixed=fixed,switch_step=t,window=[t-20,t+60],valid_controllers=len(vals),missing_or_early_terminated_seed_pairs=missing,mean_difference=float(np.mean([v.mean() for v in vals.values()])) if vals else None,CI=np.quantile(draws,[.025,.975]).tolist() if draws else None))
 results[tag]=dict(status=status[tag],arms=out);print('Safety summarize',tag,flush=True)
(HERE/'SAFETY_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n');(HERE/'LOCAL_SWITCH_SAFETY.json').write_text(json.dumps(local,indent=2)+'\n')
with (HERE/'ANALYSIS_C.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=tables[0]);w.writeheader();w.writerows(tables)
fig,axes=plt.subplots(2,1,figsize=(18,8));labels=[r['dataset']+'/'+r['arm'] for r in tables];axes[0].bar(np.arange(len(tables)),[r['collision_rate'] or 0 for r in tables]);axes[0].set_ylabel('Collision rate (raw events)');axes[1].bar(np.arange(len(tables)),[r['unknown_rate'] or 0 for r in tables]);axes[1].set_ylabel('Numerical unknown rate');axes[1].set_xticks(np.arange(len(labels)),labels,rotation=90,fontsize=4);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('arm_safety.'+ext),dpi=180)
plt.close(fig);print('C complete',flush=True)
