"""Read-only cached trajectory arithmetic. No runtime controller, Flow, projector or env.step."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib,time,csv,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.integrate import quad
START=time.perf_counter();CPU=time.process_time()
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;H=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
reads={}
def read(p):
 p=Path(p);b=p.read_bytes();reads[str(p)]=hashlib.sha256(b).hexdigest();return b
def load(p):return json.loads(read(p))
a=load(ROOT/'agent_a/geometry_contract.json');b=load(ROOT/'agent_b/results.json');c=load(ROOT/'agent_c/results.json')
for p in [ROOT/'agent_a/report.md',ROOT/'agent_b/report.md',ROOT/'agent_c/report.md']:read(p)
sc=load(H/'screening_manifest.json');cm=load(H/'confirmation_manifest.json');index=load(H/'confirm_q16_tasks.json');read(H/'implementation_audit.json');read(H/'runner.py')
assert index['manifest_sha256']==reads[str(H/'screening_manifest.json')]
source_checks={p:hashlib.sha256(read(p)).hexdigest()==v for p,v in a['source_hashes'].items()}
assert all(source_checks.values())
R=a['constants_m']['R_w'];Ds=a['constants_m']['D_s'];hh=a['constants_m']['h'];aa=a['constants_m']['a'];ww=.2;HH=a['constants_m']['H']
walls=np.array([w['endpoints'] for w in a['walls']]); dt=.05

def membership(p):
 x1,y1,x2,y2=p[:,0,0],p[:,0,1],p[:,1,0],p[:,1,1]
 common=(x2>=-.025-1e-12)&(x2<=.025+1e-12)&(y1>=-.02-1e-12)&(y1<=1e-12)&(x1>=-.65-1e-12)
 entry=common&(x1<=-.35+1e-12)&(y2>=-.02-1e-12)&(y2<=.02+1e-12)
 pas=common&(x1<=x2+.34+1e-12)&(y2>=.38-1e-12)&(y2<=.46+1e-12)
 return entry,pas

def point_segment(p,w0,w1):
 v=w1-w0; q=np.sum((p-w0)*v,axis=-1)/np.sum(v*v,axis=-1)
 q=np.clip(q,0,1);return np.linalg.norm(p-w0-q[...,None]*v,axis=-1)

def safe_metrics(p):
 # All sampled positions and all linear physical swept segments.
 wallmin=min(float(point_segment(p,w[0],w[1]).min()) for w in walls)
 A=p[:-1];B=p[1:];v=B-A;vv=np.sum(v*v,axis=-1);vvs=np.maximum(vv,1e-300)
 for W in walls:
  # Segment/segment distance = minimum endpoint-to-segment distance unless crossing.
  for E in W:
   q=np.clip(np.sum((E-A)*v,axis=-1)/vvs,0,1)
   wallmin=min(wallmin,float(np.linalg.norm(A+q[...,None]*v-E,axis=-1).min()))
  r=W[1]-W[0];cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
  denom=cross(v,r); nz=abs(denom)>1e-14;dsafe=np.where(nz,denom,1.)
  t=cross(W[0]-A,r)/dsafe;u=cross(W[0]-A,v)/dsafe
  if np.any(nz&(t>=0)&(t<=1)&(u>=0)&(u<=1)):wallmin=0.
 rel=p[:,0]-p[:,1];dv=rel[1:]-rel[:-1];dv2=np.sum(dv*dv,axis=-1)
 t=np.clip(-np.sum(rel[:-1]*dv,axis=-1)/np.maximum(dv2,1e-300),0,1)
 pairmin=float(np.linalg.norm(rel[:-1]+t[:,None]*dv,axis=-1).min())
 return wallmin-R,pairmin-Ds

states={x['uid']:x for x in cm['states'] if x['scenario']=='toy_give_way'}
initials=[]
for uid,st in states.items():
 p=np.array(st['physical']['positions']);x1,y1=p[0];x2,y2=p[1]
 # Certified stage ordering: y1 to min(y1,0); x1 to max(x1,-.65); x2 to 0.
 targetx=max(x1,-.65);targety=min(y1,0)
 cond=(-.705<=x1<=-.59 and .59<=x2<=.705 and -.02<=y1<=.02 and -.02<=y2<=.02)
 times=[max(y1,0)/.026,max(-.65-x1,0)/.05,x2/.05,(.4-y2)/.03,(.34-targetx)/.1]
 steps=[math.ceil(t/dt-1e-10) for t in times]
 en,pa=membership(p[None]);initials.append(dict(uid=uid,positions=p.tolist(),entry=bool(en[0]),pas=bool(pa[0]),reference_certified=bool(cond and sum(steps)<=850),stage_times=times,stage_steps=steps,total_steps=sum(steps),total_time_upper=sum(steps)*dt))
rows=[];positions_read=0;noise_mismatch=0;dynamics_error=0.;pairs={};ids=[]
selected=[t for t in index['tasks'] if t['state']['scenario']=='toy_give_way' and t['seed']==0 and t['chain'] in ['TT','FF']]
assert len(selected)==768
for t in selected:
 rid=t['id'];js=H/'results'/f'{rid}.json';npz=H/'results'/f'{rid}.npz';ids.append(rid)
 if not js.exists() or not npz.exists():
  rows.append(dict(id=rid,chain=t['chain'],eta_index=t['eta_index'],eta=t['eta'],state_uid=t['state']['uid'],available=False));continue
 meta=load(js);raw=read(npz)
 import io
 with np.load(io.BytesIO(raw),allow_pickle=False) as d:
  p=d['positions'];u=d['actions'];keys=d['noise_keys'];goals=d['goals']
 positions_read+=len(p)
 assert meta['id']==rid and meta['eta']==t['eta'] and meta['chain']==t['chain'] and meta['seed']==0
 assert meta['manifest_sha256']==index['manifest_sha256']
 assert np.max(np.abs(p[0]-np.array(t['state']['physical']['positions'])))<1e-12
 assert np.max(np.abs(goals-np.array(t['state']['physical']['goals'])))<1e-12
 dynamics_error=max(dynamics_error,float(np.max(abs(np.diff(p,axis=0)-dt*u))))
 en,pa=membership(p);s=p[:,0,0]-p[:,1,0];z=p[:,1,1]-p[:,0,1]
 firsten=int(np.flatnonzero(en)[0]) if en.any() else -1;firstpa=int(np.flatnonzero(pa)[0]) if pa.any() else -1
 wallmargin,pairmargin=safe_metrics(p)
 complete=-1;hold=False
 if firstpa>=0:
  exits=np.flatnonzero((s>=.34-1e-10)&(np.arange(len(p))>=firstpa))
  if len(exits):
   complete=int(exits[0]);hold=bool(np.all(z[firstpa:complete+1]>Ds))
 en_then_exit=False
 if firsten>=0:
  for pe in np.flatnonzero(pa & (np.arange(len(p))>=firsten)):
   exits=np.flatnonzero((s>=.34-1e-10)&(np.arange(len(p))>=pe))
   if len(exits) and np.all(z[pe:exits[0]+1]>Ds):en_then_exit=True;break
 safety=wallmargin>=-1e-8 and pairmargin>=-1e-8 and not meta['collision']
 term=meta['termination'];complete_record=bool(term in ['success','deadlock','collision','timeout'] and meta.get('error') is None and len(p)==len(u)+1)
 row=dict(id=rid,chain=t['chain'],eta_index=t['eta_index'],eta=json.dumps(t['eta']),state_uid=t['state']['uid'],seed=0,available=True,termination=term,numerical_failure=term=='numerical_failure',collision=meta['collision'],success=meta['success'],steps=meta['steps'],complete_record=complete_record,entry_initial=bool(en[0]),pass_initial=bool(pa[0]),entry_visited=bool(en.any()),pass_visited=bool(pa.any()),first_entry=firsten,first_pass=firstpa,first_exit_after_pass=complete,held_z_through_exit=hold,pass_then_held_exit_no_collision=bool(complete>=0 and hold and safety),entry_then_pass_held_exit=bool(en_then_exit and safety),min_wall_swept_margin_m=wallmargin,min_pair_swept_margin_m=pairmargin,safety_radius_respected=safety,max_z=float(z.max()))
 rows.append(row)
 key=(t['state']['uid'],t['eta_index']);kb=keys.copy()
 if key in pairs:
  old=pairs.pop(key);n=min(len(old),len(kb));noise_mismatch+=int(np.any(old[:n]!=kb[:n]))
 else:pairs[key]=kb
summary=[]
for chain in ['TT','FF']:
 for eta in range(16):
  rr=[r for r in rows if r['chain']==chain and r['eta_index']==eta];av=[r for r in rr if r['available']]
  out=dict(chain=chain,eta_index=eta,eta=sc['eta'][eta],denominator=len(rr),available=len(av),outcomes=dict(Counter(r['termination'] for r in av)))
  for k in ['entry_initial','pass_initial','entry_visited','pass_visited','pass_then_held_exit_no_collision','entry_then_pass_held_exit','collision','numerical_failure','complete_record','safety_radius_respected']:out[k]=sum(r[k] for r in av)
  summary.append(out)
# Region-volume proof supplies exact integrals; numerical evaluation is supplementary.
conf=lambda z:(2*hh-abs(z))*.5*(2*aa-math.sqrt(Ds*Ds-z*z))**2
confvol,conferr=quad(conf,-2*hh,2*hh,epsabs=1e-13,points=[0])
trans=lambda y:2*(.2-math.sqrt(R*R-(.2-y)**2))
transarea,ae=quad(trans,hh,.2,epsabs=1e-13)
area=4*aa*hh+transarea+2*hh*(HH-.2)
volentry=.3*.05*.02*.04;volpass=.05*.02*.08*.99
vols=dict(entry_m4=volentry,pass_m4=volpass,conflict_domain_m4=confvol,conf_quad_abserr=conferr,entry_fraction_of_conflict=volentry/confvol,single_agent_area_m2=area,whole_joint_volume_lower=2*confvol,whole_joint_volume_upper=area*area,entry_fraction_of_whole_joint_bounds=[volentry/(area*area),volentry/(2*confvol)],pass_fraction_of_whole_joint_bounds=[volpass/(area*area),volpass/(2*confvol)],entry_min_wall_clearance=.00823333333333335,entry_min_pair_distance=.325,pass_min_wall_clearance=.003233333333333338,pass_min_pair_distance=.38)
# Exact membership is finite-sample descriptive evidence, never policy reachability probability.
seedstats=[]
for raw in b['raw_seed_fixed_family']:
 q=raw['z_min_m'];actual=sum((np.array(st['physical']['positions'])[1,1]-np.array(st['physical']['positions'])[0,1])>=q for st in states.values())
 seedstats.append(dict(raw_margin=raw['raw_margin'],z_threshold=q,width_ratio=q/(2*hh),main_uniform_geometry_fraction=(2*hh-q)**2/(8*hh*hh) if q<=2*hh else 0,actual_initial_z_exceeds=actual,denominator=24,warning='Only z-threshold arithmetic; raw formula valid solely for Agent B fixed x,y1 family'))
reference_domain=dict(x1=[-.705,-.59],x2=[.59,.705],y1=[-.02,.02],y2=[-.02,.02],volume_m4=.115**2*.04**2,reference_stage_steps_upper=[16,22,282,280,198],reference_total_steps_upper=798,reference_time_upper_s=39.9,status='PROVEN all-CBF reference existence by report inequalities; domain is explicitly chosen enclosing box, not asserted exact sampler support')
res=dict(reference_initial_domain=reference_domain,status='PROVEN finite-volume reference domains; EMPIRICALLY OBSERVED cached finite-panel coverage; UNKNOWN whole-domain actual TT/FF reachability',volumes=vols,initials=initials,summary=summary,raw_seed_comparison=seedstats,selected_tasks=len(selected),available_records=sum(r['available'] for r in rows),cached_position_records_read=positions_read,paired_noise_mismatches=noise_mismatch,unpaired_keys=len(pairs),max_saved_dynamics_error=dynamics_error,source_checks=source_checks,checkpoint=a['checkpoint_sha256'],noise_protocol=dict(future_root=sc['future_root'],runner='runner.py:20-23 SHA256 UID token fold root then seed then physical step',seed=0,paired_shared_prefix_keys_checked=True),new_rollouts=0,new_flow=0,new_projection=0,new_steps=0,new_training=0,read_file_count=len(reads),read_hashes=reads,wall_seconds=time.perf_counter()-START,cpu_seconds=time.process_time()-CPU)
(HERE/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
with (HERE/'coverage.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(next(r for r in rows if r['available']).keys()));w.writeheader();w.writerows(rows)
print(json.dumps({k:res[k] for k in ['volumes','available_records','cached_position_records_read','paired_noise_mismatches','max_saved_dynamics_error','read_file_count','wall_seconds']}))
print(json.dumps(summary,indent=2))
