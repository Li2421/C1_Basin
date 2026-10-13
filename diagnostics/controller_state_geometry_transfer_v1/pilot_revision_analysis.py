"""Outcome-isolated, pre-registered pilot gate. Reads no test-state outcome."""
from pathlib import Path
import json,hashlib,itertools,datetime
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'pilot_revision'
W=[.41,.47,.53,.59];C=['flow_isotropic','flow_anisotropic'];E=[0,2,4,5,8,10,12,15]
def center(q,axes):
 for ax in axes:q=q-q.mean(ax,keepdims=True)
 return q

def components(q):
 out={}
 for mask in range(1,16):
  axes=tuple(i for i in range(4) if mask&(1<<i));other=tuple(i for i in range(4) if i not in axes)
  out['×'.join('GSCE'[i] for i in axes)]=center(q.mean(other,keepdims=True) if other else q,axes)
 return out

def stats(y):
 a=y[...,0::2].mean(-1);b=y[...,1::2].mean(-1)
 fa,fb=components(a),components(b);energies={k:float(np.mean(fa[k]*fb[k])) for k in fa}
 sa,sb=center(a,(1,3)),center(b,(1,3));energy=float(np.mean(sa*sb))
 def selected(q,v):
  ix=q.argmax(-1)[...,None];return np.take_along_axis(v,ix,-1)[...,0]
 gain=[]
 for q,v in [(a,b),(b,a)]:
  gain.append(selected(q,v)-selected(np.broadcast_to(q.mean(1,keepdims=True),q.shape),v))
 total=sum(energies.values());fraction=sum(v for k,v in energies.items() if 'S' in k)/total if total>0 else float('nan')
 return {'anova_signal':energies,'total_signal':total,'state_fraction':fraction,'state_eta_energy':energy,'oracle_adaptive_gain':float(np.mean(gain)),
         'state_eta_reliability':float(np.corrcoef(sa.ravel(),sb.ravel())[0,1]),'mean_success':float(y.mean())}

def main():
 receipt=json.loads((HERE/'preregistration_receipt.json').read_text());assert hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest()==receipt['sha256']
 cases=[c for c in json.loads((SOURCE/'cases.json').read_text()) if c['width'] in W and c['state_index']<8]
 y=np.full((4,8,2,8,4),np.nan);audit=[];nf=[];missing=[]
 for c in cases:
  p=SOURCE/'raw'/(c['rollout_uid']+'.json')
  if not p.exists():missing.append(str(p));continue
  raw=p.read_bytes();r=json.loads(raw);assert r['case']==c
  audit.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
  if r['numerical_failure']:nf.append(str(p));continue
  y[W.index(c['width']),c['state_index'],C.index(c['controller_name']),E.index(c['eta_index']),c['seed']]=r['success']
 assert not missing,('incomplete pilot',len(missing));assert not nf,('pilot unknown',len(nf));assert np.isfinite(y).all()
 out=stats(y);rng=np.random.default_rng(20261011);boots=[stats(y[:,rng.integers(0,8,8)]) for _ in range(2000)]
 for k in ['state_fraction','state_eta_energy','oracle_adaptive_gain']:
  out[k+'_ci95']=np.nanpercentile([b[k] for b in boots],[2.5,97.5]).tolist()
 point=(out['state_fraction']>=.2 and out['state_eta_energy']>=.005 and out['oracle_adaptive_gain']>=.05)
 reliable=out['state_eta_energy_ci95'][0]>0 and out['oracle_adaptive_gain_ci95'][0]>0
 out.update(passed=bool(point and reliable),point_thresholds_passed=bool(point),precision_extension_allowed=bool(point and not reliable),n_rollouts=len(cases),unknown=len(nf),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preregistration_sha256=receipt['sha256'],final_test_outcomes_read=False)
 (HERE/'pilot_revision_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');(HERE/'data'/'pilot_revision_evidence_index.json').write_text(json.dumps(audit,indent=2)+'\n');np.savez_compressed(HERE/'data'/'pilot_revision.npz',Y=y)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
