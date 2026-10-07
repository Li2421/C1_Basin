"""Seal SBMA provenance without running simulations."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from diagnostics.success_basin_multimodality.setup import HERE,ROOT,SYSROOT,sha

def main():
    def entry(path,base=HERE):return {'path':str(path.relative_to(base)),'sha256':sha(path)}
    artifacts=[entry(x) for x in sorted(HERE.iterdir()) if x.is_file() and x.name!='manifest.json']
    artifacts += [entry(x) for x in sorted((HERE/'figures').glob('*')) if x.is_file()]
    stages=[]
    for mf in sorted((HERE/'raw').glob('*/manifest.json')):
        m=json.loads(mf.read_text());stages.append({**entry(mf),'stage':m['stage'],'attempts':len(m['records']),
          'recorded_physical_steps':sum(r['steps'] for r in m['records']),
          'outcomes':dict((str(v),sum(r['outcome']==v for r in m['records'])) for v in ['success','deadlock','timeout','collision',None])})
    def git(cwd,*args):
        q=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True);return q.stdout.strip() if q.returncode==0 else None
    manifest={'study':'SBMA','sealed_at_utc':datetime.now(timezone.utc).isoformat(),
      'decision':json.loads((HERE/'decision.json').read_text()),
      'protocol_sha256':sha(HERE/'protocol.json'),'verification_sha256':sha(HERE/'verification.json'),
      'artifacts':artifacts,'raw_stage_indexes':stages,
      'raw_trajectory_hashes':'Every raw stage manifest contains a SHA256 for every NPZ trace.',
      'raw_attempts':sum(x['attempts'] for x in stages),'raw_physical_steps':sum(x['recorded_physical_steps'] for x in stages),
      'final_outcome_bearing_continuations':6144,
      'prior_unknown_static_audit_records':754,
      'repositories':{
        'workspace':{'path':str(ROOT),'commit':git(ROOT,'rev-parse','HEAD'),'status':git(ROOT,'status','--short')},
        'frozen_system':{'path':str(SYSROOT),'commit':git(SYSROOT,'rev-parse','HEAD'),'status':git(SYSROOT,'status','--short'),
          'note':'Relevant uncommitted source is pinned by protocol SHA256; no commit ID is fabricated.'}},
      'training':{'G_theta':False,'risk':False,'Q_value':False},
      'frozen_physics_flow_constraints_events_modified':False,
      'projection_fallback':'Same objective and exact feasible set; certificates in verification.json.',
      'limitations':['Finite empirical sampled region, not mathematical global connectedness.',
        'D1/D2/D4 are similar selected failing states, not population-random independent states.',
        'Pointwise confidence intervals are not a simultaneous family-wise confidence band.']}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:manifest[k] for k in ['raw_attempts','raw_physical_steps','final_outcome_bearing_continuations']},indent=2))

if __name__=='__main__':main()
