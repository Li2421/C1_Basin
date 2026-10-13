"""Verify referenced evidence, freeze a delivery manifest, or verify that manifest."""
from pathlib import Path
import argparse,json,hashlib,datetime
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
 if a.verify:
  m=read(HERE/'MANIFEST.json')
  for row in m['files']:
   f=HERE/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],f
  print({'verified_files':len(m['files']),'bytes':m['logical_bytes']});return
 required=['REPORT.md','HANDOFF.md','PREREGISTRATION.md','results.json','RUNBOOK.md','FINAL_AUDIT.json','CHECKPOINT_REPLAY_gpu_recorded.json','REPLAY_SENSITIVITY_recorded.json']
 assert all((HERE/p).exists() for p in required)
 r=json.loads((HERE/'results.json').read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 assert not r['original_strong_state_selection_goal_met'] and r['matching_precision_eta_choices_reproduced']
 assert r['audit']['passed'] and r['cost']['new_physical_rollouts']==7172
 for key in ['A. Task Overview','B. Implementation Summary','C. Experimental Results','D. Hypothesis Evaluation','E. Generalization Diagnosis','F. Limitations','G. Reproducibility']:assert key in (HERE/'REPORT.md').read_text()
 replay=read(HERE/'CHECKPOINT_REPLAY_gpu_recorded.json');assert len(replay['checks'])==138 and not any(row['argmax_changed_all_cells'] for row in replay['checks'])
 assert read(HERE/'REPLAY_SENSITIVITY_recorded.json')['summary']['max_ensemble_success_change']==0
 # Verify every indexed historical or new row/prediction without trusting the index.
 references={}
 for f in (HERE/'data').glob('*index.json'):
  for row in read(f):
   path=Path(row['path']).resolve();assert sha(path)==row['sha256'],path
   old=references.setdefault(str(path),row['sha256']);assert old==row['sha256']
 ext=[{'path':p,'sha256':h} for p,h in sorted(references.items()) if not Path(p).is_relative_to(HERE)]
 (HERE/'EXTERNAL_INPUTS.json').write_text(json.dumps(ext,indent=2)+'\n')
 check={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'delivery_artifacts_present':True,'scientific_primary_goal_met':False,'raw_cache_audit_passed':True,'matching_precision_checkpoint_argmax_reproduced':138,'indexed_evidence_unique_files_verified':len(references),'external_source_files_verified':len(ext),'strict_Q_near_equality_failures':sum(not row['within_strict_tolerance'] for row in replay['checks']),'notes':'Artifact completion is not scientific strong-selection validation. No original data deletion or modification.'}
 (HERE/'DELIVERY_CHECK.json').write_text(json.dumps(check,indent=2)+'\n')
 skip={'MANIFEST.json','bundle_receipt.json'};files=[]
 for f in sorted(HERE.rglob('*')):
  rel=f.relative_to(HERE)
  if not f.is_file() or '__pycache__' in rel.parts or str(rel) in skip:continue
  files.append({'path':str(rel),'bytes':f.stat().st_size,'sha256':sha(f)})
 manifest={'schema':'csg_delivery_manifest_v1','utc':check['utc'],'files':files,'logical_bytes':sum(x['bytes'] for x in files),'scope':'This result directory except manifest itself, bundle receipt and generated __pycache__; external inputs verified separately, not copied. SQLite not backed up by this manifest.'}
 (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(check);print({'files':len(files),'logical_MiB':manifest['logical_bytes']/2**20})
if __name__=='__main__':main()
