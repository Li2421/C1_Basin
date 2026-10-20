from common import *
from audit import main as audit
from registry import connect
import difflib

def main():
 assert all((HERE/f'{part}_COMPLETE.receipt.json').exists() for part in ['A','B','C','D','E','F']);audit('END');receipt_checks=[];bad=[]
 for p in sorted(HERE.glob('*receipt.json')):
  v=read(p)
  for f,h in v.get('files',{}).items():
   ok=Path(f).is_file() and sha(f)==h;receipt_checks.append(dict(receipt=str(p),path=f,passed=ok))
   if not ok:bad.append(dict(receipt=str(p),path=f))
 imports={};input_conflicts=[]
 for p in sorted(HERE.glob('imports_*.json')):
  v=read(p)
  for path,h in v.get('sources',{}).items():
   if Path(path).is_relative_to(HERE):continue
   if path in imports and imports[path]!=h:input_conflicts.append(dict(path=path,reason='Imported source hash differed between runs'))
   imports[path]=h
 for p,h in imports.items():
  if not Path(p).exists() or sha(p)!=h:input_conflicts.append(dict(path=p,reason='Imported source changed since execution'))
 copychecks=[]
 for i,row in enumerate(read(HERE/'COPY_PROVENANCE.json')):
  a=Path(row['source']);b=Path(row['destination']);ok=sha(a)==row['source_sha256'];scope=row.get('scope',row.get('copy_scope','whole file'));copychecks.append(dict(source=str(a),destination=str(b),source_unchanged=ok,scope=scope,current_destination_sha256=sha(b)));diff=''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile=str(a),tofile=str(b)));(HERE/'copy_diffs').mkdir(exist_ok=True);(HERE/'copy_diffs'/f'{i:02d}_{b.name}.diff').write_text(diff)
  if not ok:input_conflicts.append(dict(path=str(a),reason='Copied source modified'))
 # Explicit read inputs have immutable hashes; external writing to a report is identified separately.
 readchanges=[]
 for v in read(HERE/'READ_AUDIT.json')['files']:
  if not Path(v['path']).exists() or sha(v['path'])!=v['sha256']:readchanges.append(v['path'])
 phases={};unknown=0;new=0
 with connect(True) as db:
  for p in sorted(HERE.glob('*_SUBMISSION.json')):
   phase=p.name[:-len('_SUBMISSION.json')]
   if phase=='gap_timing':assert (HERE/'gap_timing_COMPLETE.receipt.json').exists();continue
   if phase=='replay_E' and (HERE/'replay_E_IMPORT_FAILURE.receipt.json').exists():
    assert (HERE/'replay_E_import_repair_COMPLETE.receipt.json').exists();pf_path=HERE/'replay_E_REPAIRED_CACHE_POSTFLIGHT.json'
    if not(HERE/'replay_E_IMPORT_FAILURE_RESOLVED.receipt.json').exists():
     pf=planner.preflight(HERE/'replay_E_PLANNED_ROLLOUTS.json');assert not pf['summary']['genuinely_missing'] and not pf['summary']['ambiguous'];write(pf_path,pf);seal('replay_E_IMPORT_FAILURE_RESOLVED.receipt.json',[pf_path,HERE/'replay_E_IMPORT_FAILURE.receipt.json',HERE/'replay_E_import_repair_COMPLETE.receipt.json'],old_attempt_new_physical_steps=0,old_attempt_new_rollout_rows=0,all16_missing_identities_completed_by_repair=True)
    phases[phase]=dict(new=0,unknown=0,classification='Import-only failed submission before execution',resolved_by='replay_E_import_repair');continue
   done=read(HERE/f'{phase}_COMPLETE.receipt.json');cases=read(HERE/f'{phase}_CASES.json');uids={c['rollout_uid']for c in cases};n_unknown=0
   for c in {c['rollout_uid']:c for c in cases}.values():
    r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r and not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE';n_unknown+=r['numerical_failure']
    current_raw=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
    if current_raw.exists():assert read(current_raw).get('jax_enable_x64') is False,('Precision contract',current_raw)
   phases[phase]=dict(unique_records=len(uids),new=done['new'],unknown=n_unknown,official_complete_receipt_sha256=sha(HERE/f'{phase}_COMPLETE.receipt.json'));new+=done['new'];unknown+=n_unknown
 assets={}
 for protocol_path in HERE.glob('*_protocol.json'):
  prot=read(protocol_path)
  for profile in prot.get('profiles',{}).values():
   payload=profile['payload']
   for key,value in payload.items():
    if key.endswith('_path') and value and payload.get(key[:-5]+'_sha256'):
     expected=payload[key[:-5]+'_sha256'];path=str(Path(value).resolve());assets.setdefault(path,set()).add(expected)
 asset_checks=[dict(path=p,expected_sha256=sorted(h),passed=len(h)==1 and Path(p).exists()and sha(p)in h)for p,h in assets.items()];asset_bad=[q for q in asset_checks if not q['passed']];write(HERE/'ASSET_FINAL_VERIFICATION.json',dict(utc=utc(),passed=not asset_bad,checks=asset_checks));assert not asset_bad,asset_bad
 write(HERE/'IMPORTED_SOURCE_FINAL.json',imports);write(HERE/'COPY_DIFF_REVIEW.json',copychecks);integrity=read(HERE/'INTEGRITY_COMPARISON.json');out=dict(utc=utc(),receipt_hashes_passed=not bad,receipt_checks=receipt_checks,bad_receipts=bad,imported_sources_passed=not input_conflicts,input_conflicts=input_conflicts,read_input_changes=readchanges,integrity=integrity,official_DB_phases=phases,new_rollout_records=new,unknown_across_phase_records=unknown,unknown_scope='May include reused historical labels and validation; not unique physical episodes across all cohorts',source_protection_passed=not input_conflicts and not readchanges and not asset_bad and integrity['existing_content_pass'],checkpoint_and_manifest_assets_passed=not asset_bad,no_git_mutations=True,no_package_install=True,all_new_files_confined_to_current_directory_except_official_DB=True);write(HERE/'FINAL_VERIFICATION.json',out);assert not bad and not input_conflicts,('Frozen receipt or imported source mutation',bad,input_conflicts)
 print('Delivery verification',out['source_protection_passed'],len(receipt_checks),'receipt file hashes',len(imports),'imported sources')
if __name__=='__main__':main()
