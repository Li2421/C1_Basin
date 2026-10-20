"""Seal completed evidence without weakening the strict failed acceptance gate."""
from support import *
G=guard()
def main():
 path=HERE/'QUALIFIED_DELIVERY.receipt.json'
 if path.exists():
  for p,h in read(path)['files'].items():assert sha(p)==h,p
  print('Qualified delivery hashes verified; overall protocol NOT passed.');return
 strict=read(HERE/'FINAL_VERIFICATION.json');failed=[k for k,v in strict['checks'].items() if not v]
 assert not strict['passed'] and set(failed)=={'resources_within_cap','protected_integrity'},failed
 assert not (HERE/'DELIVERY.receipt.json').exists()
 audit=read(HERE/'INTEGRITY_COMPARISON.json');assert not audit['changed'] and audit['new_external_files']
 assert read(HERE/'IMPORTED_SOURCE_AUDIT.json')['passed']
 results=read(HERE/'results.json');assert results['unknown_all_tasks']==0
 for receipt in HERE.rglob('*.receipt.json'):
  for p,h in read(receipt).get('files',{}).items():assert sha(p)==h,(receipt,p)
 write(HERE/'CLOSEOUT_STATUS.json',dict(utc=utc(),experiment_complete=True,records=23360,six_arm_records=22272,unknown=0,overall_protocol_passed=False,failed_checks=failed,statistics=results['endpoints'],protected_existing_files_changed=0,unclassified_external_new_sources=len(audit['new_external_files']),all_imported_sources_unchanged=True,source_protection_gate='Strict end audit failed; no further experiments or outcome analysis after detection; finalization is reporting and read-only verification only.',resource_gate='Peak total CPU 24; late intervention disclosed; STOP/CONT preserved episodes without restart.',original_sealed_analysis_and_plots_unchanged=True,renderer_amendment='PLOTTING_AMENDMENT.json',guard=guard()))
 files=set()
 for pattern in ['*.json','*.jsonl','*.md','*.csv','*.patch','*.npz','*.py']:
  files.update(HERE.glob(pattern))
 files.update((HERE/'figures').glob('*'))
 files.update((HERE/'vendor').rglob('*.py'))
 seal('QUALIFIED_DELIVERY.receipt.json',sorted(p for p in files if p.is_file()),passed=False,overall_protocol_passed=False,completed_evidence=True,confirmatory_statistical_endpoints_passed=all(results['endpoints'][k]['status']=='PASS' for k in ['confirmation1','confirmation2']),failed_checks=failed,full_compliance_receipt_issued=False,guard=guard())
 print('Completed evidence sealed; overall protocol NOT passed:',failed,flush=True)
if __name__=='__main__':main()
