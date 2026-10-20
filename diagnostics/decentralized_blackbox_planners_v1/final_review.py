from common import *
from audit_utils import source_inventory
G=guard()
def main():
 delivery=read(HERE/'DELIVERY.receipt.json');assert delivery['passed'];assert all(sha(p)==h for p,h in delivery['files'].items());assert read(HERE/'FINAL_VERIFICATION.json')['passed'];assert read(HERE/'INTEGRITY_COMPARISON.json')['passed'];assert read(HERE/'ASTAR_SEED_IDENTITY.json')['pairs']==672
 summary={}
 for phase,n in [('pilot',576),('tune',6912),('main',6720)]:
  rows=read(HERE/f'{phase}_DB_ROWS.json');assert len(rows)==n;assert len({r['rollout_uid'] for r in rows})==n;summary[phase]=dict(rows=n,unknown=sum(r['numerical_failure'] for r in rows))
 assert summary['main']['unknown']==1
 assert len(read(HERE/'ANALYSIS_MANIFEST.json'))==12992
 groups=read(HERE/'results.json')['groups'];assert [len(groups[f]['controllers']) for f in ['DM','AS','G','V0','ALL']]==[6,3,6,14,29]
 assert abs(groups['V0']['arms']['S500']['success']['lower']['estimate']-844/896)<1e-15
 write(HERE/'FINAL_REVIEW.json',dict(utc=utc(),passed=True,phase_counts=summary,authorized_replays=32,unlabeled_probes=576,student_records_reused=6272,astar_seed_pairs=672,all_delivery_hashes_valid=True,new_rollouts_after_completion=0,guard=guard()));source_inventory('imports_final_review.json')
 required=['DELIVERY.receipt.json','FINAL_REVIEW.json','FINAL_VERIFICATION.json','INTEGRITY_BEGIN.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','EXTERNAL_GIT_STATUS.json','SOURCE_START_COVERAGE.json','IMPORTED_SOURCE_BEGIN.json','IMPORTED_SOURCE_END.json','PREREGISTRATION.md','PREREGISTRATION.receipt.json','PREREGISTRATION_PATH_AMENDMENT.json','REPORT.md','HANDOFF.md','MECHANISM_CLOSEOUT.md','PAPER_TABLE.md','PAPER_TABLE.csv','results.json','COPY_PROVENANCE.json','VENDOR_PROVENANCE.json','COPY_DIFF.patch','EXECUTION_WRAPPER_AUDIT.md','CAPABILITY_GATE.csv','CAPABILITY_EXCLUDED_LOW.csv','CAPABILITY_EXCLUDED_HIGH.csv','RHR_GRID_TABLE.md','RHR_GRID_TABLE.csv','RHR_SELECTED.json','FROZEN_CHOICES.json','CONSTANT_CHOICES.json','GAPS.json','RESOURCE_SUMMARY.json']
 seal('FINAL_REVIEW.receipt.json',[HERE/p for p in required]+sorted((HERE/'figures').glob('*')),passed=True,guard=guard())
 print('FINAL REVIEW PASS')
if __name__=='__main__':main()
