from common import *
from registry import connect
import subprocess,time

def main():
 phase='gap_timing';cases=read(HERE/f'{phase}_CASES.json');assert all((HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() for c in cases)
 while any(' -m shared_rollout_db.src.merger ' in s and 'python' in s for s in subprocess.check_output(['ps','-eo','args='],text=True).splitlines()):time.sleep(5)
 r=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True);(HERE/f'{phase}_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 with connect(True) as db:
  rows=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=? AND path LIKE ?',(EXP,'%/gap_timing_%'))];assert len(rows)==len(cases) and all(r['classification']=='SUMMARY_ONLY' and r['rows_imported']==0 for r in rows)
  assert all(not db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() for c in cases)
 pf=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/'gap_timing_CACHE_POSTFLIGHT.json',dict(preflight=pf,intentional_summary_only_no_rollout_labels=True,source_rows=rows));seal('gap_timing_COMPLETE.receipt.json',[HERE/'gap_timing_CACHE_POSTFLIGHT.json',HERE/'gap_timing_MERGE.log'],new_timing_only_cases=len(cases),not_retried_as_missing_labels=True)
if __name__=='__main__':main()
