from common import *
from registry import connect
from audit_utils import source_inventory
import subprocess,time

def main(phase):
    cases=read(HERE/f'{phase}_CASES.json');new=read(HERE/f'{phase}_MISSING_CASES.json');assert all((HERE/phase/'raw'/f"{c['rollout_uid']}.json").is_file() for c in new)
    while True:
        others=[line for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines() if ' -m shared_rollout_db.src.merger ' in line and 'python' in line]
        if not others:break
        write(HERE/'MERGER_WAIT.json',dict(utc=utc(),other_mergers=others));time.sleep(5)
    cmd=[sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP];r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True);(HERE/f'{phase}_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    found=[]
    with connect(True) as db:
        for c in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None,c['rollout_uid'];assert not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE'
            path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            if path.exists():
                raw=read(path);assert all(raw[k]==row[k] for k in ['success','collision','timeout','deadlock','numerical_failure','episode_length']);assert sha(raw['trace_path'])==raw['trace_sha256']
            found.append(dict(row))
    pf=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');unknown={r['rollout_uid'] for r in found if r['numerical_failure']};assert pf['summary']['ambiguous']==0 and pf['summary']['genuinely_missing']==len(unknown);write(HERE/f'{phase}_DB_ROWS.json',found);write(HERE/f'{phase}_CACHE_POSTFLIGHT.json',pf);seal(f'{phase}_COMPLETE.receipt.json',[HERE/f'{phase}_DB_ROWS.json',HERE/f'{phase}_CACHE_POSTFLIGHT.json',HERE/f'{phase}_MERGE.log'],records=len(found),new=len(new),preserved_unknown=len(unknown),DB_exact=True);source_inventory(f'imports_postflight_{phase}.json');print(phase,'merged',len(found),'unknown',len(unknown),flush=True)
if __name__=='__main__':main(sys.argv[1])
