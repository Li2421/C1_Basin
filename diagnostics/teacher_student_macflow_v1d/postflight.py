from core import *
from shared_rollout_db.src.rollout_db import connect,canonical
from shared_rollout_db.src.planner import preflight
import subprocess

def merge(phase):
    p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_CASES.json');missing=read(HERE/f'{phase}_MISSING_CASES.json')
    assert all((HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() for c in missing)
    cmd=[sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',p['experiment_uid']];r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True);(HERE/f'{phase}_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0
    with connect(True) as db:
        found=[]
        for c in rows:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None and not row['conflict_quarantined'],c;assert row['compatibility_quality']=='EXACT_REUSE'
            rawpath=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            if rawpath.exists():
                raw=read(rawpath);assert all(raw[k]==row[k] for k in ['success','deadlock','timeout','collision','numerical_failure','episode_length']);assert sha(raw['trace_path'])==raw['trace_sha256']
            found.append(dict(row))
    pf=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');unknown={r['rollout_uid'] for r in found if r['numerical_failure']};assert pf['summary']['genuinely_missing']==len(unknown) and pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_POSTFLIGHT.json',pf);write(HERE/f'{phase}_DB_ROWS.json',found);write(HERE/f'{phase}_MERGE_RECEIPT.json',dict(utc=utc(),records=len(found),unknown=len(unknown),db_exact=True,missing_equals_preserved_unknown=True,official_command=cmd))
    if phase=='pilot':
        names=p['controller_order'];eligible=[];excluded=[];rates={}
        for n in names:
            rates[n]={}
            for arm in ['safety baseline','S500']:
                cases=[c for c in rows if c['base_controller_name']==n and c['arm']==arm];byid={r['rollout_uid']:r for r in found};rs=[byid[c['rollout_uid']] for c in cases];rates[n][arm]=dict(lower=sum(r['success'] and not r['numerical_failure'] for r in rs)/16,upper=sum(r['success'] or r['numerical_failure'] for r in rs)/16,unknown=sum(r['numerical_failure'] for r in rs))
            a,b=[rates[n][arm]['lower'] for arm in ['safety baseline','S500']]
            if (a<.15 and b<.15) or (a>.98 and b>.98):excluded.append(dict(name=n,reason='both_low' if a<.15 else 'both_high'))
            else:eligible.append(n)
        write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),eligible=eligible,excluded=excluded,rates=rates,study_status='INCONCLUSIVE' if len(eligible)+2<8 else 'CONTINUE',unknown_rule='lower/upper preserved; gate decisions use preregistered lower bound, no reruns'))
    print(phase,'postflight',len(found),'unknown',len(unknown),flush=True)
if __name__=='__main__':merge(sys.argv[1])
