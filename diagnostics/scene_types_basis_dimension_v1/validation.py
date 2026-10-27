"""B3 integration regression against the original production run function."""
from dispatch import *
from discovery import specs_s1
from equivalence import KEYS,OUTCOMES

def original_case(c):
    c=json.loads(json.dumps(c));c['controller_payload']['validation_engine']='literal_original_RN_run'
    c['controller_payload']['validation_worker_sha256']=sha(HERE/'worker_original.py')
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));return c

def main():
    if(HERE/'B3_EQUIVALENCE.json').exists():assert read(HERE/'B3_EQUIVALENCE.json')['passed'];return
    s=next(e['scene']for e in read(HERE/'SCENES_FROZEN.json')['entries']if e['type_id']==1 and 'scene'in e)
    cs=[];pairs=[]
    for ctl in controllers():
        for i in range(4):
            for sp in [specs_s1()[0],specs_s1()[1]]:
                c=case(s,ctl,i,sp,force_route=False,purpose='B3_equivalence');orig=original_case(c);cs.extend([c,orig]);pairs.append([c['rollout_uid'],orig['rollout_uid']])
    rows={r['rollout_uid']:r for r in execute(cs,'B3_equivalence',worker_entry='worker_original.py')};checks=[]
    for a,b in pairs:
        if a not in rows or b not in rows:checks.append(dict(a=a,b=b,equal=False,reason='incomplete'));continue
        with np.load(rows[a]['trace'])as x,np.load(rows[b]['trace'])as y:
            arrays={k:x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes()for k in KEYS}
        outcomes={k:rows[a][k]==rows[b][k]for k in OUTCOMES}
        checks.append(dict(a=a,b=b,equal=all(arrays.values())and all(outcomes.values()),arrays=arrays,outcomes=outcomes))
    passed=all(x['equal']for x in checks);dump(HERE/'B3_EQUIVALENCE.json',dict(passed=passed,pairs=len(checks),checks=checks,meaning='Two nonzero historical B3 settings, four search states, both frozen nominal controllers. No new column; original literal run versus private route adapter with routing disabled.'))
    if not passed:raise RuntimeError('B3 integration regression failed; main search must not proceed')

if __name__=='__main__':main()
