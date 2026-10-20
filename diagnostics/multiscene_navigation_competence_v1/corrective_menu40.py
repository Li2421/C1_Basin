"""Original frozen states; compatible capability arms reused by exact identity."""
from common import *
from corrective_design40 import require_ready,identities
from collect39 import checkpoint
from design_generated import ARMS
from registry import register

def prepare(family,phase):
    assert phase in ['pilot_full','probe','evaluation']
    require_ready(family)
    gate=read(HERE/f'MULTIAGENT_CAPABILITY_{family}.json');assert gate['family_passed']
    role='pilot' if phase=='pilot_full' else 'probe' if phase=='probe' else 'evaluation'
    states=[s for s in read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][family] if s['role']==role]
    assert len(states)==(32 if role=='evaluation' else 16)
    cases=[];rows=[]
    for name in gate['eligible']:
        f,kind,seed=name.split('_');assert f==family;cp=checkpoint(f,kind,int(seed),3)
        for s in states:
            for arm in (['F10','F13'] if phase=='probe' else ARMS[2:] if phase=='pilot_full' else ARMS):
                for seed in ([0,1] if phase=='evaluation' else [0]):
                    c,r=identities(s,arm,seed,cp,purpose='probe' if phase=='probe' else 'task')
                    c['controller_name']=name;cases.append(c);rows+=r
    stage=f'{phase}40_{family}';register(stage,cases,rows)
    seal(f'{stage}_INPUTS_FROZEN.receipt.json',[HERE/f'{stage}_CASES.json',HERE/f'{stage}_PREFLIGHT.receipt.json',HERE/'ROUND39_CATALOG_FROZEN.json',HERE/'corrective_menu40.py',HERE/'corrective_design40.py'],original_states_unchanged=True,role=role,G_locked_until_AF_and_final_model=True)
    return stage

def probe_postflight(stage):
    from shared_rollout_db.src.rollout_db import connect
    cases=read(HERE/f'{stage}_CASES.json');index=[];features={};incomplete=[]
    with connect(True) as db:
        # Outcome-free journals are officially indexed as SUMMARY_ONLY.
        ledger=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=? AND classification=?',(EXP,'SUMMARY_ONLY'))]
    journal_ids=set()
    for source in ledger:
        path=Path(source['path']);assert path.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
        assert sha(path)==source['sha256']
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r.get('schema')=='round40_unlabeled_probe_v1':journal_ids.add(r['case']['rollout_uid'])
    for c in cases:
        path=HERE/stage/'raw'/f"{c['rollout_uid']}.json";assert path.exists()
        r=read(path);assert not r['terminal_labels_read'] and not any(k in r for k in ['success','outcome','termination'])
        assert sha(r['trace_path'])==r['trace_sha256'] and c['rollout_uid'] in journal_ids
        values=r['statistics'] if r['complete'] else [np.nan]*5
        features.setdefault(c['controller_name'],[]).append(values)
        if not r['complete']:incomplete.append(c['rollout_uid'])
        index.append(dict(raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256'],complete=r['complete'],rollout_uid=c['rollout_uid']))
    for name,values in features.items():assert len(values)==32
    with np.errstate(invalid='ignore'):means={name:np.nanmean(values,axis=0) for name,values in features.items()}
    p=HERE/f'{stage}_FEATURES.json';write(p,dict(features=means,index=index,incomplete_preserved=incomplete,unknown_retries=0,no_task_labels=True))
    write(HERE/f'{stage}_POSTFLIGHT.json',dict(planner=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),artifact_evidence_class='SUMMARY_ONLY',all_requested_artifacts_indexed=True,task_rollout_rows_expected=0,incomplete_preserved=incomplete))
    seal(f'{stage}_FEATURES_FROZEN.receipt.json',[p,HERE/f'{stage}_POSTFLIGHT.json'],before_family_evaluation_labels_opened=True,source_only_preprocessing=True)

def run_probe(stage):
    from submit_cpu40 import submit
    from submit_stage import completed
    import subprocess,time
    if (HERE/f'{stage}_POSTFLIGHT.json').exists():return
    p=HERE/f'{stage}_SUBMISSION.json';job=read(p)['job'] if p.exists() else None
    while job is None:
        job=submit(stage,18,script='probe_measurements40.py')
        if job is None:time.sleep(5)
    while not completed(job):time.sleep(5)
    with (HERE/f'{stage}_MERGE.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],check=True,stdout=log)
    probe_postflight(stage)

if __name__=='__main__':prepare(sys.argv[1],sys.argv[2])
