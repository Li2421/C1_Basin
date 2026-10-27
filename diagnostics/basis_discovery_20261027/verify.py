"""Verify all candidate identities, frozen execution sources, cache and trace files."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,sys,platform
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src import rollout_db as db


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    rows=list(map(json.loads,open(OUT/'candidates.jsonl')));assert len({r['rollout_uid']for r in rows})==len(rows)
    sources={};traces=set();plans={};stages=[]
    for path in OUT.glob('*_planned.json'):
        plan=json.load(path.open());stages.append(path.name)
        for c in plan['exact_configs']:
            if c['rollout_uid']in plans:assert plans[c['rollout_uid']]['controller_payload']==c['controller_payload']
            plans[c['rollout_uid']]=c
    assert set(plans)=={r['rollout_uid']for r in rows}
    with db.connect(True)as con:
        for r in rows:
            c=plans[r['rollout_uid']];assert c['spec']==r['spec']
            assert db.eta_identity(c['eta_a'])[0]==r['eta_uid']
            assert r['rollout_uid']==db.uid('roll',dict(state=r['state_uid'],eta=r['eta_uid'],controller=r['controller_uid'],seed=r['seed_key']))
            q=con.execute('SELECT success,collision,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert q and q['success']==r['success']and q['collision']==r['collision']and q['numerical_failure']==r['numerical_failure']and not q['conflict_quarantined']
            st=con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(r['state_uid'],)).fetchone()
            assert json.loads(st[0])=={k:c['state'][k]for k in ('positions','velocities','goals')}
            src=c['controller_payload']['dictionary_sources'];assert src==dict(basis_dictionary=sha(ROOT/'shared_control/basis_dictionary.py'),runner=sha(OUT/'runner.py'))
            assert not r['success']or not(r['collision']or r['numerical_failure'])
            if r.get('trace'):assert Path(r['trace']).exists();traces.add(r['trace'])
    core={'shared_control/basis_dictionary.py':sha(ROOT/'shared_control/basis_dictionary.py')}
    core.update({p.name:sha(p)for p in OUT.glob('*.py')})
    result=dict(passed=True,rows=len(rows),new_rollouts=sum(r['source']=='new_rollout'for r in rows),exact_prior_reuses=sum(r['source']=='exact_prior_behavior_reuse'for r in rows),
        unique_existing_traces=len(traces),cache_conflicts=0,numerical_failures=sum(r['numerical_failure']for r in rows),stages=stages,
        execution_sources_match=True,code_sha256=core,raw_candidates_sha256=sha(OUT/'candidates.jsonl'),
        raw_candidates_bytes=(OUT/'candidates.jsonl').stat().st_size,python=sys.version,platform=platform.platform(),
        packages={n:version(n)for n in ('numpy','scipy','jax','jaxlib','clarabel')},
        resource='CPU only; Slurm cpu partition,24 available CPUs allocated,worker affinity0..23,BLAS1; Slurm accounting storage disabled.',
        data_backup='Raw JSONL/NPZ/SQLite remain local; Git commit is not a backup of ignored raw evidence.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k not in ('code_sha256','stages')})

if __name__=='__main__':main()
