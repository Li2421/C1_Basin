"""Explicit completion separates DB physical records from local trace equivalence."""
from common import *
import subprocess

def main(stage):
    # Single official merger invocation only; unlabeled probe records remain unlabeled.
    result=subprocess.check_output([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,text=True);print(result,flush=True)
    rows=read(HERE/f'{stage}_CASES.json');aliaspath=HERE/f'{stage}_LOCAL_EQUIVALENCE.json';aliases=read(aliaspath) if aliaspath.exists() else {};missing=[];unknown=0;physical=0
    for path in (HERE/stage/'dynamic_aliases').glob('*.json'):aliases[path.stem]=read(path)
    write(HERE/f'{stage}_ALL_LOCAL_EQUIVALENCE.json',aliases)
    if stage!='probe':
        orphans=[]
        with rollout_db.connect(True) as db:
            for c in rows:
                rid=c['rollout_uid'];raw=HERE/stage/'raw'/f'{rid}.json'
                if rid in aliases or not raw.exists() or db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rid,)).fetchone():continue
                r=read(raw);assert r['rollout_uid']==rid and 'outcome' in r and sha(r['trace_path'])==r['trace_sha256'];orphans.append(r)
        if orphans:
            cache_writer.append_journal(orphans,EXP,stage+'_completed_raw_orphan_recovery')
            print(subprocess.check_output([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,text=True),flush=True)
        write(HERE/f'{stage}_RAW_JOURNAL_RECOVERY.json',dict(utc=utc(),completed_records_rejournaled=[r['rollout_uid'] for r in orphans],physical_rollouts_rerun=0))
    with rollout_db.connect(True) as db:
        for c in rows:
            rid=c['rollout_uid'];raw=HERE/stage/'raw'/f'{rid}.json'
            if stage=='probe':
                if not raw.exists():missing.append(rid)
                else:
                    r=read(raw);unknown+=int(not r.get('complete'));physical+=1
            elif rid in aliases:unknown+=int(aliases[rid]['outcome']['numerical_failure'])
            else:
                r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                if r is None:missing.append(rid)
                else:unknown+=int(r['numerical_failure']);physical+=int(raw.exists())
    write(HERE/f'{stage}_COMPLETENESS.json',dict(utc=utc(),requests=len(rows),actual_physical=physical,local_equivalence=len(aliases),unknown=unknown,missing=missing,complete=not missing))
    if missing:raise RuntimeError(f'{len(missing)} unfinished {stage} identities; preserve completed/unknown and resubmit unfinished only')
    seal(f'{stage}_COMPLETE.receipt.json',[HERE/f'{stage}_COMPLETENESS.json',HERE/f'{stage}_EXECUTION_FROZEN.receipt.json',HERE/f'{stage}_ALL_LOCAL_EQUIVALENCE.json'],existing_unknown_retained=True)

if __name__=='__main__':main(sys.argv[1])
