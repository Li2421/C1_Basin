"""Use prescribed gate rollouts for timing; never inspect success/termination."""
from common import *
def main():
    cases=read(HERE/'capability_CASES.json');rows=[]
    for kind in ('transformer_bc','gradient_mpc','conflict_based_search'):
        selected=[c for c in cases if c['family']==kind][:4];values=[]
        for c in selected:
            path=HERE/'capability/raw'/f"{c['rollout_uid']}.json"
            if not path.exists():return False
            r=read(path);values.append(dict(rollout_uid=c['rollout_uid'],elapsed_seconds=r['elapsed_seconds'],physical_steps=r['physical_steps']))
        seconds=np.asarray([r['elapsed_seconds'] for r in values]);rows.append(dict(type=kind,records=values,mean_seconds=float(seconds.mean()),median_seconds=float(np.median(seconds)),max_seconds=float(seconds.max())))
    write(HERE/'NEW_TYPE_TIMING.json',dict(utc=utc(),rows=rows,source='First4 preplanned capability cases per type; no additional execution',outcome_fields_accessed=False,constructor_startup_excluded_from_runtime_timer=True,priority='Unseen primary evaluation before known-type regression'))
    seal('NEW_TYPE_TIMING.receipt.json',[HERE/'NEW_TYPE_TIMING.json',HERE/'timing.py',HERE/'PROMPT_REVISION_ACTIVE.receipt.json'],no_correction_outcomes_emitted=True)
    print(json.dumps([dict(type=r['type'],mean_seconds=r['mean_seconds'],max_seconds=r['max_seconds']) for r in rows]),flush=True)
    return True
if __name__=='__main__':main()
