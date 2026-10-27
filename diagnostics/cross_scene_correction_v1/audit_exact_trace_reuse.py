"""Read-only search for exactly repeated physical traces under distinct eta."""
from common import *
import numpy as np


def main():
    groups={};seen=set();count=0;skipped=0
    fields=('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance','clearance_types')
    for stage in('pilot','pilot_tail','main'):
        for path in(HERE/stage/'raw').glob('*.json'):
            r=read(path);c=r['case'];rid=r['rollout_uid']
            if rid in seen:continue
            seen.add(rid)
            if not c.get('cross_scene_new')or c['controller_payload']['nominal']['type_code']not in('prioritized_astar','conflict_based_search'):continue
            if r['numerical_failure']or r['collision']or c['theta']is not None:
                skipped+=1;continue
            base={k:v for k,v in c['controller_payload'].items()if k not in('method','eta_b','theta')}
            key=uid('physical_input',dict(scenario=c['scenario_uid'],state=c['state_uid'],seed=c['seed_key'],
                rng_state_uid=c.get('rng_state_uid',c['state_uid']),controller=base))
            trace=Path(r['trace_path']);assert sha(trace)==r['trace_sha256']
            digest=hashlib.sha256()
            with np.load(trace,allow_pickle=False)as arrays:
                for field in fields:
                    value=arrays[field]
                    digest.update(canonical(dict(field=field,shape=value.shape,dtype=str(value.dtype))).encode())
                    digest.update(value.tobytes())
            group=groups.setdefault((key,digest.hexdigest()),[])
            group.append(dict(rollout_uid=rid,requested_action=c['requested_action'],eta_uid=c['eta_uid'],
                scene=c['state']['scene'],state_index=c['state']['index'],controller=c['controller_payload']['nominal']['controller'],
                raw=str(path),raw_sha256=sha(path),trace=str(trace),trace_sha256=r['trace_sha256'],
                episode_length=r['episode_length'],outcome=r['outcome']))
            count+=1
    duplicates=[dict(input_fingerprint=k[0],physical_trace_sha256=k[1],records=v)
        for k,v in groups.items()if len({r['eta_uid']for r in v})>1]
    write(HERE/'EXACT_PHYSICAL_TRACE_REUSE_SEARCH.json',dict(utc=utc(),read_only=True,passed=True,
        eligible_completed_records=count,skipped_unknown_collision_or_switch=skipped,
        input_groups=len({k[0]for k in groups}),duplicate_physical_trace_groups=len(duplicates),duplicates=duplicates,
        searched_records=[r for records in groups.values()for r in records],
        exact_fields=fields,comparison='dtype, shape and array bytes; no approximate matching.',
        caveat='Physical trace equality is an opportunity audit only. New eta safety certificates may differ; no new candidate label inferred or inserted by this search.',
        new_physical_steps=0,official_journal_writes=0))
    print('Exact physical trace search:',count,'eligible records;',len(duplicates),'cross-eta duplicate groups.',flush=True)


if __name__=='__main__':main()
