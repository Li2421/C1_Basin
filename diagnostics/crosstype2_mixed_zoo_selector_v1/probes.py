"""Reuse outcome-free probe evidence, then collect only missing probe arms."""
from common import *
from zoo_design import profiles,case,freeze
from history_audit import signature
from generator_reference import normalized


def design():
    base=profiles();old=D/'crosstype1_learned_zoo_selector_v1'
    states=read(old/'locked_probe_protocol.json')['states'];assert len(states)==16
    p={**base,'states':states,'phase':'probe'};rows=[];cps={};reused=[];manifest=[]
    inventory={}
    for stage in ('old_probe','new_probe','locked_probe'):
        path=old/f'{stage}_CASES.json'
        if not path.exists():continue
        for c in read(path):
            raw=old/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():inventory[c['rollout_uid']]=raw
    allowed=set(read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')['controller_order'])
    for cohort in ('DEVELOPMENT','PPO','LOCKED'):
        allowed.update(read(HERE/f'{cohort}_CAPABILITY.json')['eligible'])
    p['controller_order']=[n for n in base['controller_order'] if n in allowed]
    for name in p['controller_order']:
        for si,s in enumerate(states):
            for index in (10,18,17,13):
                c,cp=case(base,name,s,index,probe=True);c['state_index']=si
                raw=inventory.get(c['rollout_uid'])
                if raw is not None:
                    record=read(raw);assert sha(record['trace_path'])==record['trace_sha256']
                    assert record['case']['state_uid']==s['state_uid'] and record['case']['base_controller_name']==name
                    assert not any(k in record for k in ('success','outcome','termination'))
                    reused.append(dict(case=c,raw_path=str(raw),raw_sha256=sha(raw),trace_path=record['trace_path'],trace_sha256=record['trace_sha256'],complete=record['complete'],physical_steps=record['physical_steps']))
                    manifest.append(dict(case=c,source='historical_exact',raw_path=str(raw),raw_sha256=sha(raw)))
                else:
                    rows.append(c);cps[cp['controller_uid']]=cp
                    manifest.append(dict(case=c,source='new_missing',raw_path=str(HERE/'probe/raw'/f"{c['rollout_uid']}.json")))
    # Compatible historical 200-step CEM probes can be extended from stored state.
    prefixes=[];source=D/'blackbox_families_v1';stage='probe_only_pilot_centralized_sampling'
    source_protocol=read(source/f'{stage}_protocol.json');source_states={s['state_uid']:s for s in source_protocol['states']}
    wanted={(c['base_controller_name'],signature(next(s for s in states if s['state_uid']==c['state_uid'])),c['arm']) for c in rows}
    for c in read(source/f'{stage}_CASES.json'):
        name=c['base_controller_name'];arm='F10' if c['eta_index']==10 else 'F13';s=source_states[c['state_uid']]
        if (name,signature(s),arm) not in wanted:continue
        raw=source/stage/'raw'/f"{c['rollout_uid']}.json"
        if not raw.exists():continue
        r=read(raw);assert sha(r['trace_path'])==r['trace_sha256']
        # These are unchanged original CEM nominal profiles and TT semantics.
        assert source_protocol['profiles'][name]['payload']==base['profiles'][name]['payload']
        prefixes.append(dict(controller=name,arm=arm,state_signature=signature(s),placement='TT',checkpoint_sha256=base['profiles'][name]['payload'].get('flow_checkpoint_sha256'),raw_path=str(raw),raw_sha256=sha(raw),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],old_physical_steps=r['physical_steps']))
    write(HERE/'PROBE_PREFIX_INVENTORY.json',dict(utc=utc(),prefixes=prefixes,unchanged_nominal_profiles=True,execution_semantics='TT; original RNG, state, eta, termination off; stored trajectories and known errors retained'))
    write(HERE/'PROBE_REUSE.json',dict(utc=utc(),reused=reused,new_requests=len(rows),prefixes_reused=len(prefixes),old_three_arms_not_reexecuted=True))
    write(HERE/'PROBE_MANIFEST.json',dict(utc=utc(),states=states,controllers=p['controller_order'],entries=manifest))
    freeze('probe',p,rows,cps,split='probe',task_success_labels=False)
    print('Probes:',len(reused),'exact historical records;',len(prefixes),'stored prefixes;',len(rows),'new arms')


if __name__=='__main__':globals()[sys.argv[1]]()
