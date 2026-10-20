"""Read-only history compatibility inventory. No new identities or rollouts."""
from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

ARMS=['safety_baseline','F10','F13','S100','S300','S500','Sprog','RHR']


def signature(state):
    return canonical(dict(uid=state['uid'],physical={k:state['physical'][k] for k in ('positions','velocities','goals')}))


def main():
    prereg()
    p38=read(D/'blackbox_families_v1/main_protocol.json')
    p33=read(D/'axis2_controller_v1/NEW_BASE_PROTOCOL.json')
    original=read(D/'blackbox_addendum_v1/ORIGINAL_PROFILES.json')
    flow={name:cp for name,cp in original.items() if cp['payload'].get('controller_class','Flow')=='Flow'}
    profiles=dict(flow)
    profiles.update({f'G{i}':p33['profiles'][f'G{i}'] for i in range(6)})
    profiles.update({name:cp for name,cp in p38['profiles'].items() if name.startswith(('deterministic','mixture','diffusion'))})
    assert len(flow)==14 and len(profiles)==38
    p=dict(profiles=profiles,states=[],eta=p38['eta'],native_manifest=p38['native_manifest'])
    write(HERE/'HISTORICAL_BASE_PROTOCOL.json',p)
    candidates={};source_files={};rejected=[]
    phases={
      'blackbox_families_v1':['main','source_training','tune_pilot_deterministic','gate_flow_corrected'],
      'blackbox_addendum_v1':['geometry_controls','right_hand_tune','right_hand_evaluation'],
      'axis2_controller_v1':['main','baseline'],
    }
    rule=dict(mode='progress',steps=40,speed=.35)
    for folder,stages in phases.items():
        for stage in stages:
            h=D/folder;casefile=h/f'{stage}_CASES.json';protocolfile=h/f'{stage}_protocol.json'
            if not casefile.exists() or not protocolfile.exists():
                continue
            cases=read(casefile);protocol=read(protocolfile)
            states={s['state_uid']:s for s in protocol['states']}
            source_files[str(casefile)]=sha(casefile);source_files[str(protocolfile)]=sha(protocolfile)
            for c in cases:
                name=c.get('base_controller_name');arm=c.get('arm')
                if name not in profiles or arm not in ARMS+['nominal_only']:
                    continue
                if c.get('H',700)!=700 or c.get('r',.9)!=.9:
                    continue
                if arm=='RHR' and c.get('rhr_rule')!=rule:
                    continue
                if c['state_uid'] not in states:
                    continue
                # A case's base checkpoint must be the exact current zoo instance.
                cp=protocol.get('profiles',{}).get(c.get('profile_key',name),{})
                if cp and cp.get('payload',{}).get('flow_checkpoint_sha256')!=profiles[name]['payload'].get('flow_checkpoint_sha256'):
                    continue
                key=(name,signature(states[c['state_uid']]),arm,c['seed'])
                # Keep all exact registered candidates; cache decides existence.
                candidates.setdefault(key,[]).append(dict(case=c,casefile=str(casefile),protocol=str(protocolfile),
                                                          raw_path=str(h/stage/'raw'/f"{c['rollout_uid']}.json")))
    cases=[];reuse=[];missing=[];requests=[]
    with rollout_db.connect(True) as db:
        for split,names,seeds in [('evaluation',['nominal_only']+ARMS,[0,1]),('tuning',ARMS,[0])]:
            states=read(HERE/('EVAL_STATES.json' if split=='evaluation' else 'TUNING_STATES.json'))
            for name in profiles:
                for si,s in enumerate(states):
                    for arm in names:
                        for seed in seeds:
                            key=(name,signature(s),arm,seed)
                            found=None
                            for source in candidates.get(key,[]):
                                c=source['case']
                                row=db.execute('SELECT rollout_uid FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                                if row is not None:
                                    found=source;break
                            if found is not None:
                                c=found['case'];reuse.append(dict(split=split,controller=name,arm=arm,source=found))
                            else:
                                # Explicit provisional missing identity, not registered or executable.
                                # Execution design must replace it with a protected-semantics controller identity.
                                ctl=uid('crosstype1_provisional',dict(base=profiles[name]['controller_uid'],arm=arm,
                                          H=700,unregistered_planning_only=True))
                                eta=[0.,0.,0.] if arm in ('nominal_only','safety_baseline','RHR') else p['eta'][10 if arm=='F10' else 13]
                                eid=eta_identity(eta)[0];sk=canonical(dict(future_index=seed))
                                c=dict(state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk)
                                missing.append(dict(split=split,controller=name,arm=arm,state_index=si,seed=seed))
                            requests.append({k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')})
    path=HERE/'history_inventory_PLANNED_ROLLOUTS.json';write(path,dict(requests=requests))
    pf=planner.preflight(path);write(HERE/'history_inventory_CACHE_PREFLIGHT.json',pf)
    counts={split:sum(x['split']==split for x in missing) for split in ('evaluation','tuning')}
    counts['mechanism_probe']=38*16*3
    write(HERE/'HISTORY_REUSE_CANDIDATES.json',reuse)
    write(HERE/'HISTORY_MISSING_INVENTORY.json',missing)
    write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),source_files=source_files,old_controllers=len(profiles),
          development_type_counts=dict(Flow=14,gaussian=6,deterministic=6,mixture=6,diffusion=6),
          historical_reusable_cases=len(reuse),missing_from_explicit_historical_casefiles=counts,
          missing_total=sum(counts.values()),official_cache_summary=pf['summary'],
          inventory_only=True,new_identities_registered=False,
          limitation='Provisional missing IDs cannot establish global DB absence; full execution-identity design and global DB preflight still required before rollout.',
          probe_gap='Historical R5 is 200 steps and lacks nominal/velocity/swept arrays; cannot silently extend to outcome-free 300 steps.',
          reuse_rule='A critic feature change does not invalidate old execution labels; existing unknown rows are retained and not resimulated.'))
    print(json.dumps(dict(reused=len(reuse),missing=counts,total=sum(counts.values()))))

if __name__=='__main__':main()
