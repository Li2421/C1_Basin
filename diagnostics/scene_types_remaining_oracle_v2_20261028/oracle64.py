"""Freeze nested finite pools; extend only failures, stopping at first success."""
import os
os.environ['STB_COORDINATOR']='1'
from core import *
import collections,datetime,functools
from scipy.stats import qmc
from shared_control.basis_dictionary import ALL,NEW,SIGNED
import runner
from acceleration import accelerated_case

OLD=HERE.parent/'scene_types_basis_dimension_v1'
runner.sources=functools.lru_cache(maxsize=1)(runner.sources)

def spec_key(sp):return db.canonical(sp)
def case_key(r):return (r['type_id'],r['controller'],r['state_index'])

def sample(columns,n,seed,goal_low=.5):
    cols=[x for x in ALL if x in columns]
    points=qmc.Sobol(len(cols),scramble=True,seed=seed).random_base2(6)
    result=[]
    for point in points[:n]:
        values={}
        for col,u in zip(cols,point):
            lo,hi=(goal_low,1.25) if col=='G' else (-.5,.5) if col=='P' else (0.,.75) if col=='R' else (-1.,1.) if col in SIGNED else (0.,1.)
            values[col]=float(lo+(hi-lo)*u)
        result.append(dict(eta=[values.get(x,0.) for x in ALL[:3]],coefficients={x:values[x] for x in NEW if x in values}))
    return result

def freeze():
    path=HERE/'ORACLE64_FROZEN.json'
    if path.exists():return read(path)
    original=read(OLD/'FAMILIES_FROZEN.json')['families'];ranges=read(OLD/'GOAL_RANGE_FROZEN.json')['pools']
    families=[]
    for j,f in enumerate(original):
        anchor=ranges['narrow32'] if j==0 else f['candidate_pool']
        seed=2026106400+j
        extra=sample(f['columns'],64-len(anchor),seed)
        pool=anchor+extra
        assert len(pool)==len({spec_key(sp) for sp in pool})==64
        assert pool[:16]==f['candidate_pool']
        families.append(dict(index=str(j),columns=f['columns'],dimension=f['dimension'],pool=pool,seed=seed,goal_low=.5 if 'G' in f['columns'] else None,kind='primary_family_extension',original_fixed=f['fixed']))
    # Matched G-range control: retain the existing paired32, then use the
    # exact same latent32/P/R as primary B3 for the tail.
    tail=[]
    for sp in families[0]['pool'][32:]:
        eta=sp['eta'];u=(eta[0]-.5)/.75
        tail.append(dict(eta=[1.25*u,eta[1],eta[2]],coefficients={}))
    families.append(dict(index='B3_wide',columns=['G','P','R'],dimension=3,pool=ranges['wide32']+tail,seed=families[0]['seed'],goal_low=0.,kind='matched_goal_range_control',original_fixed=original[0]['fixed']))
    value=dict(freeze_time=datetime.datetime.now().astimezone().isoformat(),families=families,K=64,original_families_sha256=sha(OLD/'FAMILIES_FROZEN.json'),original_range_sha256=sha(OLD/'GOAL_RANGE_FROZEN.json'),rule='User-requested posthoc budget extension. Fixed global pool per family, original16 retained. Only failed groups extend. Stop at first successful candidate; otherwise exhaust64 and stop. Numerical unknowns retained separately. No Generator/training/phase schedules.',statistics='First successful candidate uses one-based pool index. Report success-conditional mean and full-cohort restricted search count (64 on failure). These differ from actual new rollouts/cache hits and from elapsed simulation time.',new_attempt_limit=20000,original_45000_cap_superseded='The user explicitly requested64 instead of16 after the original study completed; new phase has its own finite bound. No continuation beyond64.')
    dump(path,value);return value

def build():
    protocol=freeze();labels=read(OLD/'HOLDOUT_LABELS.json');ranges=read(OLD/'GOAL_RANGE_LABELS.json')
    if (HERE/'ORACLE64_GROUPS.json').exists() and (HERE/'ORACLE64_ALL_planned.json').exists():
        # Recovery must preserve the original evidence paths and request set;
        # discovering newly completed raw files must not rewrite the plan.
        groups=read(HERE/'ORACLE64_GROUPS.json')['groups']
        planned=read(HERE/'ORACLE64_ALL_planned.json')['exact_configs']
        _,_,known_results=runner.register(planned,'ORACLE64_ALL')
        for c in planned:
            target=HERE/'raw'/f"{c['rollout_uid']}.json"
            if c['rollout_uid']in known_results and not target.exists():
                # A worker can be interrupted between journal append and raw
                # write. Recover the already-complete official result.
                dump(target,known_results[c['rollout_uid']])
        return protocol,groups,{c['rollout_uid']:c for c in planned}
    uidset={x['uid'] for rows in list(labels.values())+list(ranges.values()) for x in rows}
    old_rows={uid:read(OLD/'raw'/f'{uid}.json') for uid in uidset}
    known={}
    for r in old_rows.values():
        if r.get('safety_layer')=='distributed' and r['controller'] in ('centralized_sampling','gradient_mpc') and not r['spec'].get('reference'):
            known.setdefault((case_key(r),spec_key(r['spec'])),dict(uid=r['rollout_uid'],path=str(OLD/'raw'/f"{r['rollout_uid']}.json"),source='frozen_v1'))
    for p in (HERE/'raw').glob('*.json'):
        r=read(p)
        known.setdefault((case_key(r),spec_key(r['spec'])),dict(uid=r['rollout_uid'],path=str(p),source='v2_completed'))
    scenes={e['type_id']:e['scene'] for e in read(OLD/'SCENES_FROZEN.json')['entries'] if e.get('scene')}
    ctls={c['type_code']:c for c in controllers()}
    groups=[];requests={}
    for f in protocol['families']:
        source_labels=ranges['wide32'] if f['index']=='B3_wide' else labels[f['index']]
        keys=sorted({case_key(x) for x in source_labels})
        assert len(keys)==120
        for key in keys:
            g=dict(family=f['index'],columns=f['columns'],dimension=f['dimension'],type_id=key[0],controller=key[1],state_index=key[2],candidates=[])
            for j,sp in enumerate(f['pool']):
                item=dict(candidate=j+1,spec=sp)
                cached=known.get((key,spec_key(sp)))
                if cached:
                    r=read(cached['path'])
                    item.update(cached)
                    assert r['spec']==sp
                else:
                    c=accelerated_case(scenes[key[0]],ctls[key[1]],key[2],sp,purpose='oracle64_failed_group_extension')
                    requests[c['rollout_uid']]=c
                    item.update(uid=c['rollout_uid'],path=str(HERE/'raw'/f"{c['rollout_uid']}.json"),source='new_missing')
                g['candidates'].append(item)
                if cached and r['success']:
                    # Existing successes stop the actual sequential query.
                    break
            groups.append(g)
    # Every new UID is novel for this exact semantic context, even though the
    # private-path namespace produces a different code fingerprint from v1.
    planned=list(requests.values())
    assert len(planned)<20000
    dump(HERE/'ORACLE64_GROUPS.json',dict(groups=groups,total_groups=len(groups),maximum_new_unique_requests=len(planned)))
    _,missing,done=runner.register(planned,'ORACLE64_ALL')
    dump(HERE/'ORACLE64_READY.json',dict(total_groups=len(groups),maximum_new_unique_requests=len(planned),genuinely_missing=len(missing),exact_reused=len(done),new_attempt_cap=20000,group_sha256=sha(HERE/'ORACLE64_GROUPS.json')))
    return protocol,groups,{c['rollout_uid']:c for c in planned}

def main():
    from dispatch64 import execute_groups
    p,g,c=build()
    execute_groups(p,g,c)

if __name__=='__main__':main()
