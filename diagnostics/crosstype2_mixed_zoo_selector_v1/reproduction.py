"""Outcome-blind selection of 48 mandatory exact historical runtime replays."""
from common import *


def design():
    prereg();rows=[]
    old=D/'crosstype1b_generator_critic_pipeline_v1'
    pool=[]
    for stage in ('pilot','remaining'):
        pool.extend((c,stage) for c in read(old/f'{stage}_CASES.json'))
    for family in ('Flow','gaussian','diffusion','ring_ippo'):
        available=[(c,stage) for c,stage in pool if c['family']==family]
        # Two states, both fixed and progress candidates; no outcome selection.
        chosen=sorted(available,key=lambda r:(r[0]['state_index'],r[0]['candidate_index'],r[0]['base_controller_name']))
        names=list(dict.fromkeys(c['base_controller_name'] for c,_ in chosen))
        for name in names[:2]:
            for action in (0,1):
                c,stage=next((c,st) for c,st in chosen if c['base_controller_name']==name and c['candidate_index']==action)
                rows.append(dict(group='crosstype1b',family=family,case=c,protocol=str(old/f'{stage}_protocol.json'),reference=str(old/stage/'raw'/f"{c['rollout_uid']}.json")))
    old=D/'blackbox_families_v1';cases=read(old/'main_CASES.json')
    for family in ('centralized_sampling','cross_entropy'):
        available=[c for c in cases if c['family']==family and c['seed']==0]
        for instance in range(2):
            for arm in ('safety_baseline','F10','Sprog','RHR'):
                c=next(c for c in available if c['base_controller_name']==f'{family}{instance}' and c['arm']==arm)
                rows.append(dict(group='round38',family=family,case=c,protocol=str(old/'main_protocol.json'),reference=str(old/'main/raw'/f"{c['rollout_uid']}.json")))
    old=D/'blackbox_addendum_v1';cases=read(old/'right_hand_evaluation_CASES.json')
    available=[c for c in cases if c['kind']=='DM' and c['H']==700 and c['r']==.9 and c['seed']==0]
    for instance in range(4):
        for c in [c for c in available if c['base_controller_name']==f'DM{instance}'][:4]:
            rows.append(dict(group='round37',family='distributed_shooting',case=c,protocol=str(old/'right_hand_evaluation_protocol.json'),reference=str(old/'right_hand_evaluation/raw'/f"{c['rollout_uid']}.json")))
    assert len(rows)==48
    for row in rows:
        ref=read(row['reference']);assert sha(ref['trace_path'])==ref['trace_sha256']
        row['reference_sha256']=sha(row['reference'])
    write(HERE/'REPRODUCTION_CASES.json',rows)
    manifest=HERE/'reproduction_PLANNED_ROLLOUTS.json'
    write(manifest,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in rows]))
    write(HERE/'reproduction_CACHE_PREFLIGHT.json',planner.preflight(manifest))
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/'REPRODUCTION_CASES.json',manifest,HERE/'reproduction_CACHE_PREFLIGHT.json'],mandatory_replays_only=True,new_labels=0)


def run(worker):
    from generated_execution import execute
    for index,row in enumerate(read(HERE/'REPRODUCTION_CASES.json')):
        if index%2!=worker:continue
        target=HERE/'reproduction/checks'/f'{index:02}.json'
        if target.exists():continue
        assert sha(row['reference'])==row['reference_sha256']
        reference=read(row['reference']);p=read(row['protocol']);c=dict(row['case'])
        if row['group']=='round37':
            p['profiles']={c['base_controller_name']:p['profiles'][c['profile_key']]}
            c['placement']='TT'
        result=execute(c,p,{},'reproduction')
        write(HERE/'reproduction/raw'/f'{index:02}.json',result)
        with np.load(reference['trace_path'],allow_pickle=False) as a,np.load(result['trace_path'],allow_pickle=False) as b:
            x,y=a['positions'],b['positions'];positions=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
        first=(reference['first_action'] is None and result['first_action'] is None) or (reference['first_action'] is not None and result['first_action'] is not None and np.asarray(reference['first_action'],np.float64).tobytes()==np.asarray(result['first_action'],np.float64).tobytes())
        checks=dict(group=row['group'],family=row['family'],controller=c['base_controller_name'],first_action_bytes=first,positions_bytes=positions,termination=reference['outcome']==result['outcome'],steps=reference['episode_length']==result['episode_length'],jax_enable_x64=result['jax_enable_x64'])
        checks['passed']=all(checks[k] for k in ('first_action_bytes','positions_bytes','termination','steps'))
        write(target,checks);print(json.dumps(checks),flush=True)


def collect():
    checks=[read(p) for p in sorted((HERE/'reproduction/checks').glob('*.json'))];assert len(checks)==48
    groups={g:all(c['passed'] for c in checks if c['group']==g) for g in ('crosstype1b','round38','round37')}
    write(HERE/'EXECUTION_REPRODUCTION.json',dict(utc=utc(),passed=all(groups.values()),group_passed=groups,checks=checks,count=48,new_task_labels=0))


if __name__=='__main__':
    if sys.argv[1]=='run':run(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
