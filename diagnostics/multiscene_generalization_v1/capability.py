"""Frozen navigation and multi-agent gates; never replace excluded instances."""
from common import *
from design_generated import identities,ARMS
from registry import register
import math

def checkpoints(family):
    return [HERE/'policies'/f'{family}_{kind}_{seed}'/'best.msgpack' for kind in ['deterministic','mixture'] for seed in [17,23]]

def navigation(family,valid=False):
    cfg=frozen_config();catalog=read(HERE/('CATALOG_FINAL.json' if valid else 'CATALOG_FROZEN.json'))['catalog'];states=sorted([s for s in catalog[family] if s['role']=='evaluation'],key=lambda s:s['index']);rows=[];registered=[]
    for checkpoint in checkpoints(family):
        assert (checkpoint.parent/'COMPLETE.receipt.json').exists()
        for map_index in range(8):
            for episode in range(2):
                s=states[4*map_index+episode];i=(2*map_index+episode)%4
                path=np.asarray(s['paths'][i]);H=int(math.ceil(2.5*np.linalg.norm(np.diff(path,axis=0),axis=1).sum()/.5/.05/100)*100)
                solo=dict(s,role='navigation_capability',alias=f"{s['alias']}_single_agent{i}",positions=[s['positions'][i]],goals=[s['goals'][i]],velocities=[[0.,0.]],H=H)
                solo['config']=dict(s['config'],num_agents=1,max_steps=H)
                c,r=identities(solo,'learned','nominal_only',0,checkpoint,H);rows.append(c);registered+=r
    phase=('navigation_valid_' if valid else 'navigation_')+family
    register(phase,rows,registered)
    seal(f'{phase.upper()}_FROZEN.receipt.json',[HERE/'capability.py',HERE/'generated_runtime.py']+checkpoints(family),threshold=.9,episodes_per_instance=16,evaluation_geometry_only_for_gate=True,before_gate=True,valid_holdout=valid)

def navigation_result(family,valid=False):
    phase=('navigation_valid_' if valid else 'navigation_')+family
    cases=read(HERE/f'{phase}_CASES.json');result=[]
    for checkpoint in checkpoints(family):
        rows=[c for c in cases if c['checkpoint']==str(checkpoint.resolve())];raw=[read(HERE/phase/'raw'/f"{c['rollout_uid']}.json") for c in rows]
        success=sum(r['success'] for r in raw);unknown=sum(r['numerical_failure'] for r in raw)
        lower=success/16;upper=(success+unknown)/16
        result.append(dict(name=checkpoint.parent.name,checkpoint=str(checkpoint),success=success,collision=sum(r['collision'] for r in raw),timeout=sum(r['timeout'] for r in raw),unknown=unknown,episodes=16,lower=lower,upper=upper,passed=lower>=.9,gate_uncertain=lower<.9<=upper))
    output=f'NAVIGATION_VALID_CAPABILITY_{family}.json' if valid else f'NAVIGATION_CAPABILITY_{family}.json'
    write(HERE/output,dict(family=family,instances=result,eligible=[v['name'] for v in result if v['passed']],family_passed=sum(v['passed'] for v in result)>=2))
    return result

def pilot(family):
    gate=read(HERE/f'NAVIGATION_VALID_CAPABILITY_{family}.json');catalog=read(HERE/'CATALOG_FINAL.json')['catalog'];rows=[];registered=[]
    for name in gate['eligible']:
        checkpoint=HERE/'policies'/name/'best.msgpack'
        for s in catalog[family]:
            if s['role']!='pilot':continue
            for arm in ['safety_baseline','F10','S13']:
                c,r=identities(s,'learned',arm,0,checkpoint,s['H']);rows.append(c);registered+=r
    if rows:register('pilot_gate_'+family,rows,registered)
    return rows

def pilot_result(family):
    nav=read(HERE/f'NAVIGATION_VALID_CAPABILITY_{family}.json');rows=read(HERE/f'pilot_gate_{family}_CASES.json') if (HERE/f'pilot_gate_{family}_CASES.json').exists() else [];out=[]
    for name in nav['eligible']:
        stats={}
        for arm in ['safety_baseline','F10','S13']:
            cases=[c for c in rows if Path(c['checkpoint']).parent.name==name and c['arm']==arm];raw=[read(HERE/f'pilot_gate_{family}'/'raw'/f"{c['rollout_uid']}.json") for c in cases]
            success=sum(r['success'] for r in raw);unknown=sum(r['numerical_failure'] for r in raw);stats[arm]=dict(success=success,unknown=unknown,episodes=16,lower=success/16,upper=(success+unknown)/16)
        passed=max(v['lower'] for v in stats.values())>=.2 and not all(v['lower']>.98 for v in stats.values())
        reason='eligible' if passed else 'all three arms exceed98%' if all(v['lower']>.98 for v in stats.values()) else 'maximum observed lower success below20%'
        out.append(dict(name=name,arms=stats,passed=passed,reason=reason))
    eligible=[v['name'] for v in out if v['passed']]
    write(HERE/f'MULTIAGENT_CAPABILITY_{family}.json',dict(family=family,instances=out,eligible=eligible,family_passed=len(eligible)>=2))
    return eligible

if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
