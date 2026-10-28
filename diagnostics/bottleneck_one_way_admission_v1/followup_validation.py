"""Follow-up mechanism checks and independent exact acceleration certificates."""
from dispatch import *
from acceleration import accelerated_case
from validation import compare
from followup_runtime import Coordinator,DEFAULT
from types import SimpleNamespace

def units():
    cfg=SimpleNamespace(agent_radius=.1,max_speed=.5,goal_tolerance=.05,dt=.05,agent_collision_margin=0.)
    gate=dict(centre=[0.,0.],axis=[1.,0.],half_length=.5,width=.3)
    p=np.array([[-.9,0.],[.9,0.]]);g=-p;rem=np.ones(2)*1.8
    target=np.array([[.4,.1],[-.4,.1]])
    geom=SimpleNamespace(valid_points=lambda x:np.ones(len(x),bool))
    def call(state,pos=p,goal=g,remaining=rem,step=0,gates=None,filters=None):
        return state.apply(target,pos,goal,remaining,None,cfg,[gate]if gates is None else gates,filters or [],np.zeros_like(pos),step,geom)
    # Every wall-dependent scheme is exactly the old input with no gates.
    for sp in [dict(scheme='B',c=1.),dict(scheme='C',c=1.),dict(scheme='D',T=4.),dict(scheme='F',v_p=.2),dict(scheme='G',menu=DEFAULT)]:
        out,_=call(Coordinator(sp,0),gates=[]);assert out.tobytes()==target.tobytes()
    a=Coordinator(dict(scheme='B',c=1.),1);a.admission.states[0]['direction']=1
    out,_=call(a,pos=np.array([[-.2,0.],[.2,0.]]),goal=np.array([[1.,0.],[-1.,0.]]))
    assert np.array_equal(out[1],[.5,0.])
    a=Coordinator(dict(scheme='C',c=1.),1);a.admission.states[0]['direction']=1
    out,_=call(a,pos=np.array([[-.2,0.],[.2,0.]]),goal=np.array([[1.,0.],[-1.,0.]]))
    assert out.tobytes()==target.tobytes()
    a=Coordinator(dict(scheme='D',T=4.),1)
    out,_=call(a);assert np.array_equal(out[0],target[0])and np.array_equal(out[1],[0.,0.])
    out,h=call(a,pos=np.array([[-.2,0.],[.9,0.]]),step=80)
    assert h[0]['phase']=='clearing'and np.array_equal(out[0],target[0])and np.array_equal(out[1],[0.,0.])
    out,h=call(a,step=81);assert h[0]['direction']==-1 and h[0]['phase']=='green'
    # Head-on pair, shorter-route passer, connected same-heading team.
    a=Coordinator(dict(scheme='E',c=1.),0)
    p3=np.array([[-.1,0.],[.1,0.],[.3,0.]]);g3=np.array([[1.,0.],[-1.,0.],[-1.,0.]])
    filters=[SimpleNamespace(A=np.array([[1.,0.]]),lower=[0.],kinds=[['pair',0,1]])]+[SimpleNamespace(A=[],lower=[],kinds=[])for _ in range(2)]
    out,h=a.apply(np.ones((3,2))*.2,p3,g3,np.array([1.,2.,3.]),None,cfg,[],filters,np.zeros((3,2)),0,geom)
    assert h[0]['passer']==0 and h[0]['team']==[1,2]
    assert np.array_equal(out[1:],[ [.5,0.],[.5,0.]])and np.array_equal(out[0],[.2,.2])
    a=Coordinator(dict(scheme='F',v_p=.2),1);out,h=call(a)
    r=h[0]['reservations'];assert len(r)==2 and r[1]['enter']>=r[0]['leave']+1.-1e-12
    first=dict(r[0]);_,_=call(a,step=int((first['enter']+2.1)/cfg.dt))
    assert 0 not in a.reservations[0]
    call(a,step=int((first['enter']+2.1)/cfg.dt)+1);assert 0 in a.reservations[0]
    a=Coordinator(dict(scheme='G',menu=DEFAULT),1);a.admission.states[0]['direction']=1
    out,h=call(a,pos=np.array([[-.2,0.],[.7,0.]]),goal=np.array([[1.,0.],[.7,0.]]),remaining=np.array([1.,0.]))
    q=np.array(h[0]['waypoint_assignments'][0]['point'])
    assert abs(q[1])>=.2 and np.linalg.norm(out[1]-target[1])>0
    dump(HERE/'FOLLOWUP_UNIT_CHECK.json',dict(passed=True,environment_steps=0,schemes=list('BCDEFG'),checks=['wall-dependent no-passage bit identity','composite inverse exit vector','outside-only slowdown excludes inside agents','signal clearing preserves inside inputs then alternates','head-on team and shorter route passer','opposite reservations and missed-entry rebooking','valid separated waiting waypoint']))

def certify(family):
    scheme=family['scheme'];path=HERE/f'ACCELERATION_{scheme}.json'
    if path.exists():return accelerated_case if read(path)['passed']else case
    passage=read(HERE/'PASSAGES.json')['by_scene']
    scenes=[e['scene']for e in read(HERE/'SCENES.json')['entries']if passage[e['scene']['name']]['representative']and(scheme=='E'or passage[e['scene']['name']]['passages'])]
    pairs=[]
    for si in range(3):
        for sc in scenes:
            for ctl in controllers():
                spec=family['candidate_pool'][len(pairs)%3]
                pairs.append((case(sc,ctl,si,spec,purpose='followup_acceleration'),accelerated_case(sc,ctl,si,spec,purpose='followup_acceleration')))
    pairs=pairs[:50];assert len(pairs)==50
    a={r['rollout_uid']:r for r in execute([x for x,y in pairs],f'follow_{scheme}_original50',worker_entry='worker_recorded.py')}
    b={r['rollout_uid']:r for r in execute([y for x,y in pairs],f'follow_{scheme}_fast50',worker_entry='worker_accel.py')}
    checks=[compare(a[x['rollout_uid']],b[y['rollout_uid']])for x,y in pairs if x['rollout_uid']in a and y['rollout_uid']in b]
    passed=len(checks)==50 and all(x['equal']for x in checks)
    dump(path,dict(passed=passed,pairs=len(checks),checks=checks,original_seconds=sum(r['wall_seconds']for r in a.values()),fast_seconds=sum(r['wall_seconds']for r in b.values()),fallback='unaccelerated if not passed'))
    return accelerated_case if passed else case

def no_passage(family,make):
    scheme=family['scheme'];path=HERE/f'NO_PASSAGE_{scheme}.json'
    if scheme=='E':return False
    if path.exists():return read(path)['passed']
    passage=read(HERE/'PASSAGES.json')['by_scene'];scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
    refs=[x for x in read(HERE/'REPLAY_INPUTS.json')['records']if not passage[x['case']['scene']]['passages']][:20]
    assert len(refs)==20
    cases=[make(scenes[x['case']['scene']],x['case']['controller_payload']['nominal'],x['case']['state_index'],dict(x['case']['spec'],coordination=family['candidate_pool'][i%3]['coordination']),purpose='followup_no_passage')for i,x in enumerate(refs)]
    rows={r['rollout_uid']:r for r in execute(cases,f'follow_{scheme}_no_passage20')}
    checks=[compare(x['reference'],rows[c['rollout_uid']])for c,x in zip(cases,refs)if c['rollout_uid']in rows]
    passed=len(checks)==20 and all(x['equal']for x in checks)
    dump(path,dict(passed=passed,pairs=len(checks),checks=checks))
    return passed

if __name__=='__main__':units()
