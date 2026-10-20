"""Finite controller-independent proposals and method-specific admissible pools."""
from common import *
from actions_c6 import action
from schedule_c6 import THRESHOLDS,TRIGGERS

METHODS=(
 '学习双 η·进度阈值学出（不看 controller）',
 '学习双 η·进度阈值学出·在线停滞触发',
 '学习双 η·进度阈值学出·在线停滞触发·离线探测特征',
 '学习双 η·固定阈值 0.968（不看 controller）',
 '学习双 η·进度阈值学出·在线停滞触发·单一触发设置',
 '学习双 η·critic 判修不修·扩充提议',
 '人定双 η·按进度切换',
 '只加安全投影')

def physical_key(a):
    if not a.get('c6_schedule',True):
        from reference_candidate_menu import physical_key as oldkey
        return ('legacy',*oldkey(a))
    canonical=action(a['eta_a'],a['eta_b'],a.get('progress_threshold',.968),a.get('trigger','none'),a.get('late_source','given'))
    return (tuple(canonical['eta_a']),() if canonical['late_source']=='requery' else tuple(canonical['eta_b']),canonical['switch_mode'],canonical['progress_threshold'],canonical['trigger'],canonical['late_source'])

def menu(proposals):
    rows=[];seen=set()
    for trigger in TRIGGERS:
        for i in range(8):
            for j in range(8):
                for theta in THRESHOLDS:
                    a=action(proposals[i],proposals[j],theta,trigger);key=physical_key(a)
                    if key in seen:continue
                    seen.add(key);rows.append(a)
        for i in range(8):
            for theta in THRESHOLDS:rows.append(action(proposals[i],proposals[0],theta,trigger,'requery'))
    rows.append(action([0.,0.,0.],[0.,0.,0.],0.,name='safety'))
    assert len(rows)==1313 and len({physical_key(a) for a in rows})==1313
    return rows

def pick(scores,allowed,zero):
    scores=np.asarray(scores);allowed=np.asarray(allowed,bool);assert allowed.any()
    best=int(np.argmax(np.where(allowed,scores,-np.inf)))
    return best

def pools(actions):
    zero=np.asarray([not np.any(a['eta_a']) and not np.any(a['eta_b']) and a.get('late_source','given')=='given' for a in actions]);trigger=np.asarray([a.get('trigger','none') for a in actions]);theta=np.asarray([a.get('progress_threshold',.9681620884830543 if a['switch_mode']=='progress' else 0.) for a in actions]);diag=np.asarray([a.get('late_source','given')=='given' and np.array_equal(a['eta_a'],a['eta_b']) for a in actions]);progress=np.asarray([a['switch_mode']=='progress' for a in actions]);in_grid=np.isin(theta,THRESHOLDS)
    first=(trigger=='none')&(diag|(progress&in_grid));online=(trigger!='none')|zero;fixed=(trigger=='none')&(diag|(progress&(theta==.968)))
    return dict(first=first,online=online,fixed=fixed,zero=zero,trigger=trigger)

def choices(actions,noC,withC,global_trigger):
    pools_=pools(actions);z=np.flatnonzero(pools_['zero']);assert len(z)==1;z=int(z[0]);mn=np.asarray(noC).mean(0);mf=np.asarray(withC).mean(0)
    return {METHODS[0]:pick(mn,pools_['first'],z),METHODS[1]:pick(mn,pools_['online'],z),METHODS[2]:pick(mf,pools_['online'],z),METHODS[3]:pick(mn,pools_['fixed'],z),METHODS[4]:pick(mn,pools_['zero']|(pools_['trigger']==global_trigger),z),METHODS[7]:z}
