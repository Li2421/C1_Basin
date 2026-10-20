"""Predeclared finite menus, canonical physical deduplication and gate choices."""
from common import *
from actions_v3 import action,TIMES
from action_encoding import encode

METHODS=(
 '学习双 η·critic 判修不修·扩充提议',
 '学习双 η·不判修不修·扩充提议',
 '学习双 η·critic 判修不修·只用冻结提议',
 '学习双 η·critic 判修不修·不含时间表生成器',
 '学习双 η·critic 判修不修·不含切换时重查',
 '学习双 η·探测规则判修不修·扩充提议',
 '只加安全投影','人定双 η·第 500 步切换','人定双 η·按进度切换')

def physical_key(a):
    aa=action(a['eta_a'],a['eta_b'],a['switch_mode'],a['switch_step'],late_source=a.get('late_source','given'))
    return (tuple(aa['eta_a']),tuple(aa['eta_b']),aa['switch_mode'],aa['switch_step'],aa['late_source'])

def menu(frozen,generated):
    rows=[];seen={}
    def add(a,origin,**meta):
        k=physical_key(a)
        if k in seen:
            existing=rows[seen[k]]
            if origin not in existing['origins']:existing['origins'].append(origin)
            return
        seen[k]=len(rows);rows.append(dict(action=a,origins=[origin],candidate_id=len(rows),**meta))
    # Original dense catalogue order i,j,time, retaining one diagonal per pair.
    for i in range(8):
        for j in range(8):
            for t in (*TIMES,700):
                add(action(frozen[i],frozen[j],'progress' if t==700 else 'fixed',t),'冻结提议对',early_index=i,late_index=j)
    assert len(rows)==344  # 8 diagonals + 8*7 ordered non-diagonals * 6 switches.
    for index,pair in enumerate(generated):
        for t in (*TIMES,700):add(action(pair[0],pair[1],'progress' if t==700 else 'fixed',t),'时间表生成器提议对',proposal_index=index)
    for i in range(8):
        for mode in ('fixed','progress'):add(action(frozen[i],frozen[0],mode,500,late_source='requery'),'切换时重查',early_index=i)
    add(action([0,0,0],[0,0,0]),'只加安全投影');assert rows[-1]['origins']==['只加安全投影']
    assert len(rows)<=415
    return rows

def choose(rows,probability,k,delta,tau,rho,features):
    p=np.asarray(probability);assert p.shape==(3,len(rows));mu=p.mean(0);sd=p.std(0);zero=len(rows)-1
    assert rows[zero]['origins']==['只加安全投影']
    origin=[set(r['origins']) for r in rows]
    pools=[np.arange(zero),np.arange(zero),np.asarray([i for i,s in enumerate(origin[:zero]) if '冻结提议对' in s]),np.asarray([i for i,s in enumerate(origin[:zero]) if s&{'冻结提议对','切换时重查'}]),np.asarray([i for i,s in enumerate(origin[:zero]) if s&{'冻结提议对','时间表生成器提议对'}]),np.arange(zero)]
    selected={};details={}
    for index,(method,pool) in enumerate(zip(METHODS[:6],pools)):
        best=int(pool[np.argmax(mu[pool])]);margin=float(mu[best]-k*sd[best]-mu[zero]-delta)
        if index==1:chosen=int(np.argmax(mu))
        elif index==5:chosen=zero if features[5]>=tau and features[0]<=rho else best
        else:chosen=best if margin>=0 else zero
        selected[method]=chosen;details[method]=dict(candidate_id=chosen,repair=chosen!=zero,repair_best_id=best,mu_zero=float(mu[zero]),mu_best=float(mu[best]),sigma_best=float(sd[best]),margin=margin)
    selected[METHODS[6]]=zero
    return selected,details
