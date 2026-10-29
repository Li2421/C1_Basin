"""Frozen projections, exact deduplication, and bounded lower-level expansion."""
from core import *
from scipy.stats import qmc
ITEMS=('G','R','T','C','Q')
LABELS=dict(G='朝目标',R='互斥分离',T='目标右侧旋转',C='局部净空梯度',Q='通道放行')
BOX=dict(G=(.5,1.25),R=(0.,.75),T=(-1.,1.),C=(0.,1.),Q=(.5,1.))
BINDINGS=(('Q','G'),('Q','T'),('C','R'))
DEFAULT=dict(R_e=.8,R_c=.5,rho='fcfs',K=None)

def signature(active,bindings):return json.dumps(dict(active=sorted(active),bindings=dict(sorted(bindings.items()))),sort_keys=True,separators=(',',':'))

def project(values,active,bindings):
    values={k:float(values[k])if k in active else 0. for k in ITEMS}
    for dst,src in bindings.items():values[dst]=min(abs(values[src])if src=='T'else values[src],1.)if dst=='Q'else values[src]
    return values

def spec(values,active):
    co={name:values[k]for k,name in [('T','T1'),('C','C2')]if values[k]!=0.}
    out=dict(eta=[values['G'],0.,values['R']],coefficients=co)
    if 'Q'in active:
        q=values['Q'];out['one_way']=dict(DEFAULT,c1=q,c2=q,c3=q)
    return out

def title(active,bindings):
    parts=[]
    for k in ITEMS:
        if k not in active or k in bindings:continue
        label=LABELS[k]
        for dst,src in sorted(bindings.items()):
            if src==k:label+='·含'+('净空'if dst=='C'else LABELS[dst])
        parts.append(label)
    return' + '.join(parts)

def build(code,active,bindings,parents=(),source_level=4):
    active=list(active);bindings=dict(bindings);free=[k for k in ITEMS if k in active and k not in bindings]
    base=read(HERE/'FAMILIES_64.json')['families'][4]['candidate_pool'];seen=set();pool=[];provenance=[]
    sampler=qmc.Sobol(d=len(free),scramble=True,seed=2026102902);sobol=sampler.random_base2(14);si=0
    def key(v):return tuple(float(v[k]).hex()for k in ITEMS)
    for k,old in enumerate(base):
        raw=dict(G=old['eta'][0],R=old['eta'][2],T=old['coefficients'].get('T1',0.),C=old['coefficients'].get('C2',0.),Q=[1.,.75,.5][k%3]);v=project(raw,active,bindings);replacement=False
        while key(v)in seen:
            point=sobol[si];si+=1;replacement=True
            raw={item:0. for item in ITEMS}
            for item,z in zip(free,point):raw[item]=BOX[item][0]+float(z)*(BOX[item][1]-BOX[item][0])
            v=project(raw,active,bindings)
        seen.add(key(v));pool.append(spec(v,active));provenance.append(dict(index=k+1,base_index=k+1,sobol_replacement=replacement,sobol_consumed=si,values=v))
    assert len(pool)==len(seen)==64
    fixed=dict(G=.875,R=.375,T=-.75,C=0.,Q=1.);plus=dict(fixed,T=.75)
    return dict(code=code,name=title(active,bindings),active=active,bindings=bindings,free=free,dimension=len(free),parents=list(parents),level=source_level,signature=signature(active,bindings),candidate_pool=pool,candidate_provenance=provenance,fixed=spec(project(fixed,active,bindings),active),fixed_plus=spec(project(plus,active,bindings),active),no_passage_preserves_four=all(k in active and k not in bindings for k in('G','R','T','C')))

def initialize():
    path=HERE/'COMBINATIONS.json'
    if path.exists():return read(path)
    rows=[build('reference',ITEMS,{},source_level=5),build('four_1',ITEMS,{'Q':'G'}),build('four_2',ITEMS,{'Q':'T'}),build('four_3',ITEMS,{'C':'R'})]
    for code,drop in [('four_4','C'),('four_5','R'),('four_6','T'),('four_7','G'),('four_8','Q')]:rows.append(build(code,[k for k in ITEMS if k!=drop],{}))
    data=dict(seed=2026102902,order=[r['code']for r in rows],combinations=rows,levels={'5':['reference'],'4':[r['code']for r in rows if r['dimension']==4]},generation_rule='Replace projected duplicate in its original position with next scrambled Sobol point in remaining free coefficient box; exact float64 identity; dimensions3 max8,2 max6, parent coverage descending then parent ID and canonical child signature.')
    dump(path,data);return data

def lower(level,parents,coverage):
    data=read(HERE/'COMBINATIONS.json')
    if str(level)in data['levels']:return[data_row for data_row in data['combinations']if data_row['code']in data['levels'][str(level)]]
    candidates={}
    for parent in sorted(parents,key=lambda p:(-coverage[p['code']],p['code'])):
        options=[]
        for free in parent['free']:
            removed={free}|{dst for dst,src in parent['bindings'].items()if src==free};active=[k for k in parent['active']if k not in removed];bindings={d:s for d,s in parent['bindings'].items()if d in active and s in active};options.append((active,bindings))
        for dst,src in BINDINGS:
            if dst in parent['active']and src in parent['active']and dst not in parent['bindings']:options.append((parent['active'],dict(parent['bindings'],**{dst:src})))
        for active,bindings in options:
            if len(active)-len(bindings)!=level:continue
            key=signature(active,bindings)
            if key not in candidates:candidates[key]=dict(active=active,bindings=bindings,parents=[],best_parent=parent['code'],parent_coverage=coverage[parent['code']])
            if parent['code']not in candidates[key]['parents']:candidates[key]['parents'].append(parent['code'])
    ordered=sorted(candidates.items(),key=lambda kv:(-kv[1]['parent_coverage'],kv[1]['best_parent'],kv[0]));limit=8 if level==3 else 6
    selected=[]
    for i,(sig,c)in enumerate(ordered[:limit]):selected.append(build(f'level{level}_{i+1:02d}',c['active'],c['bindings'],c['parents'],level))
    data['combinations']+=selected;data['levels'][str(level)]=[r['code']for r in selected];data.setdefault('lower_selection',{})[str(level)]=dict(eligible=len(ordered),limit=limit,selected=len(selected),excluded=[dict(signature=k,**c)for k,c in ordered[limit:]])
    dump(HERE/'COMBINATIONS.json',data);return selected

if __name__=='__main__':print('Frozen combinations',len(initialize()['combinations']))
