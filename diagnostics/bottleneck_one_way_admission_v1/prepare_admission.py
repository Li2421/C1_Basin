"""Freeze geometry and recipes before any new control outcomes are inspected."""
import guard
from guard import HERE
import json,itertools,hashlib
import numpy as np
from scipy.stats import qmc
from passages import detect

def read(name):return json.loads((HERE/name).read_text())
def save(name,data):
    p=HERE/name;value=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text()==value,('Frozen input changed',name)
    else:p.write_text(value)

def main():
    scenes=read('SCENES.json')['entries'];selected={r['scene']for r in read('REPRESENTATIVE17_FROZEN.json')['selected']}
    expected=[0,0,0,0,4,0,17,1,1,1,2,10,0,0,4,17,0];entries={};checks=[]
    for e in scenes:
        sc=e['scene'];raw,merged=detect(sc['walls'],sc['config'])
        entries[sc['name']]=dict(type_id=e['type_id'],level=e['level'],title=e['title'],representative=sc['name']in selected,raw=raw,passages=merged)
        if sc['name']in selected:checks.append(dict(type_id=e['type_id'],raw=len(raw),expected=expected[e['type_id']-1],merged=len(merged)))
    save('PASSAGES.json',dict(by_scene=entries,checks=checks,reference_counts_match=all(c['raw']==c['expected']for c in checks),merge='Same axis, same transverse interval, overlapping/touching longitudinal intervals only; never bridge a wider bay.'))
    if not all(c['raw']==c['expected']for c in checks):
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n通道原始计数与参考不一致：'+json.dumps(checks)+'\n')
        raise RuntimeError('Passage reference count mismatch; inspect geometry before freezing control')
    keys=['R_e','R_c','rho','K','c1','c2','c3']
    vals=[[.4,.8,1.2],[.3,.5],['fcfs','majority'],[None,3],[.5,1.],[0.,.5,1.],[.5,1.]]
    default=dict(zip(keys,[.8,.5,'fcfs',None,1.,.5,1.]))
    full=[dict(zip(keys,t))for t in itertools.product(*vals)];assert len(full)==288
    sequence=[default];seen={json.dumps(default,sort_keys=True)}
    points=qmc.Sobol(7,scramble=True,seed=2026102901).random_base2(14)
    for point in points:
        d={k:v[min(int(x*len(v)),len(v)-1)]for k,v,x in zip(keys,vals,point)}
        tag=json.dumps(d,sort_keys=True)
        if tag not in seen:seen.add(tag);sequence.append(d)
        if len(sequence)==288:break
    assert len(sequence)==288
    save('ONE_WAY_MENU.json',dict(cartesian_count=288,default=default,all_combinations=full,ordered=sequence,sobol_seed=2026102901))
    old=read('FAMILIES_64.json')['families'];base=old[4]['candidate_pool']
    main=[dict(base[k],one_way=sequence[k])for k in range(64)]
    slow=[];seen=set()
    for x in sequence:
        d=dict(x,c2=0.,c3=0.);tag=json.dumps(d,sort_keys=True)
        if tag in seen:continue
        seen.add(tag);slow.append(dict(base[len(slow)],one_way=d))
        if len(slow)==16:break
    families=[dict(name='四列加通道单向放行',candidate_pool=main),
              dict(name='四列加只让等待方减速',candidate_pool=slow),
              dict(name='只用通道单向放行',candidate_pool=[dict(eta=[0.,0.,0.],coefficients={},one_way=x)for x in sequence[:16]]),
              dict(name='四列加让行者后退且角色可重新分配',candidate_pool=[dict(x,role_reselection=True)for x in old[5]['candidate_pool'][:16]])]
    save('FAMILIES_THIS_ROUND.json',dict(families=families,fixed_reference=base[1],fixed_admission=dict(base[1],one_way=default),selection='Pre-outcome frozen recipes; no family/dimension selection.'))
    save('DESIGN_FREEZE.json',dict(files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in ['admission.py','passages.py','runtime.py','PASSAGES.json','ONE_WAY_MENU.json','FAMILIES_THIS_ROUND.json']},rule='Only implementation corrections relative to prompt after this point; affected outcomes must be invalidated explicitly.'))
    print('Geometry and menus frozen:',checks,flush=True)

if __name__=='__main__':main()
