"""Check old-horizon position/velocity bytes before admitting old conclusions."""
from dispatch import *
from history import index,request_key
from collections import defaultdict

def geometry_key(sc):
    cfg=dict(sc['config']);cfg.pop('max_steps',None)
    return digest(dict(config=cfg,walls=sc['walls'],bottleneck_config=sc.get('bottleneck_config'),route=bool(sc['route_guidance'])))

def main():
    p=HERE/'PREFIX_CHECK.json'
    if p.exists():return read(p)
    scenes=read(HERE/'SCENES.json')['entries']
    compatible={geometry_key(dict(config=e['scene']['config'],walls=e['scene']['walls'],
        bottleneck_config=e['scene']['bottleneck_config'],route_guidance=e['scene']['route_guidance'])):e
        for e in scenes if e['scene']['v1_equivalent_except_horizon']}
    buckets=defaultdict(list)
    for item in index().values():
        if item['solo']is not None or item['outcome']not in ('success','timeout')or item['state_index']not in range(8):continue
        e=compatible.get(geometry_key(item['scenario']))
        if not e:continue
        buckets[(e['type_id'],item['controller']['type_code'],item['outcome'],bool(item['scenario']['route_guidance']))].append((e,item))
    selected=[];keys=sorted(buckets)
    for offset in range(64):
        for key in keys:
            if offset<len(buckets[key]):selected.append(buckets[key][offset])
            if len(selected)>=60:break
        if len(selected)>=60:break
    assert len(selected)>=60,('Not enough compatible prefix examples',len(selected))
    assert {a['outcome']for _,a in selected}=={'success','timeout'}
    assert {a['controller']['type_code']for _,a in selected}=={'centralized_sampling','gradient_mpc'}
    assert {a['scenario']['route_guidance']for _,a in selected}=={False,True}
    cases=[];labels=[]
    for e,item in selected:
        c=case(e['scene'],item['controller'],item['state_index'],item['spec'],purpose='v3_prefix_check')
        assert index().get(request_key(c)) is not None
        cases.append(c);labels.append(dict(new_uid=c['rollout_uid'],old_uid=item['rollout_uid'],old_trace=item['trace'],
            type_id=e['type_id'],level=e['level'],controller=item['controller']['type_code'],outcome=item['outcome'],route=bool(item['scenario']['route_guidance'])))
    dump(HERE/'PREFIX_SELECTION.json',dict(labels=labels,rule='60 stratified v1 episodes, success/timeout, both controllers, route on/off; fixed before rerunning.'))
    rows={r['rollout_uid']:r for r in execute(cases,'prefix_check',worker_entry='worker_recorded.py')}
    checks=[]
    for label in labels:
        uid=label['new_uid'];detail={}
        if uid in rows:
            with np.load(label['old_trace'])as old,np.load(rows[uid]['trace'])as new:
                for k in ('positions','velocities'):
                    a=old[k];b=new[k][:len(a)]
                    detail[k]=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        checks.append(dict(**label,arrays=detail,equal=bool(detail)and all(detail.values())))
    result=dict(passed=len(checks)>=60 and all(c['equal']for c in checks),pairs=len(checks),checks=checks,
                policy='Only if all prefix bytes match may v1 success/collision/unknown be borrowed; old timeouts rerun only when queried.')
    dump(p,result)
    if not result['passed']:
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n前缀检查未通过；禁用 v1 结局复用，所有所需候选按本轮6倍时限运行。\n')
    print('PREFIX',result['passed'],len(checks),flush=True);return result

if __name__=='__main__':main()
