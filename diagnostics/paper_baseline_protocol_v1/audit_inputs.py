"""Audit immutable copies; retain only the exact frozen deployment choices."""
import guard
from guard import HERE
import collections, datetime, hashlib, json

BASE = HERE/'inputs/diagnostics'
def read(path): return json.loads(path.read_text())
def dump(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    split=read(BASE/'hetero_set_critic_selfcheck_v1/SPLIT.json')
    first=read(BASE/'hetero_set_critic_selfcheck_v1/FINAL_FROZEN.json')
    second=read(BASE/'hetero_set_learned_planners_v1/FROZEN_T2.json')
    learners=read(BASE/'hetero_set_learned_planners_v1/LEARNED.json')
    assert len(learners)==6 and len(split['train'])==28 and len(first['val'])==8
    assert len(first['test'])==len(second['test2'])==12
    assert not set(split['train']) & (set(first['test'])|set(second['test2']))
    assert not set(first['test'])&set(second['test2'])
    model_records=[]
    for learner in learners:
        folder=BASE/'hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset/learned'/learner['ckpt']
        files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()for name in ('meta.json','params.msgpack')}
        model_records.append(dict(name=learner['name'],copied_checkpoint=str(folder.relative_to(HERE)),files=files))
    all_selected=[];groups=[]
    names={x['name']for x in learners}
    for filename,frozen,testkey in [('eval.jsonl',first,'test'),('t2_eval.jsonl',second,'test2')]:
        raw=[json.loads(line)for line in(BASE/'hetero_set_learned_planners_v1'/filename).open()]
        coupled=[r for r in raw if 'key'in r]
        index=collections.defaultdict(list)
        for r in coupled: index[(r['scene'],r.get('name'),r['state'],r['key'])].append(r)
        selected=[]
        for scene in frozen[testkey]:
            key='p'+str(frozen['choices'][scene]['ens16'])
            for name in sorted(names):
                for si in range(8,24):
                    for method,rawkey in [('学习选修正参数',key),('只加安全投影','menu:safety'),('不加安全投影','nominal'),('经典目标右侧规则','rhr_goal')]:
                        values=index[(scene,name,si,rawkey)]
                        assert len(values)==1, ('Missing/duplicate frozen label',filename,scene,name,si,rawkey,len(values))
                        selected.append(dict(method=method,source_file=filename,source_key=rawkey,**values[0]))
        assert len(selected)==12*6*16*4
        groups.append(dict(source=filename,physical_states=192,selected_rows=len(selected),learned_policy_rows=sum(r['method']=='学习选修正参数'for r in selected),solo_rows_excluded=sum('robot'in r for r in raw),old_label_schema_has_completion_time=all('completion_time'in r for r in selected),old_label_schema_has_trace=all('trace'in r for r in selected)))
        all_selected.extend(selected)
    dump('FROZEN_HISTORICAL_LABELS.json',dict(records=all_selected,used_for_new_parameter_selection=False,prefix_reuse_check_pending=True))
    scenes=read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries']
    rep=read(BASE/'scene_types_oracle_coverage_v3/REPRESENTATIVE17_FROZEN.json')['selected']
    selected={r['scene']for r in rep};train=[r for r in scenes if r['scene']['name']not in selected]
    assert len(scenes)==49 and len(train)==32 and len(rep)==17
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),models=model_records,
        benchmark_A=dict(train_scenes=28,val_scenes=8,test_scenes=24,test_physical_states=384,labels=groups),
        benchmark_B=dict(all_scenes=49,train_scenes=32,test_scenes=17,existing_states_per_scene=sorted({len(r['scene']['states'])for r in scenes}),train_by_type=dict(collections.Counter(r['type_id']for r in train))),
        frozen_choice_only=True,centre_rotation_excluded=True,rollouts_started=0,
        unresolved=['Historical label JSONL contains only outcome labels, not per-step traces or completion times; inspect other saved artifacts before claiming time/prefix reuse.','PoC directory has no Git metadata; record unavailable Git audit rather than an empty clean diff.'])
    dump('INPUT_AUDIT.json',result)
    print('Copied input audit passed:6 checkpoints,24 held-out scenes,384 physical states,32 breadth training scenes; no rollout.')

if __name__=='__main__': main()
