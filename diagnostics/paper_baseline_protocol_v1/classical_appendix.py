"""Historical seven-controller appendix only; no simulation or re-selection."""
import guard
from guard import HERE
import json,csv,collections,hashlib

NAMES={0:'集中式随机采样模型预测控制',1:'梯度下降模型预测控制',2:'分布式模型预测控制（打靶）',
       4:'交叉熵方法模型预测控制',5:'分布式交叉熵方法模型预测控制',6:'冲突搜索多机规划',7:'优先级时空A*规划'}
METHODS={'pipe':'冻结流水线','safety':'只加安全投影','nominal':'不加安全投影','rhr_goal':'经典目标右侧规则'}


def run():
    base=HERE/'inputs/diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset'
    first=json.loads((base/'FINAL_FROZEN.json').read_text());second=json.loads((base/'FROZEN_T2.json').read_text())
    choices={**first['choices'],**second['choices']};parts={'validation':first['val'],'test':first['test']+second['test2']}
    scenes={s['name']:s for s in json.loads((base/'scenes_all2.json').read_text())}
    names=set(sum(parts.values(),[]));index={};conflicts=set();sources={};missing_files=[]
    files=['test_menu.jsonl','test_props.jsonl','test_nominal.jsonl','test8.jsonl','test_rhr.jsonl','ens_val.jsonl',
           'fresh.jsonl','val8.jsonl','rhr_goal.jsonl','val_nominal.jsonl','learned/eval.jsonl','learned/t2_eval.jsonl']
    for name in files:
        path=base/name
        if not path.exists():missing_files.append(name);continue
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            r=json.loads(line)
            if 'robot'in r or r.get('scene')not in names or r.get('controller')not in NAMES or not 8<=r.get('state',-1)<=23:continue
            action=r.get('key')or r.get('action')
            if action is None:continue
            action=action[5:]if action.startswith('menu:')else action
            key=(r['scene'],action,r['controller'],r['state']);value=int(r['success'])
            if key in index and index[key]['success']!=value:conflicts.add(key)
            index[key]=dict(success=value,source=name,source_key=action,outcome=r.get('outcome'))
    records=[];table=[]
    for part,selected in parts.items():
        for controller,display in NAMES.items():
            for method,label in METHODS.items():
                rows=[]
                for scene in selected:
                    action='p'+str(choices[scene]['ens16'])if method=='pipe'else method
                    for state in range(8,24):
                        key=(scene,action,controller,state)
                        result=None if key in conflicts else index.get(key)
                        row=dict(part=part,scene=scene,category=scenes[scene]['cluster'],state=state,
                                 controller=controller,controller_display=display,method=label,result=result)
                        rows.append(row);records.append(row)
                known=[r['result']for r in rows if r['result']is not None]
                table.append(dict(part=part,controller=controller,controller_display=display,method=label,
                    planned=len(rows),observed=len(known),successes=sum(r['success']for r in known),
                    success_rate=sum(r['success']for r in known)/len(known)if known else None,
                    missing=len(rows)-len(known),completion_time_seconds=None,
                    geometry='Historical original wall-collision model; not the physically aligned primary benchmark.'))
    result=dict(rows=table,records=records,source_sha256=sources,missing_source_files=missing_files,
        conflicting_identities=[list(k)for k in sorted(conflicts)],new_rollouts=0,
        limitation='Historical descriptive appendix, not a matched comparison to new external methods. Existing labels and frozen scene-level choices only; missing times are not zero. No claim of equivalent execution on the new consistent geometry.')
    (HERE/'CLASSICAL_CONTROLLER_APPENDIX.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with(HERE/'CLASSICAL_CONTROLLER_APPENDIX.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    return result


if __name__=='__main__':run()
