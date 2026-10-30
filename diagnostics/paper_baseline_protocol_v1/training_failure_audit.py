"""Recompute selected training failures; never changes menus or outcomes."""
import guard
from guard import HERE
import json,collections,datetime


def run():
    results=[]
    revision=json.loads((HERE/'EXECUTION_FROZEN.json').read_text())['revision']
    for path in sorted((HERE/'tuning').glob('A_*.json'))+sorted((HERE/'tuning').glob('B_*.json')):
        selected=json.loads(path.read_text())
        if not selected.get('available'):continue
        index=selected['global_index'];benchmark=selected['benchmark'];method=selected['method'];rows=[]
        for filename in sorted((HERE/'tuning').glob('tune_'+benchmark+'_'+method+f'_v{revision:03d}_*_records.json')):
            rows.extend(r for r in json.loads(filename.read_text())['records']if r['index']==index)
        identity=[(r['scene'],r['state'],r['nominal'])for r in rows]
        assert len(identity)==len(set(identity)),'Duplicated training evidence'
        collision=[r for r in rows if r['result']['collision']]
        margins=[r['result'].get('minimum_swept_clearance')for r in collision];margins=[v for v in margins if v is not None]
        groups={}
        for row in rows:groups.setdefault(str(row['category']),[]).append(row)
        results.append(dict(benchmark=benchmark,method=method,selected_index=index,episodes=len(rows),
            outcomes=dict(collections.Counter(r['result']['outcome']for r in rows)),
            unhandled_episodes=sum(bool(r['result'].get('paper_original',{}).get('numerical_unknown',r['result'].get('numerical_failure',0)))for r in rows),
            native_error_type_episodes=dict(collections.Counter(key for r in rows for key in r['result'].get('paper_error_types',{}))),
            speed_clipped_episodes=sum(r['result'].get('speed_clipped_steps',0)>0 for r in rows),
            collision_margin_min=min(margins)if margins else None,collision_margin_max=max(margins)if margins else None,
            negative_clearance_bins=[dict(deeper_than_m=eps,count=sum(v< -eps for v in margins))for eps in (1e-9,1e-6,1e-4,.001,.01)],
            collision_types=dict(collections.Counter(json.dumps(r['result'].get('collision_types'),sort_keys=True)for r in collision)),
            category_counts={k:dict(episodes=len(v),outcomes=dict(collections.Counter(r['result']['outcome']for r in v)))for k,v in groups.items()},
            examples=[dict(scene=r['scene'],state=r['state'],nominal=r['nominal'],query_uid=r['query_uid'],
                minimum_swept_clearance=r['result'].get('minimum_swept_clearance'),trace=r['result'].get('trace'))for r in sorted(collision,key=lambda r:(r['scene'],r['state'],r['nominal']))[:5]]))
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),training_only=True,results=results,new_rollouts=0,
        interpretation='Descriptive selected-training outcomes. Clearance bins diagnose magnitude and do not change the fixed collision threshold. These statistics do not select any new settings or establish a causal explanation; author-original safety-distance rows are a separate fixed reference.')
    (HERE/'TRAINING_FAILURE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':run()
