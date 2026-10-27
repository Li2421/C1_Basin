"""Point estimates only from sealed reuse rows and landed evaluation outcomes."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
from result_inventory import qualification
import collections,numpy as np


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix='POINT_COMPARISON_TRAIN_FIXED_ALL4_20261022';proof=prefix+'.receipt.json'
    if (HERE/proof).exists():verify_receipt(proof);return
    result=[];sources=[]
    for scene in tuple(NAMES):
        selection='TRAIN_SELECTED_FIXED_20261022_'+scene;verify_receipt(selection+'.receipt.json')
        selected=read(HERE/(selection+'.json'));states={(s['N'],s['index']) for s in read(HERE/f'LEARNING_SPLIT_REVISED_20261021_{scene}.json')['evaluation']}
        eligible={r['controller'] for r in qualification(scene) if r['qualified']};records=[]
        for stage,method in [(f'distributed_baseline_{scene}','safety'),(f'learning_eval_revised_20261021_trainfixed_{scene}','train_selected_fixed'),(f'learning_eval_revised_20261021_{scene}','learned_scene')]:
            verify_receipt(stage+'_PREFLIGHT.receipt.json');reuse={r['rollout_uid']:r['cached_row'] for r in read(HERE/f'{stage}_REUSED_ROWS.json')}
            for c in read(HERE/f'{stage}_CASES.json'):
                s=c['state']
                if (s['N'],s['index']) not in states:continue
                rid=c['rollout_uid'];path=HERE/stage/'raw'/f'{rid}.json'
                if path.exists():r=read(path);source=dict(raw=str(path),sha256=sha(path))
                else:r=reuse[rid];source=dict(preflight=stage+'_PREFLIGHT.receipt.json',preflight_sha256=sha(HERE/f'{stage}_PREFLIGHT.receipt.json'),cached_raw_record_hash=r['raw_record_hash'])
                assert all(c[k]==r[k] for k in ('rollout_uid','state_uid','controller_uid','eta_uid','seed_key'))
                ctl=c['controller_payload']['nominal'];records.append(dict(controller=ctl['controller'],type_code=ctl['type_code'],method=method,state=(s['N'],s['index']),seed=c['seed'],success=r['success'],upper=min(1,r['success']+r['numerical_failure']),unknown=r['numerical_failure'],collision=r['collision'],timeout=r['timeout']))
                sources.append(dict(scene=scene,logical_stage=stage,rollout_uid=rid,**source))
        controllers=[]
        for ctl in sorted({r['controller'] for r in records}):
            rr=[r for r in records if r['controller']==ctl];arms={}
            for method in ('safety','train_selected_fixed','learned_scene'):
                rows=[r for r in rr if r['method']==method];assert len(rows)==2*len(states)
                assert {(r['state'],r['seed']) for r in rows}=={(s,seed) for s in states for seed in (0,1)}
                arms[method]=dict(requests=len(rows),success_lower=float(np.mean([r['success'] for r in rows])),success_upper=float(np.mean([r['upper'] for r in rows])),unknowns=sum(r['unknown'] for r in rows),collisions=sum(r['collision'] for r in rows),timeouts=sum(r['timeout'] for r in rows))
            controllers.append(dict(controller=ctl,type_code=rr[0]['type_code'],qualified=ctl in eligible,arms=arms,learned_minus_best_fixed=arms['learned_scene']['success_lower']-arms['train_selected_fixed']['success_lower']))
        tables=[]
        for typ in sorted({r['type_code'] for r in controllers})+['type_equal']:
            rr=[r for r in controllers if r['qualified'] and (typ=='type_equal' or r['type_code']==typ)];groups=collections.defaultdict(list)
            for r in rr:groups[r['type_code']].append(r)
            arms={method:{metric:float(np.mean([np.mean([r['arms'][method][metric] for r in rows]) for rows in groups.values()])) for metric in ('success_lower','success_upper')} for method in ('safety','train_selected_fixed','learned_scene')}
            for method,values in arms.items():values.update(unknowns=sum(r['arms'][method]['unknowns'] for r in rr),collisions=sum(r['arms'][method]['collisions'] for r in rr),requests=sum(r['arms'][method]['requests'] for r in rr))
            tables.append(dict(type_code=typ,qualified_controllers=len(rr),arms=arms,learned_minus_best_fixed=arms['learned_scene']['success_lower']-arms['train_selected_fixed']['success_lower'],identified_difference_bounds=[arms['learned_scene']['success_lower']-arms['train_selected_fixed']['success_upper'],arms['learned_scene']['success_upper']-arms['train_selected_fixed']['success_lower']]))
        result.append(dict(scene=scene,selected_menu_index=selected['selected_menu_index'],evaluation_states=len(states),all_controllers=48,qualified_controllers=len(eligible),tables=tables,controllers=controllers))
    write(HERE/(prefix+'.json'),dict(utc=utc(),scope='Four completed scenes; revised distributed semantics;exact matching evaluation states/controllers/seeds;qualified type-equal primary,all48 per-controller supplementary',evidence='Posthoc development point estimates;final5000-bootstrap confidence intervals pending',scenes=result,new_rollouts=0))
    write(HERE/(prefix+'_SOURCES.json'),sources)
    text='# 训练选定固定对照：四场景配对成功率\n\n只读已完成评估与官方精确复用行。类型→controller→state等权，seed成对；能力门合格实例为主，48个实例逐项数据见JSON。这里只给点估计及unknown识别界，最终5000次bootstrap CI另报。\n\n| 场景 | 训练选定菜单（0起） | 安全层成功率 | 最佳固定成功率 | 学习成功率 | 学习−最佳固定（百分点） | 固定/学习unknown |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for scene in result:
        r=next(t for t in scene['tables'] if t['type_code']=='type_equal');a=r['arms'];text+=f"| {NAMES[scene['scene']]} | {scene['selected_menu_index']} | {100*a['safety']['success_lower']:.6f}% | {100*a['train_selected_fixed']['success_lower']:.6f}% | {100*a['learned_scene']['success_lower']:.6f}% | {100*r['learned_minus_best_fixed']:.6f} | {a['train_selected_fixed']['unknowns']}/{a['learned_scene']['unknowns']} |\n"
    (HERE/(prefix+'.md')).write_text(text)
    seal(proof,[Path(__file__),HERE/(prefix+'.json'),HERE/(prefix+'.md'),HERE/(prefix+'_SOURCES.json')]+[HERE/f'TRAIN_SELECTED_FIXED_20261022_{s}.receipt.json' for s in tuple(NAMES)],new_rollouts=0,point_estimates_only=True)
    print(text,flush=True)


if __name__=='__main__':main()
