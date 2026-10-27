"""Matched evaluation and class scopes for the frozen training-selected action."""
from experiment import *
from analysis_standard_supplement_20261022 import FIXED as SELECTION_PREFIX
from registry import verify_receipt
from statistics_training_selected_reader_20261022 import summarize

SCOPES=('all_qualified','controller_failure','remaining_instances','safety_layer_global_failure')


def primary_task(task,rows,classes):
    scenes=task.get('scenes') or [task['scene']];scope=task['scope'];chosen=[];manifests={}
    for scene in scenes:
        proof=SELECTION_PREFIX+'_'+scene+'.receipt.json';verify_receipt(proof)
        manifests[scene]=dict(receipt=proof,sha256=sha(HERE/proof),selected_menu_index=read(HERE/(SELECTION_PREFIX+'_'+scene+'.json'))['selected_menu_index'])
        census={r['controller']:r for r in classes if r['scene']==scene and r['layer']=='distributed'}
        for row in rows:
            if row['scene']!=scene or row['layer']!='distributed' or row['method'] not in ('safety','train_selected_fixed','learned_scene'):continue
            c=census[row['controller']]
            include=scope=='all_qualified' or scope=='controller_failure' and c['remove_in_revised_exclusion'] or scope=='remaining_instances' and not c['remove_in_revised_exclusion'] or scope=='safety_layer_global_failure' and c['classification']=='安全层导致的全局失败'
            if include:chosen.append(row)
    # Restrict all arms to the same held-out state scope, before any outcomes.
    states={(r['scene'],r['N'],r['state']) for r in chosen if r['method']=='learned_scene'}
    chosen=[r for r in chosen if (r['scene'],r['N'],r['state']) in states]
    arm_keys={m:{(r['scene'],r['controller'],r['N'],r['state'],r['seed']) for r in chosen if r['method']==m} for m in ('safety','train_selected_fixed','learned_scene')}
    assert len({frozenset(keys) for keys in arm_keys.values()})==1,('Unmatched primary comparison',task,{m:len(k) for m,k in arm_keys.items()})
    result=summarize(chosen,[('distributed',m) for m in ('safety','train_selected_fixed','learned_scene')],scenes)
    result.update(fixed_selection=manifests,instance_scope=scope,selection_uses_evaluation_outcomes=False,selection_repeated_in_bootstrap=False,primary_reference='Single action selected once per scene by training-label type-equal success, ties by menu order',gain_interpretation='修正推动controller自身停住的实例，不计死锁解决' if scope=='controller_failure' else '成功增益须结合逐机器人滤波器平衡点死锁诊断判断；被已到达者阻挡不计死锁证据')
    return result
