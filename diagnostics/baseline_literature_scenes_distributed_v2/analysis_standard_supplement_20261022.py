"""User requested posthoc taxonomy and training-only fixed-action selection."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import collections
import copy
import numpy as np

PREFIX='INSTANCE_CLASSES_SUPPLEMENT_20261022'
FIXED='TRAIN_SELECTED_FIXED_20261022'
LABELS=('安全层导致的全局失败','墙前陷阱实例','controller级失败')


def classify(eligible,upper,counts):
    if not eligible:return '未通过能力门'
    if upper>.05:return '非全局失败实例'
    total=sum(counts.values())
    safety=counts.get('robot_filter_equilibrium_deadlock',0)+counts.get('blocked_by_arrived',0)
    if total and safety>total/2:return LABELS[0]
    if total and counts.get('wall_filter_equilibrium',0)>total/2:return LABELS[1]
    return LABELS[2]


def taxonomy():
    proof=PREFIX+'.receipt.json'
    if (HERE/proof).exists():verify_receipt(proof);return
    verify_receipt('INSTANCE_CLASSES_REVISED.receipt.json')
    old=read(HERE/'INSTANCE_CLASSES_REVISED.json');new=copy.deepcopy(old);changes=[]
    for row in new['instances']:
        previous=row['classification'];counts=row['primary_unarrived_agent_episode_categories']
        row['classification']=classify(row['qualified'],row['success_upper'],counts)
        row['previous_classification']=previous
        row['remove_in_revised_exclusion']=bool(row['qualified'] and row['classification']==LABELS[2])
        row['robot_deadlock_evidence_eligible']=bool(row['qualified'] and row['classification']!=LABELS[2])
        row['primary_safety_global_failure_agent_episodes']=counts.get('robot_filter_equilibrium_deadlock',0)+counts.get('blocked_by_arrived',0)
        row['primary_robot_deadlock_agent_episodes']=counts.get('robot_filter_equilibrium_deadlock',0)
        for point in row['sensitivity']:
            point['classification']=classify(row['qualified'],row['success_upper'],point['unarrived_agent_episode_categories'])
        row['sensitivity_changes']=any(p['classification']!=row['classification'] for p in row['sensitivity'])
        if previous!=row['classification']:changes.append({k:row[k] for k in ('scene','layer','controller','previous_classification','classification')})
    discrimination=[]
    for source in old['discrimination']:
        scene,layer=source['scene'],source['layer'];rr=[r for r in new['instances'] if r['scene']==scene and r['layer']==layer and r['qualified']]
        discrimination.append(dict(scene=scene,layer=layer,qualified_instances=len(rr),controller_level_failure_instances=sum(r['classification']==LABELS[2] for r in rr),state_related_failure_instances=sum(r['state_related_failure_definite'] for r in rr),safety_layer_global_failure_instances=sum(r['classification']==LABELS[0] for r in rr),wall_trap_instances=sum(r['classification']==LABELS[1] for r in rr),instances_with_4a_prime=sum(r['at_least_one_4a_prime'] for r in rr),robot_deadlock_evidence_eligible_instances=sum(r['robot_deadlock_evidence_eligible'] and r['at_least_one_4a_prime'] for r in rr),robot_deadlock_agent_episodes=sum(r['primary_robot_deadlock_agent_episodes'] for r in rr if r['robot_deadlock_evidence_eligible'])))
    new.update(utc=utc(),definition='Qualified success upper<=.05: strict majority of unarrived robot-episodes in robot_filter_equilibrium_deadlock OR blocked_by_arrived => safety-layer global failure; strict wall_filter_equilibrium majority => wall trap; otherwise controller failure. Deadlock evidence uses only robot_filter_equilibrium_deadlock.',discrimination=discrimination,sensitivity_changes=[r for r in new['instances'] if r['sensitivity_changes']],classification_changes=changes,source='INSTANCE_CLASSES_REVISED.json',new_rollouts=0,new_training=0)
    write(HERE/(PREFIX+'.json'),new)
    text='# 补充修订实例归类\n\n只用原封存文件中的九格计数重算，原 INSTANCE_CLASSES_REVISED 不改。安全层全局失败多数判据合并逐机器人滤波器平衡点死锁和被已到达者阻挡；死锁证据计数只用前者。其余候选阈值、严格多数、墙前陷阱优先顺序、W=5s/τ=.05 和九格不变。timeout 不等于死锁。\n\n| 场景 | 层 | 合格实例 | controller 级失败 | 安全层全局失败 | 墙前陷阱 | 有真正死锁类别且可作证据的实例 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in discrimination:text+='| '+' | '.join(str(r[k]) for k in ('scene','layer','qualified_instances','controller_level_failure_instances','safety_layer_global_failure_instances','wall_trap_instances','robot_deadlock_evidence_eligible_instances'))+' |\n'
    text+='\n主归类改变 '+str(len(changes))+' 个实例；完整变化、原始互斥计数、九格归类和另一层旁证见 JSON。新 rollout=0，新训练=0。\n'
    (HERE/(PREFIX+'.md')).write_text(text)
    seal(proof,[Path(__file__),HERE/(PREFIX+'.json'),HERE/(PREFIX+'.md'),HERE/'INSTANCE_CLASSES_REVISED.receipt.json'],new_rollouts=0,new_training=0,old_files_preserved=True)
    print('Taxonomy changes',changes,flush=True)


def select(scene):
    prefix=FIXED+'_'+scene;proof=prefix+'.receipt.json'
    if (HERE/proof).exists():verify_receipt(proof);return read(HERE/(prefix+'.json'))
    bundle=f'LEARNING_BUNDLE_REVISED_20261021_{scene}'
    if not (HERE/(bundle+'.receipt.json')).exists():return None
    verify_receipt(bundle+'.receipt.json')
    with np.load(HERE/(bundle+'.npz')) as z:labels=z['labels'];types=z['types'];controllers=z['controllers']
    assert labels.shape==(16,32,15) and set(np.unique(labels))<={-1,0,1}
    unique=sorted(set(types.tolist()));assert len(unique)==8
    menu=read(HERE/'LEARNING_MENU_REVISED.json');assert len(menu)==15
    rows=[]
    for ai,action in enumerate(menu):
        per_type=[]
        for typ in unique:
            values=labels[types==typ,:,ai];assert values.shape==(2,32)
            per_type.append(dict(type_code=typ,requests=values.size,successes=int((values==1).sum()),unknowns=int((values<0).sum()),success_lower=float((values==1).mean()),success_upper=float((values!=0).mean())))
        # All types have exactly64 labels; the integer numerator makes ties exact.
        rows.append(dict(menu_index=ai,action=action,successes=sum(r['successes'] for r in per_type),unknowns=sum(r['unknowns'] for r in per_type),requests=512,type_equal_success_lower=sum(r['success_lower'] for r in per_type)/8,type_equal_success_upper=sum(r['success_upper'] for r in per_type)/8,per_type=per_type))
    ranking=sorted(rows,key=lambda r:(-r['successes'],r['menu_index']))
    for rank,row in enumerate(ranking,1):row['rank']=rank
    result=dict(utc=utc(),scene=scene,selection_basis='Revised execution training labels only; equal type weights; unknown retained as failure in lower-bound primary score, upper bound separately shown; exact ties resolved by ascending original menu index.',selected_menu_index=ranking[0]['menu_index'],selected_action=ranking[0]['action'],ranking=ranking,training_states=32,training_controllers=controllers.tolist(),training_controller_types=types.tolist(),evaluation_outcomes_used_for_selection=False,menu_sha256=sha(HERE/'LEARNING_MENU_REVISED.json'),new_rollouts=0,new_training=0)
    write(HERE/(prefix+'.json'),result)
    text=f'# {scene}：15 个菜单动作训练标签排序\n\n仅按训练 state 的类型等权成功率下界选动作；unknown 保留并另报上界。平局按原菜单顺序。评估结果不参与选择。\n\n| 排名 | 菜单索引（从0开始） | 成功率下界 | 成功率上界 | 成功/请求 | unknown |\n|---:|---:|---:|---:|---:|---:|\n'
    for r in ranking:text+=f"| {r['rank']} | {r['menu_index']} | {100*r['type_equal_success_lower']:.6f}% | {100*r['type_equal_success_upper']:.6f}% | {r['successes']}/{r['requests']} | {r['unknowns']} |\n"
    (HERE/(prefix+'.md')).write_text(text)
    seal(proof,[Path(__file__),HERE/(prefix+'.json'),HERE/(prefix+'.md'),HERE/(bundle+'.receipt.json'),HERE/'LEARNING_MENU_REVISED.json'],selection_frozen_before_evaluation=True,evaluation_outcomes_used=False,new_rollouts=0)
    print(scene,'training-selected fixed action',result['selected_menu_index'],'score',ranking[0]['type_equal_success_lower'],flush=True)
    return result


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    taxonomy()
    for scene in NAMES:select(scene)


if __name__=='__main__':main()
