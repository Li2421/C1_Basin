"""User-requested one-scene-per-type cohort, frozen without correction labels."""
from core import *

PATH=HERE/'REPRESENTATIVE17_FROZEN.json'


def in_requested_scope(group):
    if not PATH.exists()or group['family']==4:return True
    return (group['type_id'],group['level'])in scope_keys()


from functools import lru_cache
@lru_cache(None)
def scope_keys():return frozenset((e['type_id'],e['level'])for e in read(PATH)['selected'])


def selected_scenes():
    scenes=read(HERE/'SCENES.json')['entries']
    if not PATH.exists():return scenes
    keys={(e['type_id'],e['level'])for e in read(PATH)['selected']}
    return [e for e in scenes if(e['type_id'],e['level'])in keys]


def freeze():
    if PATH.exists():
        previous=read(PATH)
        if previous.get('selection_policy')=='geometry_middle_only_v2':return previous
        # User clarified geometry-only selection before any later-stage run.
        assert not(HERE/'family_4_complete.json').exists()
        dump(HERE/'REPRESENTATIVE17_SAFETY_PROPOSAL_NOT_EXECUTED.json',previous)
    scenes=read(HERE/'SCENES.json')['entries'];base=read(HERE/'SAFETY_ONLY.json')['cases'];gates=read(HERE/'SOLO_GATES.json')['entries']
    selected=[];audit=[]
    for t in range(1,18):
        entries=[e for e in scenes if e['type_id']==t];knobs=sorted(e['scene']['knob']for e in entries);mid=float(np.median(knobs))
        evidence=[]
        for e in entries:
            rows=[r for r in base if r['type_id']==t and r['level']==e['level']]
            gate=next(g for g in gates if g['type_id']==t and g['level']==e['level'])
            assert len(rows)==16 and len(gate['applicable_controllers'])==2
            evidence.append(dict(type_id=t,level=e['level'],scene=e['scene']['name'],title=e['title'],
                knob=e['scene']['knob'],num_agents=e['scene']['config']['num_agents'],success=sum(r['success']for r in rows),
                completed=len(rows),parameter_middle_distance=abs(e['scene']['knob']-mid),both_solo_gates_passed=True))
        pick=min(evidence,key=lambda e:(e['parameter_middle_distance'],e['num_agents'],e['level']))
        reason='Middle physical knob only. For two equidistant ladder entries choose lower robot count, then original level order. No outcome-based ranking.'
        selected.append(dict(**pick,selection_reason=reason,geometry_middle_selected=True))
        audit.extend(evidence)
    result=dict(user_instruction='这个跑完之后以后每类场景就跑一个，你选择参数适中的，既不太难也不太简单的',
        applies_after='full49-scene four-column Oracle@64',scene_count=17,physical_states=136,nominal_cases=272,
        selected=selected,all_ladder_selection_evidence=audit,
        criterion='Select median physical ladder knob only: width, number, obstacle count or radius as prescribed per type. Two-entry tie: lower robot count. Passed geometry/solo gates establish validity; all success labels are descriptive only. No map/radius/horizon changes.',
        selection_policy='geometry_middle_only_v2',user_clarification='难度是看场景来的，一般来说越宽肯定越简单',
        safety_labels_used_for_selection=False,correction_labels_used=False,paper_labels_used=False,frozen_scenes_sha256=sha(HERE/'SCENES.json'),
        safety_only_sha256=sha(HERE/'SAFETY_ONLY.json'),solo_gates_sha256=sha(HERE/'SOLO_GATES.json'),
        inherited_states=list(range(8)),original_49_scene_four_column_preserved=True,
        scope_override='Later paper profiles and remaining correction families use only these17 scenes. Original full49 four-column evidence is retained; no unselected future cases are marked as search failures.')
    dump(PATH,result)
    lines=['# 后续每类一个代表场景\n',result['criterion'],
           '\n当前四列仍完成49场景784案例；之后各论文profile136个物理初态，各C1族272案例，8个初态与两个冻结nominal保持不变。',
           '\n按用户澄清，纯按场景物理参数取中间档，不用任何方法成功率选场景。仅安全过滤结果作为描述，不参与排序。物理中档不保证对每个方法都有中等成功率。',
           '\n|类型|档位|机器人数量|变化物理参数|仅安全过滤成功/16（描述）|\n|---|---:|---:|---:|---:|']
    lines.extend(f"|{e['title']}|{e['level']}|{e['num_agents']}|{e['knob']}|{e['success']}|"for e in selected)
    (HERE/'REPRESENTATIVE17.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':freeze()
