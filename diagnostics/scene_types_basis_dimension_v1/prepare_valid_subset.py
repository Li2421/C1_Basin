"""Invalidate the erroneous transverse prototype; never choose a geometry option."""
from core import *
import datetime,shutil

def main():
    frozen=read(HERE/'SCENES_FROZEN.json')
    if frozen.get('valid_subset_revision')==2:return
    archive=HERE/'scene_revisions/v1_before_invalidating_type2';archive.mkdir(parents=True,exist_ok=True)
    for name in ('SCENES_FROZEN.json','CALIBRATION.json','FIVE_LAYER.json','GEOMETRY_CONSISTENCY.json','SCENE_DESCRIPTORS.json'):
        if(HERE/name).exists():shutil.copyfile(HERE/name,archive/name)
    reason='The prototype used transverse lines instead of longitudinal queues. With its adopted radius0.1m, the requested3m field cannot contain one-sided4–6-agent queues at2.5diameter spacing. Radius/extent clarification is pending; neither proposed relaxation has been executed. Invalid prototype data are retained but excluded.'
    for j,e in enumerate(frozen['entries']):
        if e['type_id']==2:
            frozen['entries'][j]=dict(type_id=2,title=e['title'],status='geometry_gap',gap=reason,prototype_archive=str(archive/'SCENES_FROZEN.json'),geometry_options_pending=True)
    frozen.update(valid_subset_revision=2,stage_suffix='_valid15_v2',revision_time=datetime.datetime.now().astimezone().isoformat(),revision_reason=reason,no_correction_seen=False,unchanged_valid_scene_definition_hashes=True,original_freeze_before_all_corrections=True)
    dump(HERE/'SCENES_FROZEN.json',frozen)
    layers=read(HERE/'FIVE_LAYER.json');layers['2']=dict(geometry_valid=False,single_agent_paths='requested longitudinal prototype not instantiated',joint_feasible='not_certified',controller_gate='not_assessed_on_requested_geometry',deadlock_resolution='not_evaluated',gap=reason);dump(HERE/'FIVE_LAYER.json',layers)
    g=read(HERE/'GEOMETRY_CONSISTENCY.json');g['2']=dict(applicable=False,reason=reason,previous_prototype_margins=str(archive/'GEOMETRY_CONSISTENCY.json'));dump(HERE/'GEOMETRY_CONSISTENCY.json',g)
    cal=read(HERE/'CALIBRATION.json');cal.setdefault('invalidated_types',{})['2']=reason;dump(HERE/'CALIBRATION.json',cal)
    with(HERE/'GAPS.md').open('a')as f:f.write('\n开阔垂直交叉流：纵列场景实现错误，当前标为缺口，原横列结果全部留档但不用于选族或验证。6×6m/半径0.1m与3×3m/半径0.05m两项几何提案已生成且通过纯几何检查，等待用户决定。其余15类维持原冻结定义，原有兼容结果复用。\n')
    with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n2026-10-28：使错误交叉流原型失效，SCENES_FROZEN revision2仅把该条目改为缺口，未改变其余15类的任何状态/几何/时限/控制器。原始freeze在所有修正之前已完成，其余15类的定义hash保持不变。S1/S2以valid15版本清单重新官方preflight并复用已有正确结果；不重新仿真。S3/S4与留出只使用合法子集，几何选择没有被默认批准。\n')

if __name__=='__main__':main()
