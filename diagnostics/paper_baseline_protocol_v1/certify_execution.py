"""Mandatory archived-label replays and fresh original/accelerated pairs."""
from dispatch import execute,status
from core import *
from request_factory import case
from scene_catalog import catalog,frozen_eta
from learned_setup import rows as learned_rows
from collections import defaultdict
import copy
import datetime
import shutil


CONTROL_ARRAYS=('positions','velocities','nominal','safe','actions','goals',
                'swept_clearance','clearance_types','projection_rewrite',
                'route_guides','route_remaining')


def query_uid(row):return row.get('target_query_uid',row['rollout_uid'])


def compare_traces(first,second,prefix=None):
    checks={}
    with np.load(first,allow_pickle=False)as a,np.load(second,allow_pickle=False)as b:
        for key in CONTROL_ARRAYS:
            if key not in a or key not in b:
                assert key not in ('positions','velocities','actions','goals'),('Missing control evidence',key)
                continue
            x,y=a[key],b[key]
            if prefix is not None and key!='goals':
                length=prefix+1 if key in ('positions','velocities')else prefix
                x,y=x[:length],y[:length]
            checks[key]=dict(first_shape=list(x.shape),second_shape=list(y.shape),
                first_dtype=str(x.dtype),second_dtype=str(y.dtype),
                equal=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes())
    return dict(passed=bool(checks)and all(x['equal']for x in checks.values()),arrays=checks,prefix_steps=prefix)


def validation_plan():
    scenes=catalog('A',version='original');models={r['controller']:r for r in learned_rows()}
    labels=read(HERE/'FROZEN_VALIDATION_LABELS.json')['records']
    selected={name:sorted((r for r in labels if r['method']==name),
                         key=lambda r:(r['state'],r['scene'],r['name']))
              for name in ('只加安全投影','学习选修正参数')}
    planned={}
    for method,limit in (('只加安全投影',200),('学习选修正参数',64)):
        records=[]
        for old in selected[method][:limit]:
            eta=[0.,0.,0.]if method=='只加安全投影'else frozen_eta(old['scene'])[1]
            request=case('A',scenes[old['scene']],models[old['name']],old['state'],
                spec=dict(eta=eta,coefficients={}),accelerated=False,
                phase='certification',purpose='mandatory_archived_label_replay')
            records.append(dict(request=request,archived_label=old))
        assert len(records)==limit
        assert len({r['archived_label']['name']for r in records})==6
        assert len({scenes[r['archived_label']['scene']]['factor']for r in records})==4
        planned[method]=records
    return planned


def historical_b_plan(count=34):
    history=read(HERE/'OURS_HISTORY_INDEX.json')['records']
    eligible=read(HERE/'HISTORY_FINGERPRINT_SCREEN.json')['eligible_for_prefix_verification']
    by_type=defaultdict(list)
    for uid in eligible:
        r=history[uid]['result']
        by_type[(r['type_id'],r['controller'])].append(uid)
    chosen=[]
    for key in sorted(by_type):chosen.append(min(by_type[key]))
    for uid in sorted(eligible):
        if len(chosen)>=count:break
        if uid not in chosen:chosen.append(uid)
    assert len(chosen)==count
    scenes=catalog('B');controls={c['type_code']:c for c in controllers()}
    plan=[]
    for uid in chosen:
        archived=history[uid]['result'];scene=scenes[archived['scene']]
        request=case('B',scene,controls[archived['controller']],archived['state_index'],spec=archived['spec'],
                     accelerated=False,phase='certification',purpose='archived_native_prefix_check')
        source=Path(archived['trace']);target=HERE/'inputs/ours_history/traces'/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
        assert sha(target)==archived['trace_sha256'],('Archived trace hash mismatch',uid)
        plan.append(dict(request=request,source_rollout_uid=uid,archived_label=archived,
                         copied_trace=str(target),trace_sha256=sha(target)))
    dump(HERE/'HISTORICAL_B_PREFIX_PLAN.json',dict(records=plan,selection='First UID per training type/controller, then UID order; no success labels used.'))
    return plan


def run():
    output=HERE/'ENV_REUSE_PHASE_COMPLETE.json'
    if output.exists()and read(output).get('complete')and read(output).get('execution_release_sha256')==sha(HERE/'EXECUTION_FROZEN.json'):return read(output)
    revision=read(HERE/'EXECUTION_FROZEN.json')['revision']
    stage=lambda name:f'{name}_v{revision:03d}'
    if output.exists()and read(output).get('execution_release_sha256')!=sha(HERE/'EXECUTION_FROZEN.json'):
        previous=HERE/'certification_revisions'/f'before_v{revision:03d}'
        previous.mkdir(parents=True,exist_ok=True)
        for name in ('ENV_REUSE_PHASE_COMPLETE.json','ENV_CHECK.json','REUSE_CHECK.json','HISTORY_PREFIX_CHECK.json','ACCELERATION_CERTIFICATE.json','HISTORICAL_B_PREFIX_PLAN.json'):
            source=HERE/name
            if source.exists()and not(previous/name).exists():shutil.copyfile(source,previous/name)
    plan=validation_plan()
    mandatory=plan['只加安全投影']+plan['学习选修正参数']
    original=execute([r['request']for r in mandatory],stage('mandatory_A_replays'))
    results={query_uid(r):r for r in original};checks=[]
    for item in mandatory:
        uid=item['request']['rollout_uid'];old=item['archived_label'];new=results.get(uid)
        equal=new is not None and bool(old['success'])==bool(new['success'])and old['outcome']==new['outcome']
        checks.append(dict(method=old['method'],scene=old['scene'],state=old['state'],nominal=old['name'],
            target_query_uid=uid,archived_outcome=old['outcome'],new_outcome=None if new is None else new['outcome'],
            passed=equal,replayed_completion_time=None if new is None else new['completion_time']))
    for method,filename,required in (('只加安全投影','ENV_CHECK.json',200),('学习选修正参数','REUSE_CHECK.json',64)):
        subset=[r for r in checks if r['method']==method]
        dump(HERE/filename,dict(passed=len(subset)==required and all(r['passed']for r in subset),
            cases=len(subset),required=required,records=subset,
            legacy_completion_time_stored=False,
            time_evidence='Archived labels have no times. Fresh literal archived-kernel replays supply reference times; new common-path equality is checked in the paired certificate. Historical timestamps are not imputed.'))
    if not all(r['passed']for r in checks):
        dump(output,dict(complete=False,benchmark_A_eligible=False,reason='Mandatory archived labels failed or missing; inspect before retrying.'))
        raise RuntimeError('Mandatory A execution replay mismatch or infrastructure gap')
    b_plan=historical_b_plan()
    native=execute([r['request']for r in b_plan],stage('historical_B_prefix_replays'))
    b_results={query_uid(r):r for r in native};prefixes=[]
    for item in b_plan:
        uid=item['request']['rollout_uid'];row=b_results.get(uid);old=item['archived_label']
        comparison=compare_traces(item['copied_trace'],row['trace'],prefix=50)if row else dict(passed=False,reason='missing')
        equal=row is not None and all(row[k]==old[k]for k in ('success','collision','timeout','numerical_failure','episode_length'))
        prefixes.append(dict(target_query_uid=uid,source_rollout_uid=item['source_rollout_uid'],
                             prefix=comparison,full_outcome_and_length_equal=equal,passed=comparison['passed']and equal))
    archive_passed=all(r['passed']for r in prefixes)
    dump(HERE/'HISTORY_PREFIX_CHECK.json',dict(passed=archive_passed,records=prefixes,samples=len(prefixes),
        source_hash_screen='HISTORY_FINGERPRINT_SCREEN.json',
        note='Sampled deterministic execution checks; source UIDs and new query UIDs are kept distinct.'))
    # Mandatory A originals are reused as pair originals; no duplicate episodes.
    pairs=plan['只加安全投影'][:132]+plan['学习选修正参数'][:34]+b_plan
    assert len(pairs)==200
    originals={**results,**b_results};accelerated=[]
    for item in pairs:
        original_case=item['request']
        scene=catalog(original_case['benchmark'],version='original')[original_case['scene']]
        request=case(original_case['benchmark'],scene,original_case['controller_payload']['nominal'],
            original_case['state_index'],spec=original_case['spec'],accelerated=True,
            phase='certification',purpose='new_original_accelerated_pair')
        accelerated.append(request)
    fast_rows=execute(accelerated,stage('acceleration_200_pairs'))
    fast={query_uid(r):r for r in fast_rows};paired=[]
    for item,request in zip(pairs,accelerated):
        uid=item['request']['rollout_uid'];old=originals.get(uid);new=fast.get(request['rollout_uid'])
        comparison=compare_traces(old['trace'],new['trace'])if old and new else dict(passed=False,reason='missing')
        equal=bool(old and new and all(old[k]==new[k]for k in ('success','collision','timeout','numerical_failure','episode_length')))
        paired.append(dict(original_uid=uid,accelerated_uid=request['rollout_uid'],benchmark=request['benchmark'],
                           arrays=comparison,outcome_equal=equal,passed=comparison['passed']and equal))
    passed=all(r['passed']for r in paired)
    certificate=dict(passed=passed,planned_pairs=200,observed_pairs=sum(bool(r['arrays'].get('arrays'))for r in paired),records=paired,
        execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'),
        planned_original_cases=len(mandatory)+len(b_plan),
        reused_mandatory_A_originals=166,extra_B_originals=34,
        executed_requested_originals=len(mandatory)+len(b_plan),executed_requested_accelerated=len(accelerated),
        loader_and_input_hashes_equal_v3=read(HERE/'ACCELERATION_PREPARATION.json'),
        original_episode_seconds=sum(originals[r['request']['rollout_uid']]['wall_seconds']for r in pairs if r['request']['rollout_uid']in originals),
        accelerated_episode_seconds=sum(r['wall_seconds']for r in fast_rows),
        action='Enable four approved changes'if passed else'Disable acceleration and continue with original path; do not infer control failure.',
        cpu_only=True)
    dump(HERE/'ACCELERATION_CERTIFICATE.json',certificate)
    if not passed:
        with(HERE/'GAPS.md').open('a')as stream:stream.write('\n- 非主线：新加速配对未全部通过，关闭加速，后续用原路径。\n')
    if not archive_passed:
        with(HERE/'GAPS.md').open('a')as stream:stream.write('\n- 历史本地控制器前缀核对未全部通过，不复用相关旧标签，补跑所需训练局。\n')
    result=dict(complete=True,time=datetime.datetime.now().astimezone().isoformat(),benchmark_A_eligible=True,
                mandatory_A_labels_passed=True,new_acceleration_passed=passed,historical_B_reuse_eligible=archive_passed,
                execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'),
                history_index_sha256=sha(HERE/'OURS_HISTORY_INDEX.json'),
                fingerprint_screen_sha256=sha(HERE/'HISTORY_FINGERPRINT_SCREEN.json'),
                historical_time_evidence='Fresh replay reference, not previously stored completion timestamps.')
    dump(output,result)
    status('environment_reuse_certification',1,1,'强制历史标签、实际环境与新加速轨迹核对已保存；不通过的加速或历史复用保持关闭。')
    return result


if __name__=='__main__':run()


def main():
    import importlib,port_gates
    importlib.reload(port_gates)
    gates=port_gates.run()
    assert gates['complete'],'Aligned applicability/pilot gates incomplete'
    result=run()
    return result
