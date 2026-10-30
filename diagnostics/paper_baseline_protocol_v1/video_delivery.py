"""Study-specific full-trace videos using the recovered shared renderer only."""
import guard
from guard import HERE
import json,hashlib,importlib.util,datetime,traceback
from pathlib import Path
import numpy as np
from method_names import NAMES


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run():
    output=HERE/'VIDEO_DELIVERY.json'
    if output.exists()and read(output).get('finished'):return read(output)
    analysis=read(HERE/'ANALYSIS_PLAN_FROZEN.json');native=read(HERE/'NATIVE_REPRODUCTION_RESULTS.json')['methods']
    methods=['学习选修正参数']+[m for m,r in native.items()if r['reproducible']]
    evidence=read(HERE/'NORMALIZED_EVIDENCE.json')['rows'];names=read(HERE/'SCENE_NAMES.json');selected=[];wanted=set()
    for category,definition in analysis['videos']['selection'].items():
        for method in methods:
            rows=[r for r in evidence if r['benchmark']=='A'and r['part']=='test'and r['scene']==definition['scene']and r['state']==definition['state']
                  and r['method']==method and r['variant']==('frozen'if method=='学习选修正参数'else'global_tuned')
                  and(method!='学习选修正参数'or r['nominal']==definition['nominal'])]
            assert len(rows)<=1,'HARD_STOP: duplicate prespecified video evidence'
            row=rows[0]if rows else None;result=row.get('result')if row else None;trace=result.get('trace')if result else None
            uid=(result.get('rollout_uid')or(Path(trace).stem if trace else None))if result else None
            selected.append(dict(category=category,scene=definition['scene'],state=definition['state'],method=method,
                method_display=NAMES.get(method,method),nominal=definition['nominal']if method=='学习选修正参数'else'paper',
                uid=uid,evidence=row,source_trace=trace))
            if uid:wanted.add(uid)
    parents={};patterns=('test_A_*_planned.json','frozen_ours_consistent_geometry_planned.json','conditional_old_success_time_backfill_planned.json')
    for pattern in patterns:
        for path in sorted((HERE/'plans').glob(pattern)):
            for request in read(path)['exact_configs']:
                if request['rollout_uid']in wanted:parents.setdefault(request['rollout_uid'],request)
    renderer_path=HERE/'video_tools/batch_videos_generic.py';provenance=read(HERE/'video_tools/PROVENANCE.json')
    assert sha(renderer_path)==provenance['adapted_sha256'],'HARD_STOP: recovered renderer changed'
    spec=importlib.util.spec_from_file_location('study_shared_video_renderer',renderer_path);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    records=[];folder=HERE/'videos';folder.mkdir(exist_ok=True)
    for item in selected:
        identity={k:item[k]for k in ('category','scene','state','method','method_display','nominal','uid','source_trace')}
        try:
            uid=item['uid'];request=parents.get(uid);source=item['source_trace']
            if not source or not Path(source).is_file():raise FileNotFoundError('Prespecified historical/raw trace unavailable; no re-rollout or alternate state.')
            if request is None:raise FileNotFoundError('Exact parent request unavailable; do not guess controller/checkpoint/configuration.')
            result=item['evidence']['result'];raw=HERE/'raw'/(uid+'.json')
            if raw.is_file():
                actual=read(raw)
                assert(actual['success'],actual['outcome'])==(result['success'],result['outcome']),'Timing replay disagrees with historical label'
                result=actual
            source_sha=sha(source);receipt=folder/'receipts'/(uid+'.json')
            if result.get('trace_sha256'):assert result['trace_sha256']==source_sha,'Source trace hash differs from actual raw record'
            path=folder/(item['category']+'__'+('ours'if item['method']=='学习选修正参数'else item['method'])+'.mp4')
            if receipt.exists():
                prior=read(receipt)
                if path.is_file()and prior['source_trace_sha256']==source_sha and prior['video_sha256']==sha(path)and prior['renderer_sha256']==sha(renderer_path):
                    records.append(dict(identity,rendered=True,receipt=prior,reused=True));continue
            with np.load(source,allow_pickle=False)as saved:
                arrays={k:saved[k]for k in ('positions','goals','swept_clearance')}
                if'velocities'in saved:arrays['velocities']=saved['velocities']
            assert np.array_equal(arrays['positions'][0],np.asarray(request['state']['positions'])),'Source initial positions disagree'
            assert np.array_equal(arrays['goals'],np.asarray(request['state']['goals'])),'Source goals disagree'
            steps=len(arrays['positions'])-1;assert steps==result['episode_length'],'Source terminal frame count differs'
            config=request['scenario_payload']['config'];nominal=request['controller_payload']['nominal'];checkpoint=nominal.get('checkpoint')
            checkpoint_id=nominal['checkpoint_sha256']if item['method']=='学习选修正参数'else{'dac_author_si':'b6cf71fdce40','mgr_author_unicycle_tracking':'84aaefd705ab'}[item['method']]
            checkpoint=checkpoint or'nonlearned author source '+checkpoint_id
            controller_display=item['method_display']
            if item['method']=='学习选修正参数':
                model=nominal['controller'];kind=next(k for k in ('mlp','diffusion','attention')if k in model)
                label={'mlp':'多层感知机行为克隆','diffusion':'扩散策略','attention':'注意力规划器'}[kind]
                controller_display+=' / '+label+'·种子'+('1'if'_s1_'in model else'0')
            full_initial=dict(state=request['state'],scenario=request['scenario_payload'],controller=request['controller_payload'],
                seed=request['seed'],known_simulator_reset=dict(step_count=0,done=False,collided=False))
            meta=dict(batch_id='paper_baseline_protocol_v1',scenario='heterogeneous_scene',display_label=names[item['scene']]['display'],
                role='method_rollout',rollout_id=uid,controller=request['controller_uid'],controller_display=controller_display,
                checkpoint=checkpoint,checkpoint_label=checkpoint_id,seed=request['seed'],source_record=str(HERE/'raw'/(uid+'.json')),
                initial_state_sha256=hashlib.sha256(json.dumps(full_initial,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                config=config,start_step=0,episode_steps=steps,complete=True,termination=result['outcome'],collision=bool(result['collision']),
                safety_enabled=True,source_trace=source,source_trace_sha256=source_sha,
                interpretation='Actual simulator outcome retained; not labeled deadlock. Full physical initial state, controller/checkpoint and seed recorded. No simulation rerun.')
            arrays.update(steps=np.arange(steps+1),walls=np.asarray(request['scenario_payload']['walls'],float).reshape(-1,2,2),metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)))
            exported=folder/'exported_traces'/(uid+'.npz');exported.parent.mkdir(exist_ok=True);np.savez_compressed(exported,**arrays)
            verified,data=renderer.load_trace(exported);frames=renderer.render_video(path,verified,data)
            proof=dict(source_trace=source,source_trace_sha256=source_sha,exported_trace=str(exported),exported_trace_sha256=sha(exported),
                video=str(path),video_sha256=sha(path),frames=frames,expected_frames=steps+1,dt=config['dt'],fps=1/config['dt'],metadata=meta,
                renderer_sha256=sha(renderer_path),all_saved_states_once=True,frame_count_verified=True)
            save(receipt,proof);records.append(dict(identity,rendered=True,receipt=proof,reused=False))
        except Exception as exc:
            if isinstance(exc,PermissionError)or'HARD_STOP'in str(exc):raise
            records.append(dict(identity,rendered=False,error=repr(exc),traceback=traceback.format_exc()))
        save(HERE/'VIDEO_PROGRESS.json',dict(time=datetime.datetime.now().astimezone().isoformat(),processed=len(records),planned=len(selected),rendered=sum(r['rendered']for r in records)))
    result=dict(finished=True,complete=all(r['rendered']for r in records)and len(records)==len(selected),
        planned=len(selected),rendered=sum(r['rendered']for r in records),records=records,
        shared_renderer_provenance=provenance,new_rollouts=0,
        interpretation='Private recovered shared renderer with generic-scene validation and the prespecified study selection. No claim that the legacy mandatory four-scene/deadlock-role gate was passed. Missing traces/cases remain gaps; no alternate favorable examples or fabricated frames.')
    save(output,result)
    with(HERE/'REPORT.md').open('a')as stream:stream.write('\n## 非主线完整轨迹视频\n\n按冻结的每类首场景、首测试状态及固定名义控制器，已渲染'+str(result['rendered'])+'/'+str(result['planned'])+'段。私有副本复用历史共享渲染器，每个实际状态一次、原dt播放，并核验编码帧数；终止事件照录，不把超时标死锁。缺口、来源SHA和视频清单见VIDEO_DELIVERY.json，旧四场景专用验收不冒充本轮验收。\n')
    return result


if __name__=='__main__':print(json.dumps({k:v for k,v in run().items()if k!='records'},ensure_ascii=False))
