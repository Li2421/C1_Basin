"""Read-only selection/export/render of real full Ring simulation episodes."""
from pathlib import Path
import sys,json,hashlib,datetime,subprocess
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from new_benchmark_common.batch_videos import load_trace,render_video,sha256
OUT=P/'videos';OUT.mkdir(exist_ok=True);EXPORT=OUT/'source_traces';EXPORT.mkdir(exist_ok=True)
BATCH='ring_correction_video_examples_20261020'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def select():
 selected=[];safe=None;counts={}
 for dn,kinds in [('crosstype1c_two_eta_schedule_v1',['Flow','gaussian','diffusion']),('crosstype3_unseen_controller_schedule_v1',['transformer_bc'])]:
  d=ROOT/'diagnostics'/dn;index=read(d/'NEW_EVAL_METHOD_UIDS.json');pool=read(d/'new_evaluation_protocol.json');config=pool['native_manifest']['scenario_config']
  for kind in kinds:
   candidates=[]
   for r in index:
    if r['type']!=kind:continue
    for q in (0,1):
     pf=d/'new_evaluation/raw'/f"{r['method_uids']['self_schedule'][q]}.json";bf=d/'new_evaluation/raw'/f"{r['method_uids']['all_safety_baseline'][q]}.json"
     if not (pf.exists() and bf.exists()):continue
     s=read(pf);b=read(bf)
     if not s['success'] or s['numerical_failure'] or s['collision']:continue
     if b['success'] or b['numerical_failure'] or b['collision'] or b['outcome']!='timeout':continue
     with np.load(b['trace_path'],allow_pickle=False) as z:
      pos=z['positions'];goals=z['goals'];tail=pos[-201:];unfinished=np.linalg.norm(tail[1:]-goals,axis=-1)>config['goal_tolerance'];moves=np.linalg.norm(np.diff(tail,axis=0),axis=-1);freeze=float(np.sum((moves<1e-4)&unfinished)/max(1,unfinished.sum()));remaining=int(np.sum(np.linalg.norm(pos[-1]-goals,axis=-1)>config['goal_tolerance']));net=float(np.sum(np.linalg.norm(tail[0]-goals,axis=-1)-np.linalg.norm(tail[-1]-goals,axis=-1)))
     candidates.append(dict(type=kind,controller=r['controller'],state_uid=r['state_uid'],seed=q,baseline_record=str(bf),correction_record=str(pf),source_experiment=str(d),tail_freeze_fraction_unfinished=freeze,tail_goal_progress=net,unfinished_robots=remaining))
   assert candidates,kind
   candidates.sort(key=lambda c:(-c['tail_freeze_fraction_unfinished'],-c['unfinished_robots'],abs(c['tail_goal_progress']),c['controller'],c['state_uid'],c['seed']))
   selected.append(candidates[0]);counts[kind]=len(candidates)
  if safe is None:
   for r in index:
    for q,rid in enumerate(r['method_uids']['all_safety_baseline']):
     f=d/'new_evaluation/raw'/f'{rid}.json'
     if not f.exists():continue
     raw=read(f)
     if raw['success'] and not raw['collision'] and not raw['numerical_failure']:
      safe=dict(type=r['type'],controller=r['controller'],state_uid=r['state_uid'],seed=q,record=str(f),source_experiment=str(d));break
    if safe is not None:break
 write(P/'SELECTED_CASES.json',dict(selected=selected,independent_safety_success=safe,candidate_counts=counts,selection='visual examples selected for high tail freeze among unchanged successful correction/failed Safety pairs; descriptive only, not aggregate performance',no_rollout_generated=True))
 return selected,safe

def export(record_path,kind,label):
 raw=read(record_path);source=Path(raw['trace_path']);assert sha256(source)==raw['trace_sha256'],source
 with np.load(source,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 d=Path(record_path).parents[2];p=read(d/'new_evaluation_protocol.json');cfg=p['native_manifest']['scenario_config'];c=raw['case'];profile=p['profiles'][c['base_controller_name']]['payload']
 initial={k:raw[k] for k in ('initial_positions','initial_velocities','goals','rng_namespace','future_index','future_root','original_uid')};initial['config']=cfg
 assert len(a['positions'])==raw['episode_length']+1
 assert np.array_equal(a['positions'][0],np.asarray(raw['initial_positions']))
 assert len(a['swept_clearance'])==raw['episode_length']
 ckpt=profile.get('flow_checkpoint_path') or 'nonlearned controller';cksha=profile.get('flow_checkpoint_sha256') or 'none'
 meta=dict(batch_id=BATCH,scenario='ring_exchange',rollout_id=raw['rollout_uid'],controller=c['base_controller_name']+' / '+cksha[:8],controller_uid=raw['controller_uid'],checkpoint=str(ckpt)+' sha256='+cksha,seed=c['seed'],source_record=str(Path(record_path).resolve()),source_record_sha256=sha256(record_path),source_trace=str(source),source_trace_sha256=sha256(source),initial_state_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest(),config=cfg,start_step=0,episode_steps=raw['episode_length'],complete=True,termination=raw['outcome'],collision=bool(raw['collision']),safety_enabled=True,deadlock_detected=False,deadlock_criterion=None,display_label='Ring '+kind,method=label,eta_a=c['eta_a'],eta_b=c['eta_b'],switch_step=c['switch_step'],switch_mode=c['switch_mode'],trace_preserves_every_step=True)
 dest=EXPORT/f"{raw['rollout_uid']}.npz";np.savez_compressed(dest,positions=a['positions'],velocities=a['velocities'],steps=np.arange(len(a['positions'])),goals=a['goals'],walls=np.empty((0,2,2)),swept_clearance=a['swept_clearance'],metadata_json=np.array(json.dumps(meta)))
 loaded_meta,arrays=load_trace(dest);return dest,loaded_meta,arrays

def main():
 selected,safe=select();records=[]
 for i,s in enumerate(selected,1):
  bp,bm,bd=export(s['baseline_record'],s['type'],'Safety baseline');sp,sm,sd=export(s['correction_record'],s['type'],'pipline (learned two eta and switch)')
  assert bm['seed']==sm['seed'] and bm['config']==sm['config'] and bm['initial_state_sha256']==sm['initial_state_sha256']
  assert np.array_equal(bd['positions'][0],sd['positions'][0]) and np.array_equal(bd['velocities'][0],sd['velocities'][0]) and np.array_equal(bd['goals'],sd['goals'])
  out=OUT/f"{i:02d}_{s['type']}_failure_vs_pipline.mp4"
  frames=render_video(out,sm,sd,baseline=(bp,bm,bd),panel_labels=('Safety baseline | FAILED: '+bm['termination'],'pipline + safety | SUCCESS'))
  records.append(dict(file=out.name,role='failure_vs_correction_success',type=s['type'],frames=frames,fps=1/sm['config']['dt'],duration_seconds=frames*sm['config']['dt'],sha256=sha256(out),baseline_trace=str(bp.relative_to(OUT)),correction_trace=str(sp.relative_to(OUT)),baseline=bm,correction=sm,selection_diagnostics=s))
  print('rendered',out.name,'frames',frames,flush=True)
 assert safe is not None
 tp,tm,td=export(safe['record'],safe['type'],'Safety only success');out=OUT/'05_independent_safety_success.mp4';frames=render_video(out,tm,td,panel_labels=('Safety baseline | SUCCESS',));records.append(dict(file=out.name,role='safety_success',frames=frames,fps=1/tm['config']['dt'],sha256=sha256(out),trace=str(tp.relative_to(OUT)),metadata=tm))
 # Verify every delivered MP4 decodes and preserves the complete simulator frame count.
 for r in records:
  v=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(OUT/r['file'])]));assert int(v['streams'][0]['nb_read_frames'])==r['frames'];r['ffprobe']=v
 write(OUT/'manifest.json',dict(batch_id=BATCH,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=True,scope='supplementary Ring failure-vs-correction examples; not formal deadlock batch',formal_deadlock_evidence=False,formal_deadlock_gap='Ring has no recorded formal deadlock detector; baseline termination is shown unchanged as timeout',no_new_rollouts=True,no_physics_or_controller_changes=True,playback='real time; dt0.05;20fps; every simulator state included once; shorter side held at terminal state',videos=records))
 text='''# Ring：失败与 correction 成功对比\n\n本目录共5个完整仿真MP4：4组左右对比，另附1个独立的安全成功案例。\n\n- 左侧：原controller仅开启safe filter，本例最终timeout。\n- 右侧：同controller、同初态、同seed，学习双η与切换时间表（pipline），并开启相同safe filter，最终无碰撞成功。\n- Ring没有正式deadlock判据，因此这些是失败/停滞/超时与成功的对比，不标作已证实deadlock。\n- 全程原速20fps，每个仿真状态保留一帧；较早结束的一侧停留在真实终态，另一侧播放到终止。\n- 只读取已有仿真轨迹，未补跑、未改变物理、controller、事件或成功标准。\n- 这是为展示挑选的案例，不代表总体成功率。筛选依据及逐条输入、checkpoint、η、切换步、原始终止事件和SHA256见manifest.json。\n- source_traces/包含完整导出证据，可追溯到manifest中的原始记录。\n'''
 (OUT/'README.md').write_text(text)
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
