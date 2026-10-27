"""Saved-terminal static contact audit; no actor, projection, plant or journal."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import qualification
from correction_revision_20261021 import contact_mask,contact_radius
from states import LiteratureConfig
import numpy as np,collections,time

def candidates():
 paths=set(HERE.glob('*/raw/*.json'))
 for original in read(HERE/'LOCAL_DIAGNOSIS_SCOPE.json')['originals']:
  for entry in read(HERE/('analysis_'+original)/'GROUP_INDEX.json'):
   paths.update(Path(r['path']) for r in read(entry['file']))
 def order(p):
  s=p.parent.parent.name
  priority=0 if 'baseline' in s or 'fixed' in s or s.startswith(('safety_','fixed_')) else 1 if 'eval' in s else 2 if 'learning_labels' in s else 3
  return priority,str(p)
 return sorted(paths,key=order)

def main():
 if 'SLURM_JOB_ID' not in os.environ:os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 verify_receipt('REVISION_STATIC_AUDIT_CODE_20261021.receipt.json');verify_receipt('CORRECTION_REVISION_DEFINITION_20261021.receipt.json')
 started=time.monotonic();q={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES};seen_paths=set();seen_uids=set();cached={};groups={};hovers=[];sources=[];scanned=0
 while True:
  paths=[p for p in candidates() if str(p) not in seen_paths]
  if not paths:break
  for path in paths:
   seen_paths.add(str(path));data=path.read_bytes();raw=json.loads(data);rid=raw['rollout_uid']
   if rid in seen_uids:continue
   seen_uids.add(rid);c=raw['case'];scene=c['state']['scene'];layer=c.get('safety_layer',raw.get('safety_layer','central'));cfg=LiteratureConfig(**c['scenario_payload']['config']);n=c['state']['N'];method=c['method'];ctl=c['controller_payload']['nominal']['controller']
   folder=path.parent.parent.name;scope='capability' if 'capability' in folder else 'training_labels' if 'learning_labels' in folder else 'main_or_evaluation'
   qualified=ctl in q[scene];key=(scene,layer,method,scope,qualified)
   if key not in groups:groups[key]=dict(episodes=0,timeouts=0,wandering_episodes=0,unarrived_wandering_robots=0,contact_off_unarrived_wandering_robots=0)
   group=groups[key];group['episodes']+=1;group['timeouts']+=raw['timeout'];scanned+=1
   if raw['timeout']:
    distance=np.asarray(raw.get('final_goal_distances',[]),float)
    if not len(distance):
     with np.load(raw['trace_path']) as z:distance=np.linalg.norm(z['positions'][-1]-np.asarray(c['state']['goals']),axis=-1)
    unarrived=distance>cfg.goal_tolerance
    if np.any(unarrived) and np.all(distance[unarrived]<=3*cfg.goal_tolerance):
     radius=contact_radius(cfg);cachekey=(raw['trace_sha256'],radius,cfg.goal_tolerance)
     if cachekey not in cached:
      assert sha(raw['trace_path'])==raw['trace_sha256']
      with np.load(raw['trace_path']) as z:p=z['positions'][-1].copy()
      g=np.asarray(c['state']['goals']);computed=np.linalg.norm(g-p,axis=-1);assert np.array_equal(computed,distance),('Terminal distance mismatch',path)
      mask=contact_mask(p,g,cfg.goal_tolerance,radius);missing=computed>cfg.goal_tolerance
      delta=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(delta,np.inf)
      cached[cachekey]=dict(contact_open=mask.tolist(),unarrived=missing.tolist(),goal_distances=computed.tolist(),nearest_center_distances=delta.min(axis=1).tolist(),R_c=radius)
     info=cached[cachekey];missing=np.asarray(info['unarrived']);mask=np.asarray(info['contact_open']);assert np.array_equal(missing,unarrived)
     group['wandering_episodes']+=1;group['unarrived_wandering_robots']+=int(missing.sum());group['contact_off_unarrived_wandering_robots']+=int((missing&~mask).sum())
     hovers.append(dict(rollout_uid=rid,scene=scene,layer=layer,method=method,scope=scope,qualified=qualified,controller=ctl,raw=str(path),raw_sha256=hashlib.sha256(data).hexdigest(),trace_sha256=raw['trace_sha256'],**info))
   sources.append(dict(rollout_uid=rid,raw=str(path),raw_sha256=hashlib.sha256(data).hexdigest()))
   if scanned%2048==0:write(HERE/'CORRECTION_REVISION_STATIC_PROGRESS_20261021.json',dict(utc=utc(),processed_unique_saved_rollouts=scanned,wandering_episodes=len(hovers),unique_wandering_traces=len(cached),new_rollouts=0))
  # Rescan to include old-circle episodes that landed during this static audit.
 summary=[]
 for key,g in sorted(groups.items()):
  scene,layer,method,scope,qualified=key
  summary.append(dict(scene=scene,layer=layer,method=method,scope=scope,qualified=qualified,**g,wandering_fraction=g['wandering_episodes']/g['episodes'],contact_off_fraction_of_unarrived_wandering_robots=g['contact_off_unarrived_wandering_robots']/g['unarrived_wandering_robots'] if g['unarrived_wandering_robots'] else None))
 write(HERE/'CORRECTION_REVISION_STATIC_WANDERING_20261021.json',dict(utc=utc(),passed_descriptive_audit=True,definition='timeout and at least1 unarrived robot and all unarrived goal distances<=3*original goal_tolerance',snapshot='Every unique landed rollout UID in v2 raw plus all completed logical-stage reused raw sources at audit closure; later drain arrivals retained for final audit extension',processed_unique_saved_rollouts=scanned,unique_wandering_traces=len(cached),wandering_episodes=len(hovers),summary=summary,hovers=hovers,new_rollouts=0,new_training=0,no_Rc_tuning=True,seconds=time.monotonic()-started))
 write(HERE/'CORRECTION_REVISION_STATIC_SOURCES_20261021.json',sources)
 text='# 原语义目标附近徘徊与接触开关静态检查\n\n只读取已有保存证据的终止构型，没有调用控制器、投影或物理，没有新rollout。超时且存在未到达机器人、全部未到达距目标≤3×原到达容差即为徘徊；统计开关关闭的未到达机器人比例，仅描述，不据此调Rc。每个rollout UID只计一次；确定性两seed是不同UID，保留各自权重。能力门、训练菜单与主线/评估分开报告，JSON给出全部。\n\n| 场景 | 层 | 方法 | 合格 | 局数 | 徘徊局 | 比例 | 徘徊未到达机器人开关关闭比例 |\n|---|---|---|---|---:|---:|---:|---:|\n'
 for r in summary:
  if r['scope']=='main_or_evaluation' and r['method'] in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene'):
   rate=r['contact_off_fraction_of_unarrived_wandering_robots'];text+='| '+' | '.join(map(str,(NAMES[r['scene']],r['layer'],r['method'],r['qualified'],r['episodes'],r['wandering_episodes'],f"{100*r['wandering_fraction']:.4f}%",'N/A' if rate is None else f'{100*rate:.4f}%')))+' |\n'
 text+='\n全部原结果不修改。此次源清单与终态检查在单元门收据中封存；最终报告将同时报告原/修订语义与两个阶段的数据。\n'
 (HERE/'CORRECTION_REVISION_STATIC_WANDERING_20261021.md').write_text(text)
 seal('CORRECTION_REVISION_STATIC_WANDERING_20261021.receipt.json',[HERE/n for n in ('static_wandering_revision_20261021.py','CORRECTION_REVISION_STATIC_WANDERING_20261021.json','CORRECTION_REVISION_STATIC_WANDERING_20261021.md','CORRECTION_REVISION_STATIC_SOURCES_20261021.json','REVISION_STATIC_AUDIT_CODE_20261021.receipt.json')],new_rollouts=0,new_training=0,no_Rc_tuning=True)
 print(dict(processed=scanned,wandering=len(hovers),seconds=time.monotonic()-started),flush=True)
if __name__=='__main__':main()
