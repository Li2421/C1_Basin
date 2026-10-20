from common import *
def main():
 per={}
 for stage in ['pilot','source','target']:
  if not (HERE/f'{stage}_CASES.json').exists():continue
  for c in read(HERE/f'{stage}_CASES.json'):
   q=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");key=f"N{c['N']}_{c['kind']}";d=per.setdefault(key,dict(step_times=[],crossing_fraction=[],throughput_per_second=[],final_goal_fraction=[],physical_episodes=0))
   if q['deterministic_source'] is not None:continue
   with np.load(q['trace_path']) as z:
    times=z['step_seconds'];d['step_times'].append(times);pos=z['positions'];goals=z['goals'];cfg=q['actual_config'];gate=cfg['barrier_x'][0];direction=np.sign(goals[:,0]-gate);exit=cfg['barrier_thickness']/2+cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin'];crossed=np.any((pos[:,:,0]-gate)*direction[None]>exit,axis=0);duration=(len(pos)-1)*cfg['dt'];d['crossing_fraction'].append(float(crossed.mean()));d['throughput_per_second'].append(float(crossed.sum()/duration) if duration else None);d['final_goal_fraction'].append(q['final_goal_fraction']);d['physical_episodes']+=1
 for k,d in per.items():
  times=np.concatenate(d.pop('step_times')) if d['step_times'] else np.array([]);d.update(completed_native_steps=len(times),mean_step_seconds=float(times.mean()) if len(times) else None,p95_step_seconds=float(np.quantile(times,.95)) if len(times) else None,mean_gate_crossing_fraction=float(np.mean(d['crossing_fraction'])),mean_final_goal_fraction=float(np.mean(d['final_goal_fraction'])),scope='All fresh pilot/main arms in this class, deterministic copies excluded; completed-step times include first JIT calls, episode elapsed separately includes failed-solver time')
 write(HERE/'PER_STEP_PERFORMANCE.json',per)
if __name__=='__main__':main()
