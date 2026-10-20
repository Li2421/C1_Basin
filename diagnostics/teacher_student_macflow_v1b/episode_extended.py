"""Connect extra arms through runtime.action; the frozen episode loop is unchanged."""
from core import *
from episode_frozen import episode as frozen_episode
from actions import nominal_only,right_hand_local
class ArmRuntime:
    def __init__(self,base):self.base=base;self.additional_mode=None
    def __getattr__(self,name):return getattr(self.base,name)
    def nominal(self,env,key,traced=False):return self.base.nominal(env,key,traced=traced)
    def action(self,env,key,eta,chain='old'):
        if self.additional_mode=='nominal':a,d=nominal_only(self.base,env,key,eta,np.zeros(4,int))
        elif self.additional_mode=='rhr_local':a,d=right_hand_local(self.base,env,key,eta,self.slow_counts)
        else:return self.base.action(env,key,eta,chain)
        return a,dict(min_linear_residual=d['minimum_local_linear_residual']),d['projection_calls']
class CountedRuntime(ArmRuntime):
    # The simulator exposes current velocities. Track prior low-speed states locally,
    # matching the frozen loop's post-step counter before the next action.
    def __init__(self,base):super().__init__(base);self.last_step=0;self.slow_counts=np.zeros(4,int)
    def action(self,env,key,eta,chain='old'):
        if env.step_count==0:self.last_step=0;self.slow_counts[:]=0
        elif env.step_count!=self.last_step:
            mask=(np.linalg.norm(env.velocities,axis=1)<.01)&(np.linalg.norm(env.goals-env.positions,axis=1)>env.config.goal_tolerance);self.slow_counts=np.where(mask,self.slow_counts+1,0);self.last_step=env.step_count
        return super().action(env,key,eta,chain)
def episode(c,p,cache,phase):
    routed=dict(c)
    if c['pipeline'] in ['nominal','rhr_local']:
        rt=cache[c['base_controller_name']];rt.additional_mode=c['pipeline'];routed['pipeline']='central'
    result=frozen_episode(routed,p,cache,phase)
    result['case']=c;result['protocol_sha256']=sha(HERE/f'{phase}_protocol.json');result['execution_code_sha256']=sha(__file__)
    # Correct metadata for the extra arms; physical trace arrays remain unchanged.
    path=Path(result['trace_path'])
    if c['pipeline'] in ['nominal','rhr_local']:
        with np.load(path) as z:values={k:z[k] for k in z.files}
        meta=json.loads(str(values['metadata_json']));meta['safety_enabled']=c['pipeline']!='nominal';meta['display_label']=c['base_controller_name']+' | '+c['pipeline']+' | '+c['arm'];values['metadata_json']=np.array(json.dumps(meta))
        if c['pipeline']=='nominal':values['neighbor_counts']=np.zeros((result['physical_steps'],4),int)
        else:
            positions=values['positions'][:-1];values['neighbor_counts']=np.stack([((np.linalg.norm(q[:,None]-q[None,:],axis=-1)<=1.5)&(~np.eye(4,dtype=bool))).sum(1) for q in positions]);vel=values['velocities'][1:];goals=values['goals'];counts=np.zeros(4,int);active=[]
            for t,q in enumerate(positions):
                active.append((counts>=40)&(np.linalg.norm(goals-q,axis=1)>cache[c['base_controller_name']].config.goal_tolerance));mask=(np.linalg.norm(vel[t],axis=1)<.01)&(np.linalg.norm(goals-values['positions'][t+1],axis=1)>cache[c['base_controller_name']].config.goal_tolerance);counts=np.where(mask,counts+1,0)
            values['rhr_active']=np.asarray(active,bool)
        np.savez_compressed(path,**values);result['trace_sha256']=sha(path)
    return result
