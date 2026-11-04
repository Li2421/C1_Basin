"""Shadow original/vector lidar on actual native-clock saved observations.

No controller actions, QPs, new closed-loop traces or approximate caching.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.sched_setaffinity(0,{23})
from clock_plant import *
from circulation_port_v2 import m,vector_lidar

def main():
    cases=[c for c in read(D/'ACTIVE_NATIVE_CASES.json') if c['payload']['method']=='circulation' and (D/'raw'/(c['rollout_uid']+'.json')).exists()]
    # Select by geometry/size only, using all available structural categories.
    categories={}
    for c in cases:
        cfg=c['scenario']['config'];key=(len(c['state']['positions']),len(c['scenario']['walls']),cfg['agent_radius'],bool(c['scenario'].get('external_motion')))
        categories.setdefault(key,c)
    chosen=sorted(categories.values(),key=lambda c:(-len(c['state']['positions']),-len(c['scenario']['walls']),c['name']))[:12]
    # Always include prescribed moving disks, independently of size ordering.
    moving=next((c for c in cases if c['scenario'].get('external_motion')),None)
    if moving and moving not in chosen:chosen.append(moving)
    checks=[];queries=0
    for c in chosen:
        r=read(D/'raw'/(c['rollout_uid']+'.json'));env=ClockPlant(c['scenario'],c['state'],.02,c['payload'].get('external_root'));scale=.1/env.config.agent_radius
        sensor=m.DetectObstacle(detect_max_dist=m.SceneSetup.sense_dist,angle_res_rad=np.pi/180)
        for j,wall in enumerate(env.walls):sensor.register_obstacle_bounded('wall'+str(j),np.c_[wall*scale,np.zeros(2)])
        angles=m.SimSetup.robot_angle_bound;unit=np.stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))],axis=1)
        with np.load(r['trace']) as z:
            for k in sorted({0,len(z['control_times'])//2,len(z['control_times'])-1}):
                t=z['control_times'][k];idx=int(np.flatnonzero(z['times']==t)[0]);pos=np.c_[z['positions'][idx]*scale,np.zeros(env.num_agents)]
                for j,p in enumerate(env.external_at(frac(t))):sensor.register_obstacle_bounded('external_'+str(j),unit*env.external_radius*scale+np.r_[p*scale,0.])
                for i in range(env.num_agents):sensor.register_obstacle_bounded(i,unit*.1+pos[i])
                for i in sorted({0,env.num_agents-1}):
                    args=(pos[i,0],pos[i,1],0.)
                    original=m.DetectObstacle.get_sensing_data(sensor,*args,exclude=[i]);fast=vector_lidar(sensor,*args,exclude=[i])
                    assert original.dtype==fast.dtype and original.shape==fast.shape and original.tobytes()==fast.tobytes(),(c['name'],k,i)
                    queries+=1
        checks.append(dict(case=c['name'],uid=c['rollout_uid'],agents=env.num_agents,walls=len(env.walls),external_disks=env.external.shape[1],passed=True))
    out=dict(passed=True,cases=len(checks),queries=queries,beams=360,full_arrays='bytewise identical',checks=checks,
        original_certificate=str(OLD/'LIDAR_VECTOR_CERTIFICATE.json'),acceleration_source_sha256=sha(OLD/'circulation_port_v2.py'),
        new_controller_calls=0,new_rollouts=0,scope='exact sensor kernel at actual saved native-clock inputs; full run outcomes checked separately')
    dump(D/'NATIVE_LIDAR_QUERY_CERTIFICATE.json',out);print('native lidar shadow passed',len(checks),queries)

if __name__=='__main__':main()
