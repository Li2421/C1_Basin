"""Read-only checkpoint/runtime extraction and analytic geometry arithmetic; no rollout."""
import os
for k,v in {'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}.items(): os.environ[k]=v
import sys,json,hashlib,time,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
F=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
for p in (F,F/'source',F/'source/scripts'):sys.path.insert(0,str(p))
import numpy as np
from validation_v2.adapters import OtherRuntime
from single_integrator.environment import Config,GiveWayEnv
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    raise TypeError(type(o))
def main():
    start=time.monotonic();rt=OtherRuntime('toy_give_way');c=rt.config;env=GiveWayEnv(c)
    prior=ROOT/'diagnostics/tt_ff_correction_realization_20261017_v1/pre_evaluation_audit.json'
    audit=json.loads(prior.read_text()); hashes={p:sha(p) for p in audit['source_hashes']}
    assert hashes==audit['source_hashes'],'Frozen source identity changed'
    R=c.agent_radius+c.wall_radius+c.wall_collision_margin+rt.cbf.separation_buffer
    Ds=2*c.agent_radius+c.agent_collision_margin+rt.cbf.separation_buffer
    w=c.corridor_width/2; L=c.corridor_half_length; h=w-R;H=c.bay_top-R;a=L-R
    assert c.wall_layout_scale==1
    contract={'status':'PROVEN analytic finite-segment geometry with checkpoint-loaded constants',
      'frozen_root':str(F),'checkpoint':str(rt.checkpoint),'checkpoint_sha256':sha(rt.checkpoint),
      'runtime_config':c.to_dict(),'source_default_config':Config().to_dict(),'cbf_config':rt.cbf.to_dict(),
      'source_hashes':hashes,'reused_evidence':str(prior),'reused_evidence_sha256':sha(prior),
      'walls':[{'name':n,'endpoints':v} for n,v in zip(env.wall_names,env.walls)],'goals':env.goals,
      'constants_m':{'R_w':R,'D_s':Ds,'w':w,'L':L,'h':h,'H':H,'a':a,'b_minus':-h,'b_plus':h,
        'single_agent_x_min':-a,'single_agent_x_max':a,'single_agent_y_min':-h,'single_agent_y_max':H,
        'main_channel_full_center_height':2*h,'bay_full_center_width':2*h,'max_relative_y_if_y1_le0':H+h,
        'S_full_cartesian_x2_s_max':a-h,'D_exit_conservative_choice':Ds},
      'free_space_cross_section':{'-h <= y <= h':'abs(x)<=a',
        'h < y < w':'abs(x)<=w-sqrt(R_w**2-(w-y)**2)',
        'w <= y <= H':'abs(x)<=h'},
      'free_space_boundary_note':'At y=h only the first branch applies: full |x|<=a. The one-sided width above h tends to w, so this section jumps. The full channel safe-ceiling line is a legal zero-area boundary branch.',
      'joint_constraints':['p1 and p2 in cross-section Omega','s=x1-x2','z=y2-y1','s*s+z*z >= D_s*D_s'],
      'positive_volume_entry_box':{'x1':[-.55,-.40],'x2':[-.02,.02],'y1':[-.01,0.],'y2':[0.,.01],
         'wall_h_lower_bound':.19-R,'pair_distance_lower_bound':.38,'volume_m4':.15*.04*.01*.01},
      'positive_volume_pass_box':{'x1':[-.55,.55],'x2':[-.02,.02],'y1':[-.01,0.],'y2':[.38,.42],
         'wall_h_lower_bound':.18-R,'pair_distance_lower_bound':.38,'volume_m4':1.1*.04*.01*.04},
      'new_rollouts':0,'new_projection_calls':0,'new_training':0,'wall_seconds':time.monotonic()-start}
    (HERE/'geometry_contract.json').write_text(json.dumps(contract,indent=2,default=serial)+'\n')
    print(json.dumps({'constants':contract['constants_m'],'wall_seconds':contract['wall_seconds']}))
if __name__=='__main__':main()
