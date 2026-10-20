from core import *
from guard import guard
import shutil,difflib
G=guard()
def main():
 assert (HERE/'INTEGRITY_BEGIN.json').exists();records=[]
 files={
 'pipeline.py':D/'teacher_student_macflow_v1d/pipeline.py','episode_frozen.py':D/'teacher_student_macflow_v1d/episode_frozen.py','fair_rhr.py':D/'teacher_student_macflow_v1d/fair_rhr.py','registration.py':D/'teacher_student_macflow_v1d/registration.py',
 'controllers.py':D/'axis2_controller_v1/controllers.py','legacy_controllers.py':D/'axis2_controller_v1/legacy_controllers.py','gaussian.py':D/'axis2_controller_v1/gaussian.py',
 'critic_fit.py':D/'teacher_student_macflow_v1f/critic_fit.py','freeze_choices.py':D/'teacher_student_macflow_v1f/freeze_choices.py','audit_utils.py':D/'teacher_student_macflow_v1f/audit_utils.py',
 'metrics.py':D/'teacher_student_line_closeout_v1/metrics.py','stats.py':D/'teacher_student_line_closeout_v1/stats.py',
 'core.py':D/'teacher_student_macflow_v1f/core.py','guard.py':D/'teacher_student_macflow_v1f/guard.py'}
 for dst,src in files.items():
  if not (HERE/dst).exists():shutil.copyfile(src,HERE/dst)
  diff=''.join(difflib.unified_diff(src.read_text().splitlines(True),(HERE/dst).read_text().splitlines(True),fromfile=str(src),tofile=dst));records.append(dict(source=str(src),sha256=sha(src),copy=dst,copy_sha256=sha(HERE/dst),diff=diff))
 for dst,src in [('EVAL_STATES.json',D/'teacher_student_macflow_v1f/EVAL_STATES.json'),('PILOT_STATES.json',D/'teacher_student_macflow_v1/PILOT_STATES.json'),('TUNING_STATES.json',D/'teacher_student_macflow_v1c/TUNING_STATES.json')]:shutil.copyfile(src,HERE/dst)
 p=read(D/'axis2_controller_v1/NEW_BASE_PROTOCOL.json');names=[f'DM{i}' for i in range(6)]+[f'AS{i}' for i in range(6)]+[f'G{i}' for i in range(6)];profiles={n:p['profiles'][n] for n in names if n in p['profiles']};grid=[(.05,10,False),(.05,10,True),(.05,50,False),(.05,50,True),(.10,10,True),(.10,50,True)]
 for i,g in enumerate(grid):profiles[f'AS{i}']=dict(payload=dict(controller_class='AS',nominal_parameters=g,scenario_uid=p['profiles']['DM0']['payload']['scenario_uid']))
 for n in [f'G{i}' for i in range(6)]:
  v=profiles[n]['payload'];assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256'];assert sha(Path(v['flow_checkpoint_path']).parent/'NORMALIZATION.json')==v['normalization_sha256']
 write(HERE/'CONTROLLERS.json',dict(controller_order=names,profiles=profiles,native_manifest=p['native_manifest']))
 from registration import normalized_state
 write(HERE/'PROBE_STATES.json',[normalized_state(s) for s in read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states']])
 from fair_rhr import grid as rgrid
 rules=rgrid()+[dict(mode='neighbor_right',distance=d,speed=v,steps=0) for d in [.8,1.2] for v in [.10,.25]];write(HERE/'RHR_GRID.json',rules)
 write(HERE/'COPY_PROVENANCE.json',records)
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),sources=[str(D/x) for x in ['axis2_controller_v1','decentralized_pipeline_v1','teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c','teacher_student_macflow_v1d','teacher_student_macflow_v1f','teacher_student_line_closeout_v1']],reuse=['14 original distilled students: all seven arms, frozen predictions, and mechanisms reuse round30g manifest','Round33 Gaussian checkpoints and exact distributed random shooting implementations; no training','Round28 state/RNG aliases, frozen safety/corrector, frozen critic weights and transforms','Round30g metrics.py and stats.py copied byte for byte'],missing=['New A* implementation has no prior trajectories','Round33 joint safety labels incompatible with decentralized local safety; nominal-only exact physical/RNG matches eligible for reuse','R5 probes under decentralized safety missing for these families','24-rule per-family tuning on round30c states missing'],no_new_rollouts_yet=True,guard=guard()))
 print('SETUP PASS')
if __name__=='__main__':main()
