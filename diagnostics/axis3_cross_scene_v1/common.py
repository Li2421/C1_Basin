from core import *
EXP='exp_axis3_cross_scene_v1'
SCENES=['toy_give_way','double_bottleneck','four_way_intersection','ring_exchange']
ARMS=['nominal_only','safety_baseline','F10','F13','F15','S13_t1','S13_t2','S15_t1','S15_t2'];MENU=ARMS[2:]
from shared_rollout_db.src import rollout_db,cache_writer,planner
for m in [rollout_db,cache_writer,planner]:assert Path(m.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db'),m.__file__
assert cache_writer.ROOT.resolve()==(ROOT/'shared_rollout_db').resolve()
os.environ['CUDA_CACHE_PATH']=str(HERE/'cuda_cache');(HERE/'cuda_cache').mkdir(exist_ok=True)
R13=D/'cross_scene_eta_schedule_v1';R19=D/'four_controller_axis_v2';R20=D/'cross_scene_probe_selection_v1'
