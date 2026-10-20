from core import *
EXP='exp_axis4a_geometry_horizon_v1'
from shared_rollout_db.src import rollout_db,cache_writer,planner
for m in [rollout_db,cache_writer,planner]:assert Path(m.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db'),m.__file__
assert cache_writer.ROOT.resolve()==(ROOT/'shared_rollout_db').resolve()
os.environ['CUDA_CACHE_PATH']=str(HERE/'cuda_cache');(HERE/'cuda_cache').mkdir(exist_ok=True)
R10=D/'controller_universal_eta_schedule_v1';R16=D/'ring_controller_critic_v1'
