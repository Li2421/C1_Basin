from core import *
EXP='exp_axis2_controller_v1'
MENU=['F10','F13','F2','S200','S300','S500']
# Preload authoritative modules before PoC changes import search paths.
from shared_rollout_db.src import rollout_db,cache_writer,planner
for m in [rollout_db,cache_writer,planner]:assert Path(m.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db'),m.__file__
assert cache_writer.ROOT.resolve()==(ROOT/'shared_rollout_db').resolve()
os.environ['CUDA_CACHE_PATH']=str(HERE/'cuda_cache');(HERE/'cuda_cache').mkdir(exist_ok=True)
