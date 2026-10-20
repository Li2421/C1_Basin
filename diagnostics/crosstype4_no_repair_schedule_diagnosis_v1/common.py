"""Round38 output boundary and authoritative DB imports."""
from core import *
EXP='exp_crosstype4_no_repair_schedule_diagnosis_v1'
import shared_rollout_db
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
from shared_rollout_db.src import rollout_db,cache_writer,planner
for m in (rollout_db,cache_writer,planner):assert Path(m.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==(ROOT/'shared_rollout_db/journals').resolve()
os.environ['CUDA_CACHE_PATH']=str(HERE/'cuda_cache');(HERE/'cuda_cache').mkdir(exist_ok=True)
