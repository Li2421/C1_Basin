from core import *
EXP='exp_axis4b_agent_count_v1'
from shared_rollout_db.src import rollout_db,cache_writer,planner
for m in [rollout_db,cache_writer,planner]:assert Path(m.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert cache_writer.ROOT.resolve()==(ROOT/'shared_rollout_db').resolve()
