"""Select only source-verified operational planner continuation paths."""
from common import *
from checkpoint_store import enabled as target_enabled


def validation(c):
    if target_enabled(c):return HERE/'CHECKPOINT_REPLAY_CHECK.json'
    if not c.get('cross_scene_new'):return None
    if c['controller_payload']['nominal']['type_code']not in('prioritized_astar','conflict_based_search'):return None
    p=HERE/'CHECKPOINT_PLANNER_SCOPE_CHECK.json'
    if not p.exists():return None
    proof=read(p);assert proof['passed']
    for name,expected in proof['implementation_sources'].items():assert sha(name)==expected,name
    return p
