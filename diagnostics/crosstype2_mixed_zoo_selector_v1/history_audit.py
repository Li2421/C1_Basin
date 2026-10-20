"""Stable history identity helper, independent of any old write constructors."""
from common import *
from shared_rollout_db.src.rollout_db import canonical


def signature(state):
    return canonical(dict(uid=state['uid'],physical={k:state['physical'][k] for k in ('positions','velocities','goals')}))
