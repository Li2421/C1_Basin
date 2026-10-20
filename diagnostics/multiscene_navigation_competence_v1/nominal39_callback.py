"""Only the nominal source changes; the copied projection/correction chain is intact."""
from common import *
from policies39 import load_policy as load_native39
from navigation_field import NavigationGrid
from batch_fields39 import IndependentFields
from observation import observe as observe35

def observed39(env):
    fields=env.round40_navigation_batch;p=env.positions
    near=fields.waypoint(p,.5)-p;far=fields.waypoint(p,1.5)-p
    # Preserve actual multirobot neighbors; only independent static goal-field
    # arithmetic is batched. IndependentFields.observed is for solo episodes.
    return np.concatenate([observe35(env),near,far],axis=1).astype(np.float32)

def load_policy(checkpoint):
    actor=load_native39(checkpoint)
    def policy(env,key):
        goals=env.goals.tobytes()
        if getattr(env,'round40_field_goals',None)!=goals:
            grid=NavigationGrid(env.instance.geometry)
            env.round40_navigation_fields=[grid.field(goal) for goal in env.goals]
            env.round40_navigation_batch=IndependentFields(env.round40_navigation_fields)
            env.round40_field_goals=goals
        return actor.from_observation(env,observed39(env),key)
    return policy
