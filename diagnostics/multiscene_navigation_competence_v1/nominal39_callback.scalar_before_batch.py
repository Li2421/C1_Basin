"""Only the nominal source changes; the copied projection/correction chain is intact."""
from common import *
from policies39 import load_policy as load_native39
from navigation_field import NavigationGrid

def load_policy(checkpoint):
    actor=load_native39(checkpoint)
    def policy(env,key):
        goals=env.goals.tobytes()
        if getattr(env,'round40_field_goals',None)!=goals:
            grid=NavigationGrid(env.instance.geometry)
            env.round40_navigation_fields=[grid.field(goal) for goal in env.goals]
            env.round40_field_goals=goals
        return actor(env,env.round40_navigation_fields,key)
    return policy
