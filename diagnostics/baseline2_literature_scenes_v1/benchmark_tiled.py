"""Finite no-route search benchmark; pure queries, no simulation steps."""
from experiment import *
import numpy as np
import time
from types import SimpleNamespace
from certified_planning import Grid as Old
from tiled_planning import Grid as New


def main():
    box=np.asarray([[[-3.,-3.],[3.,-3.]],[[3.,-3.],[3.,3.]],
                    [[3.,3.],[-3.,3.]],[[-3.,3.],[-3.,-3.]]])
    env=SimpleNamespace(walls=box,config=SimpleNamespace(agent_radius=.1,agent_collision_margin=0.),
                        instance=SimpleNamespace(geometry=SimpleNamespace(margin=.1)))
    rows=[]
    for cls in (Old,New):
        grid=cls(env,.05,.1);grid.max_time=160
        grid.reserved=[np.asarray([[1.,0.]])]
        start=time.monotonic();path=grid.search(np.array([-1.,0.]),np.array([1.,0.]));seconds=time.monotonic()-start
        assert path is None
        rows.append(dict(backend=cls.__module__,seconds=seconds,result='None'))
        print(cls.__module__,seconds,flush=True)
    write(HERE/'TILED_PARENT_LARGE_QUERY_TIMING.json',dict(
        utc=utc(),rows=rows,canonical_over_tiled=rows[0]['seconds']/rows[1]['seconds'],
        no_route_by_stationary_reserved_goal=True,same_search_budget=160,
        new_physical_steps=0,not_a_scientific_outcome=True))


if __name__=='__main__':main()
