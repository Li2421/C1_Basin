import sys, numpy as np
from dataclasses import replace
sys.path.insert(0, '/home/zhihan/research/Basin_C1')
from bottleneck_family.scenario import Config, Geometry, build_instance
from bottleneck_family.__main__ import preset
L = {
 'long_tunnel':      dict(barrier_thickness=3.0, openings=(((0.0, 0.5),),)),
 'clutter':          'preset',
 'wall_side_door':   dict(openings=(((3.0, 0.5),),)),
 'zigzag_three':     dict(barrier_x=(-3.0, 0.0, 3.0), openings=(((1.0, 0.5),), ((-1.0, 0.5),), ((1.0, 0.5),))),
 'low_room':         dict(half_height=2.0, openings=(((0.0, 0.5),),)),
 'chamber_pillars':  dict(barrier_x=(-1.5, 1.5), openings=(((0.0, 0.5),), ((0.0, 0.5),)), extra_rectangles=((-0.4, 0.6, 0.4, 1.4), (-0.4, -1.4, 0.4, -0.6))),
 'narrow_door':      dict(openings=(((0.0, 0.4),),)),
 'chicane':          dict(barrier_x=(-0.6, 0.6), openings=(((-0.5, 0.5),), ((0.5, 0.5),))),
}
for name, kw in L.items():
    try:
        base = preset('clutter', 4) if kw == 'preset' else replace(Config(num_agents=4), **kw)
        base = replace(base, max_steps=max(2000, 800*4), split='train')
        g = Geometry(base)
        ok = 0
        for s in range(16):
            inst = build_instance(replace(base, seed=2026102600 + s)); ok += 1
        print(f'{name:16s} OK  walls={len(g.walls):3d} rects={len(g.rectangles)} min_opening/diam={min(w for os in base.openings for _,w in os)/(2*base.agent_radius):.2f} instances={ok}/16')
    except Exception as e:
        print(f'{name:16s} FAIL {type(e).__name__}: {e}')
