"""Generated rectangular maps with inherited physical and path semantics."""
from dataclasses import replace
import math
import numpy as np
from bottleneck_family.scenario import Config, Geometry, Instance, rectangle_walls
from bottleneck_family.environment import BottleneckEnv

class GeneratedGeometry(Geometry):
    def __init__(self, config, rectangles):
        self.config = config
        rects = [tuple(map(float, r)) for r in rectangles]
        for x0, y0, x1, y1 in rects:
            if not (-config.half_length <= x0 < x1 <= config.half_length and
                    -config.half_height <= y0 < y1 <= config.half_height):
                raise ValueError('rectangle outside workspace or inverted')
        self.rectangles = np.asarray(rects, dtype=float).reshape(-1, 4)
        outer = (-config.half_length, -config.half_height, config.half_length, config.half_height)
        self.walls = np.concatenate([rectangle_walls(outer)] + [rectangle_walls(r) for r in rects])
        self.margin = config.agent_radius + config.wall_radius + config.wall_collision_margin
        nodes = []
        delta = self.margin + .01
        for x0, y0, x1, y1 in rects:
            nodes.extend([(x0-delta,y0-delta), (x1+delta,y0-delta),
                          (x1+delta,y1+delta), (x0-delta,y1+delta)])
        self.nodes = np.asarray([p for p in nodes if self.valid_points(np.asarray([p]))[0]], dtype=float).reshape(-1,2)
        self.graph = self._graph(self.nodes)

class GeneratedEnv(BottleneckEnv):
    def __init__(self, geometry, positions, goals, config=None):
        self.config = config or geometry.config
        self.num_agents = self.config.num_agents
        self.action_shape = (self.num_agents, 2)
        self.instance = Instance(self.config, geometry, np.asarray(positions,float),
                                 np.asarray(goals,float), ())
        self.walls = geometry.walls
        self.pair_indices = tuple(zip(*np.triu_indices(self.num_agents, 1)))
        self._i, self._j = np.triu_indices(self.num_agents, 1)
        self.reset()

def make_geometry(family, seed, n=4, H=10000):
    rng = np.random.default_rng(seed)
    c = Config(num_agents=n, seed=int(seed), max_steps=int(H))
    L, Y, t = c.half_length, c.half_height, c.barrier_thickness
    rects = []; resources = []
    if family in 'ABC':
        xs = [0.] if family != 'B' else np.linspace(-2.4,2.4,int(rng.integers(2,4))).tolist()
        allopens = []
        for i,x in enumerate(xs):
            if family == 'C':
                k=int(rng.integers(2,4)); centers=np.linspace(-2.4,2.4,k)+rng.uniform(-.15,.15,k)
                op=[(float(y),float(rng.uniform(.45,.9))) for y in centers]
            else:
                op=[(float(rng.uniform(-1.3,1.3)),float(rng.uniform(.45,.9)))]
            allopens.append(tuple(op)); low=-Y
            for y,w in op:
                rects.append((x-t/2,low,x+t/2,y-w/2)); low=y+w/2
                resources.append([x,y,w])
            rects.append((x-t/2,low,x+t/2,Y))
        # Validate the sampled doors with the original validator. Config's
        # compulsory Gap fields are placeholders; generated rectangles are
        # the authoritative geometry for every family, including non-Gap maps.
        replace(c,barrier_x=tuple(xs),openings=tuple(allopens))
    elif family == 'D':
        w=float(rng.uniform(.9,1.6)); h=w/2
        rects=[(-L,-Y,-h,-h),(h,-Y,L,-h),(-L,h,-h,Y),(h,h,L,Y)]
        resources=[[0.,0.,w]]
    elif family == 'E':
        height=float(rng.uniform(1.0,2.2)); width=height*float(rng.uniform(1.,2.))
        rects=[(-width/2,-height/2,width/2,height/2)]
        resources=[[0.,-height/2-(Y-height/2)/2,Y-height/2],
                   [0.,height/2+(Y-height/2)/2,Y-height/2]]
    elif family == 'F':
        # Two crossed walls, each half with a door, yielding four connected rooms.
        w=float(rng.uniform(.5,.9)); y=2.+float(rng.uniform(-.3,.3)); x=4.+float(rng.uniform(-.5,.5))
        rects=[(-t/2,-Y,t/2,-y-w/2),(-t/2,-y+w/2,t/2,y-w/2),
               (-t/2,y+w/2,t/2,Y),(-L,-t/2,-x-w/2,t/2),
               (-x+w/2,-t/2,-t/2,t/2),(t/2,-t/2,x-w/2,t/2),(x+w/2,-t/2,L,t/2)]
        resources=[[0.,-y,w],[0.,y,w],[-x,0.,w],[x,0.,w]]
    elif family == 'G':
        for _ in range(int(rng.integers(3,7))):
            x=float(rng.uniform(-4.5,4.5));y=float(rng.uniform(-2.4,2.4));w=float(rng.uniform(.5,1.7));h=float(rng.uniform(.5,1.3))
            rects.append((x-w/2,y-h/2,x+w/2,y+h/2))
        resources=[]
    else: raise ValueError(family)
    return GeneratedGeometry(c,rects), resources

def task(geometry, family, seed):
    rng=np.random.default_rng(seed); p=[]
    if family in 'ABC':
        for side in [-1,1,-1,1]:
            p.append([side*float(rng.uniform(5.3,7.2)),float(rng.uniform(-2.8,2.8))])
        p=np.asarray(p);g=np.asarray([[-np.sign(q[0])*float(rng.uniform(5.3,7.2)),float(rng.uniform(-2.8,2.8))] for q in p])
    elif family=='D':
        p=np.asarray([[-6.,0.],[6.,0.],[0.,-3.],[0.,3.]])+rng.uniform(-.12,.12,(4,2));g=np.asarray([[5.,0.],[-5.,0.],[0.,2.3],[0.,-2.3]])+rng.uniform(-.12,.12,(4,2))
    elif family=='F':
        p=np.asarray([[-6.,-2.8],[6.,2.8],[-6.,2.8],[6.,-2.8]])+rng.uniform(-.2,.2,(4,2));g=np.asarray([[5.,2.2],[-5.,-2.2],[5.,-2.2],[-5.,2.2]])+rng.uniform(-.2,.2,(4,2))
    elif family=='G':
        p=rng.uniform([-7.2,-3.2],[7.2,3.2],(4,2))
        order=rng.permutation(4);p=p[order]
        g=p[[1,0,3,2]]+rng.uniform(-1.,1.,(4,2))
    else:
        p=np.asarray([[-6.,-2.5],[6.,2.5],[-6.,2.5],[6.,-2.5]])+rng.uniform(-.25,.25,(4,2));g=np.asarray([[5.,2.0],[-5.,-2.0],[5.,-2.0],[-5.,2.0]])+rng.uniform(-.25,.25,(4,2))
    if not geometry.valid_points(np.vstack([p,g])).all():raise ValueError('invalid starts/goals')
    ii,jj=np.triu_indices(4,1)
    if min(np.linalg.norm(q[ii]-q[jj],axis=1).min() for q in [p,g])<=.32:raise ValueError('overlap')
    paths=[geometry.path(a,b) for a,b in zip(p,g)]
    if any(x is None for x in paths):raise ValueError('no single-robot path')
    length=max(np.linalg.norm(np.diff(x,axis=0),axis=1).sum() for x in paths)
    H=int(math.ceil((2.5*length/geometry.config.max_speed/geometry.config.dt)/100)*100)
    return p,g,paths,H
