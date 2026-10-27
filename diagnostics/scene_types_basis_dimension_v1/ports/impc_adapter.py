"""Full IMPC-OB author planning stack with explicit Python/geometry/plant adapters."""
import sys,types,inspect,contextlib,io,time
from pathlib import Path
import numpy as np
HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/paper_baselines_20261028')

class IMPC:
    def __init__(self,env,seed,options=None):
        from paper_adapters import vendor_path,SolverMonitor
        root=vendor_path('impc_ob');self.monitor=SolverMonitor();self.priority_counts={};self.max_eta=1.;self.mapping_seconds=0.
        # Binary supplied by authors targets Python3.9. Same convex-hull Euclidean
        # distance in two dimensions through GEOS, with no controller approximation.
        from shapely.geometry import MultiPoint
        geom=types.ModuleType('opengjk.opengjkc');geom.gjk=lambda a,b:MultiPoint(np.asarray(a)[:,:2]).convex_hull.distance(MultiPoint(np.asarray(b)[:,:2]).convex_hull)
        sys.modules['opengjk.opengjkc']=geom
        import SET,geometry
        cfg=env.config;options=options or {};self.options=options;scale=cfg.agent_radius/.3 if options.get('scale_lengths')else 1.
        SET.Num=env.num_agents;SET.K=options.get('K',8);SET.h=cfg.dt;SET.REALFLY=False;SET.next_interval=1;SET.interval=1;SET.compute_model='norm';SET.core_num=1;SET.show=False
        SET.buffer=.03*scale;SET.ExtendWidth=cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin
        SET.resolution=min(.02,cfg.agent_radius/2)
        points=np.concatenate([env.positions,env.goals,np.asarray(env.walls).reshape(-1,2)]);lo=points.min(0)-.5;hi=points.max(0)+.5
        SET.map_range={'x':[float(lo[0]),float(hi[0])],'y':[float(lo[1]),float(hi[1])]}
        SET.obstacle_list=geometry.Build_ExtensionZone([geometry.line(np.asarray(a),np.asarray(b))for a,b in env.walls],SET.ExtendWidth)
        # Exact author occupancy rule (corner rasterization), serial and without file writes.
        obstacles=geometry.Build_ExtensionZone(SET.obstacle_list,.1*scale);w=int((hi[1]-lo[1])/SET.resolution);length=int((hi[0]-lo[0])/SET.resolution)
        # Vectorize the same polygon containment for all grid corners; cells touched
        # by occupied corners are occupied, as in geometry.grid_width.
        from shapely.geometry import Polygon
        from shapely import contains_xy
        xs=lo[0]+np.arange(length+1)*SET.resolution;ys=lo[1]+np.arange(w)*SET.resolution
        X,Y=np.meshgrid(xs,ys,indexing='ij');occupied=np.zeros(X.shape,bool)
        for ob in obstacles:occupied|=contains_xy(Polygon(ob.vertex_list),X,Y)
        corners=occupied[:-1]|occupied[1:];grid=corners.copy();grid[:,:-1]|=corners[:,1:];SET.path_obstacle_list=grid.astype(int)
        # Prefer isolated current OMPL wheels over authors' bundled Python3.9 binaries.
        for k in list(sys.modules):
            if k=='ompl' or k.startswith('ompl.'):sys.modules.pop(k)
        sys.path.insert(0,str(HERE/'deps_ompl17'))
        from ompl import util as ou
        ou.RNG.setSeed(20261028+seed)
        import path_planning
        # OMPL2 nanobind removed State(space); allocation has identical state semantics.
        # OMPL1.7 includes the required author ABITstar planner and original State API.
        import uav,run
        self.run=run;self.agents=[]
        # Only author logging is suppressed, not exceptions or solver diagnostics.
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(env.num_agents):
                a=uav.uav2D(i,env.positions[i].copy(),env.goals[i].copy(),'Obstacle-transitor'if len(env.walls)else'Free-transitor',SET.K)
                if len(env.walls)and a.path is None:raise RuntimeError('Author ABITstar failed to return a path within original bounded retry budget')
                a.Vmax=cfg.max_speed;a.r_a=cfg.agent_radius;a.r_min=2*cfg.agent_radius+cfg.agent_collision_margin;a.r_max=a.Vmax*a.h*a.K
                if options.get('scale_lengths'):
                    a.Umax*=scale;a.epsilon*=scale;a.E*=scale;a.W*=scale
                self.agents.append(a)
    def action(self,env,t):
        for i,a in enumerate(self.agents):
            a.p=env.positions[i].copy();a.state[:2]=a.p
        import SET
        with contextlib.redirect_stdout(io.StringIO()):self.agents=self.run.run_one_step(self.agents,SET.obstacle_list)
        for a in self.agents:
            k=str(a.priority);self.priority_counts[k]=self.priority_counts.get(k,0)+1;self.max_eta=max(self.max_eta,float(a.eta))
        # Execute first planned waypoint through the original SI integrator.
        return (np.asarray([a.p for a in self.agents])-env.positions)/env.config.dt
    def stats(self):return dict(solver=self.monitor.stats(),priority_agent_steps=self.priority_counts,max_right_hand_eta=self.max_eta,options=self.options,initial_paths=[getattr(a,'path',None)for a in self.agents],port='Full author dynamic priority + right-hand penalty + safe corridors + ABITstar. Planned first-step displacement tracked by original SI plant. Author internal double-integrator prediction retained; fixed timestep/radius/speed mapped to benchmark. Python3.9 GJK binary replaced by GEOS convex-hull distance; OMPL1.7 author API; occupancy vectorized. Optional uniform physical-length scaling and original-duration prediction horizon explicit. Original wall-clock planner stopping retained: fixed OMPL seed is not bitwise replay guarantee; paths saved.')
