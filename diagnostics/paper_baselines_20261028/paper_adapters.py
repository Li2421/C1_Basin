"""Author implementations on the frozen SI benchmark. Ports are labelled as ports.

No old heuristic baseline is imported. Each worker uses only one vendor namespace.
"""
import sys,time,types,random
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
sys.path.insert(0,str(HERE/'vendor/qpswift/python'))

class SolverMonitor:
    def __init__(self):
        self.calls=0;self.failures=[];self.seconds=0.
        import cvxopt.solvers as solvers
        solvers.options['show_progress']=False
        for name in ('qp','coneqp'):
            original=getattr(solvers,name)
            def checked(*args,_fn=original,_name=name,**kw):
                t=time.monotonic();self.calls+=1
                try:
                    result=_fn(*args,**kw)
                    if result.get('status')!='optimal':self.failures.append(dict(solver=_name,status=result.get('status')))
                    return result
                except Exception as exc:
                    self.failures.append(dict(solver=_name,error=str(exc)));raise
                finally:self.seconds+=time.monotonic()-t
            setattr(solvers,name,checked)
    def swift(self):
        import qpSWIFT
        original=qpSWIFT.run
        def checked(*args,**kw):
            if len(args)==7:kw['opts']=args[-1];args=args[:-1]
            kw['opts']=dict(kw.get('opts',{}),OUTPUT=1);self.calls+=1;t=time.monotonic()
            try:
                result=original(*args,**kw)
                if result['basicInfo']['ExitFlag']!=0:self.failures.append(dict(solver='qpSWIFT',info=result['basicInfo']))
                return result
            except Exception as exc:self.failures.append(dict(solver='qpSWIFT',error=str(exc)));raise
            finally:self.seconds+=time.monotonic()-t
        qpSWIFT.run=checked
    def stats(self):return dict(calls=self.calls,failures=self.failures,seconds=self.seconds)

def vendor_path(name):
    root=HERE/'vendor'/name
    if name=='impc_ob':root=root/'src'
    # Vendor projects use absolute, generic imports. Workers are single-use.
    for p in root.iterdir():
        mod=p.stem if p.suffix=='.py'else p.name
        for k in list(sys.modules):
            if k==mod or k.startswith(mod+'.'):sys.modules.pop(k)
        if p.is_dir() and not p.name.startswith('.') and not (p/'__init__.py').exists():
            package=types.ModuleType(mod);package.__path__=[str(p)];sys.modules[mod]=package
    sys.path.insert(0,str(root));return root

class DAC:
    """Full author Circular2D controller + full geometric detector and CLF shaping."""
    def __init__(self,env,seed,options=None):
        vendor_path('dac');self.monitor=SolverMonitor();self.monitor.swift()
        options=options or {}
        from agents.circle2DAgent import Circle2DAgent
        from shapely.geometry import LineString
        self.agents=[];self.type_counts={};self.shaping_steps=0
        # Actual benchmark walls are line segments; physical wall margin is geometric.
        width=env.config.wall_radius+env.config.wall_collision_margin
        caps=[dict(cen_pos=p,radius=width)for w in env.walls for p in w]if width else []
        workspace=types.SimpleNamespace(obs_circ_set=caps,obs_poly_set={str(i):LineString(w).buffer(width,cap_style='flat')if width>0 else LineString(w)for i,w in enumerate(env.walls)})
        for i in range(env.num_agents):
            a=Circle2DAgent();a.cur_pos=env.positions[i].copy();a.tar_pos=env.goals[i].copy()
            a.agent_r=env.config.agent_radius;a.vmax=env.config.max_speed;a.CLF_shaping_flag=True
            if options.get('threshold_profile')=='author_narrow':a.cbf_boundary_hold={'agentCA':.5,'obsCA':.01}
            a.init_workspace(workspace);self.agents.append(a)
    def action(self,env,t):
        old_types=[a.deadlock_type for a in self.agents];out=[]
        for i,a in enumerate(self.agents):
            a.cur_pos=env.positions[i].copy();a.vel=env.velocities[i].copy()
            ids=[j for j in range(env.num_agents)if j!=i]
            a.nei_agent_pos=[env.positions[j].copy()for j in ids];a.nei_agent_vel=[env.velocities[j].copy()for j in ids]
            a.nei_agent_deadlock_type=[old_types[j]for j in ids]
        for a in self.agents:
            a.CLF_CBF_QP_controller();out.append(a.vel.copy())
            typ=str(a.deadlock_type);self.type_counts[typ]=self.type_counts.get(typ,0)+1
            self.shaping_steps+=int(abs(a.ellip_theta)>1e-8 or np.max(np.abs(np.array(a.ellip_s)-1))>1e-8)
        return np.asarray(out)
    def stats(self):return dict(solver=self.monitor.stats(),deadlock_types=self.type_counts,shaping_agent_steps=self.shaping_steps,port='Original SI controller; common speed/radius, exact line-wall geometry with physical margin; author parameters unchanged.')

class MGR:
    """Full author MGR/obstacle-rotation/CBF stack, SI-command interface port."""
    def __init__(self,env,seed,options=None):
        vendor_path('mgr');self.monitor=SolverMonitor();random.seed(seed)
        options=options or {};self.options=options
        from mgr_geometry import install,placement_map_extension
        RotatedRect=install()
        from PyQt5.QtCore import QPointF
        from simulator import Simulator
        from agent import Agent
        # Author sets otherwise iterate by process-dependent object addresses.
        # Integer robot IDs define reproducible tie ordering without future labels.
        Agent.__hash__=lambda agent:int(agent.name)
        from obstacles import RectObstacle,CircObstacle
        self.Q=QPointF;self.circle_steps=0;self.rotated_steps=0;self.peak_circles=0
        points=np.concatenate([env.positions,env.goals,np.asarray(env.walls).reshape(-1,2)])
        # Translation only; map boundary is a circle-placement constraint, not a new physical wall.
        self.shift=1.-points.min(0);extent=float(np.max(points.max(0)+self.shift)+1.)
        sim=Simulator(sim_width=int(np.ceil(extent*75)),sim_height=int(np.ceil(extent*75)),agent_radius=env.config.agent_radius,env_scale=extent,tau=options.get('tau',.5))
        self.sim=sim
        width=env.config.wall_radius+env.config.wall_collision_margin
        for wall in env.walls:
            a,b=np.asarray(wall)+self.shift;delta=b-a
            lo=np.minimum(a,b);hi=np.maximum(a,b)
            # Capsule decomposition: segment rectangle and endpoint circles.
            if abs(delta[0])>1e-10 and abs(delta[1])>1e-10:
                rect=RotatedRect(a,b,width);sim.static_obstacles.append(types.SimpleNamespace(type='rect',rect=rect,center=rect.center(),width=rect.width(),height=rect.height()))
            else:sim.static_obstacles.append(RectObstacle(lo[0]-(width if delta[0]==0 else 0),lo[1]-(width if delta[1]==0 else 0),max(hi[0]-lo[0],2*width),max(hi[1]-lo[1],2*width)))
            if width:
                sim.static_obstacles.extend(CircObstacle(*p,width)for p in (a,b))
        for i,(p,g)in enumerate(zip(env.positions+self.shift,env.goals+self.shift)):
            a=Agent(env.config.agent_radius,sim.scale,QPointF(*p),QPointF(*g),sim.tau,bool(len(env.walls)),str(i));a.max_speed=env.config.max_speed;a.arrival_threshold=env.config.goal_tolerance;sim.addAgent(a)
        placement_map_extension(sim);sim.initPlacementMap()
    def action(self,env,t):
        sim=self.sim;sim.sim_time=t*env.config.dt
        for i,a in enumerate(sim.agents):
            a.position=self.Q(*(env.positions[i]+self.shift));a.velocity=self.Q(*env.velocities[i]);a.at_target=bool(np.linalg.norm(env.goals[i]-env.positions[i])<=env.config.goal_tolerance)
        sim.update();self.peak_circles=max(self.peak_circles,len(sim.avoidance_circles));out=[]
        # Author update ordering retained; commands execute simultaneously in SI plant.
        for a in sim.agents:
            a.update_escape_status();v,omega=a.decide_velocity();a.heading+=omega*env.config.dt
            if getattr(self,'native_actuation',False):out.append([v*np.cos(a.heading),v*np.sin(a.heading)])
            else:out.append([a.velocity.x(),a.velocity.y()])
            self.circle_steps+=int(a.avoidance_circle is not None);self.rotated_steps+=int(a.goal_rotated)
        return np.asarray(out)
    def stats(self):return dict(solver=self.monitor.stats(),circle_agent_steps=self.circle_steps,obstacle_rotation_agent_steps=self.rotated_steps,peak_circles=self.peak_circles,port='Full author decision stack; uses returned safe planar velocity on original SI plant and integrates virtual heading. Not native unicycle reproduction; no unicycle guarantee claimed.')

class MGRNative(MGR):
    native_actuation=True
    def stats(self):
        s=super().stats();s['port']='Full author decision and final unicycle actuation stack; original heading-first Euler endpoint represented as velocity input to common SI plant. Simultaneous physical state updates, actual wall capsules/rotated rectangles, common physical speed/radius and success tolerance; no artificial map-boundary clamping. These are explicit benchmark adaptations.';s['options']=self.options;return s

METHODS={'dac_author_si':DAC,'mgr_author_unicycle_tracking':MGRNative}
from impc_adapter import IMPC
METHODS['chen_impc_ob_si_tracking']=IMPC
from formula_baselines import Wang,WangCP,RiskAdaptive
METHODS.update(wang2017_full_di_tracking=Wang,wang2017_consistent_perturbation_di_tracking=WangCP,zhang2025_risk_si=RiskAdaptive)
