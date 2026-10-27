"""Optional paper-dynamics stage; controls never change nominal actor code."""
from experiment import *
import numpy as np
import jax
import time
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from reservation_certificate_v2_flat_static import ReservationNominal as Nominal
from controllers import bound
from distributed_safety import local_basis
from central_reference import progress
from dynamics_safety import *
from single_integrator.environment import point_segment_distance,segment_distance
jax.config.update('jax_enable_x64',False)

def roots_in(coefficients,dt):
    coefficients=np.trim_zeros(np.asarray(coefficients,float),'f')
    if len(coefficients)<=1:return []
    return [float(x.real) for x in np.roots(coefficients) if abs(x.imag)<1e-9 and 0<x.real<dt]

def quadratic_pair_clearance(p,v,a,dt,radius):
    polynomial=[.5*(a@a),1.5*(v@a),v@v+p@a,p@v]
    times=[0.,dt,*roots_in(polynomial,dt)]
    return min(np.linalg.norm(p+t*v+.5*t*t*a)-radius for t in times)

def quadratic_wall_clearance(p,v,a,dt,segment,margin):
    left,right=segment;direction=right-left;length=np.linalg.norm(direction);normal=np.asarray([-direction[1],direction[0]])/length
    times=[0.,dt]
    for endpoint in (left,right):
        d=p-endpoint;times+=roots_in([.5*(a@a),1.5*(v@a),v@v+d@a,d@v],dt)
    signed=np.asarray([.5*(normal@a),normal@v,normal@(p-left)])
    times+=roots_in(signed,dt)+roots_in([2*signed[0],signed[1]],dt)
    along=np.asarray([.5*(direction@a),direction@v,direction@(p-left)])/(length*length)
    times+=roots_in(along,dt)+roots_in(along-np.asarray([0.,0.,1.]),dt)
    points=np.asarray([p+t*v+.5*t*t*a for t in times])
    return float(point_segment_distance(points,segment[0],segment[1]).min()-margin)

def curved_wall_minimum(p,v,a,dt,segments,margin):
    if not len(segments):return np.inf
    end=p+dt*v+.5*dt*dt*a
    distances=point_segment_distance(p,segments[:,0],segments[:,1])
    endpoint=point_segment_distance(end,segments[:,0],segments[:,1])
    upper=float(min(distances.min(),endpoint.min()))
    travel=dt*np.linalg.norm(v)+.5*dt*dt*np.linalg.norm(a)
    # Distance to a closed segment is 1-Lipschitz. Every omitted segment's
    # lower bound exceeds an already attained candidate, so the exact minimum
    # is attained among the retained finite segments.
    ids=np.flatnonzero(distances-travel<=upper+1e-12)
    return min(quadratic_wall_clearance(p,v,a,dt,segments[k],margin) for k in ids)

class DynamicalEnv(LiteratureEnv):
    def __init__(self,state,c):super().__init__(state,c);self.headings=np.zeros(self.num_agents)
    def advance(self,control):
        c=self.config;before=self.positions.copy();oldv=self.velocities.copy()
        if c.scene=='lit_doorway':
            pn=[];hn=[]
            for i in range(self.num_agents):
                pi,hi=unicycle_next(before[i],self.headings[i],control[i],c.dt);pn.append(pi);hn.append(hi)
            after=np.asarray(pn);self.headings=np.asarray(hn)
            newv=control[:,0,None]*np.column_stack([np.cos(self.headings),np.sin(self.headings)])
        elif c.scene=='lit_circle_five':after=before+c.dt*oldv+.5*c.dt*c.dt*control;newv=oldv+c.dt*control
        else:after=before+c.dt*oldv;newv=oldv+c.dt*control
        if c.scene=='lit_circle_five':
            pair=min((quadratic_pair_clearance(before[i]-before[j],oldv[i]-oldv[j],control[i]-control[j],c.dt,2*c.agent_radius+c.agent_collision_margin) for i,j in self.pair_indices),default=np.inf)
            wall=min((curved_wall_minimum(before[i],oldv[i],control[i],c.dt,self.walls,self.instance.geometry.margin) for i in range(self.num_agents)),default=np.inf)
        else:
            pair=float(point_segment_distance(np.zeros(2),before[self._i]-before[self._j],after[self._i]-after[self._j]).min()-2*c.agent_radius-c.agent_collision_margin) if len(self._i) else np.inf
            wall=float(segment_distance(before[:,None],after[:,None],self.walls[None,:,0],self.walls[None,:,1]).min()-self.instance.geometry.margin) if len(self.walls) else np.inf
        collision=bool(pair<=0 or wall<=0 or not self.instance.geometry.valid_points(after).all())
        self.positions=after;self.velocities=newv;self.step_count+=1;self.history.append(after.copy());self.collided|=collision
        success=bool(np.all(np.linalg.norm(self.goals-after,axis=-1)<=c.goal_tolerance))
        self.termination='collision' if collision else 'success' if success else 'timeout' if self.step_count>=c.max_steps else 'running'
        self.done=self.termination!='running'
        return dict(termination=self.termination,collision=collision,swept_clearance=float(min(pair,wall)),
            sampled_speed_violation=float(np.maximum(np.linalg.norm(newv,axis=-1)-c.max_speed,0).max()))

def run(c):
    started=time.monotonic();cfg=LiteratureConfig(**c['scenario_payload']['config']);env=DynamicalEnv(c['state'],cfg)
    actor=Nominal(c['controller_payload']['nominal']);d0=np.linalg.norm(env.goals-env.positions,axis=-1)
    rng_uid=c['rng_state_uid'];root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];headings=[env.headings.copy()]
    actions=[];desireds=[];nominal=[];safe=[];active=[];rewrites=[];clear=[];events=[];switch=None;error=None;termination='timeout'
    bound_accel=None if cfg.scene=='lit_circle_five' else 1.5
    for t in range(cfg.max_steps):
        try:
            desired=bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64)
            desired[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
            if c['theta'] is not None and switch is None and (progress(env.positions,env.goals,d0,cfg.goal_tolerance)>=c['theta'] or t>=round(6*cfg.max_steps/7)):switch=t
            eta=np.asarray(c['eta_b'] if switch is not None else c['eta_a'])
            problems=[];first=[];first_nom=[];first_next_velocity=[];labels=[];step_events=[]
            for i in range(env.num_agents):
                if cfg.scene=='lit_doorway':
                    control,diag=project_unicycle(desired[i],env.positions,env.headings,i,cfg,env.walls)
                    _,heading_next=unicycle_next(env.positions[i],env.headings[i],control,cfg.dt)
                    first_next_velocity.append(control[0]*np.asarray([np.cos(heading_next),np.sin(heading_next)]))
                    first_nom.append(diag['nominal_front_velocity']);first.append(diag['safe_front_velocity']);labels.append(dict(A=diag['active_A'],kinds=diag['active_labels']))
                    problems.append(None);step_events.append(dict(first=diag))
                else:
                    A,lower,kinds=acceleration_problem(env.positions,env.velocities,i,cfg,env.walls)
                    target=track_acceleration(desired[i],env.velocities[i],cfg.dt,bound_accel)
                    control,diag=solve_soc(target,A,lower,bound_accel);first.append(control);first_nom.append(target)
                    first_next_velocity.append(env.velocities[i]+cfg.dt*control)
                    relaxation=diag['minimax_relaxation']+(1e-8 if diag['restored'] else 0.)
                    ids=np.flatnonzero(A@control-lower+relaxation<=1e-7)
                    rows=A[ids].tolist();ks=[kinds[k] for k in ids]
                    if bound_accel is not None and abs(np.linalg.norm(control)-bound_accel)<=1e-7:
                        rows.append((-control/max(np.linalg.norm(control),1e-30)).tolist());ks.append(['acceleration_bound',i])
                    labels.append(dict(A=rows,kinds=ks));problems.append((A,lower,kinds));step_events.append(dict(first=diag))
            first=np.asarray(first);first_nom=np.asarray(first_nom);first_next_velocity=np.asarray(first_next_velocity)
            correction=np.zeros_like(desired)
            for i in range(env.num_agents):correction[i]=np.asarray(eta)@local_basis(env.positions,env.goals,first_next_velocity,i,np.inf,cfg.max_speed)
            final=[]
            for i in range(env.num_agents):
                target_velocity=desired[i]+correction[i]
                if cfg.scene=='lit_doorway':control,diag=project_unicycle(target_velocity,env.positions,env.headings,i,cfg,env.walls)
                else:
                    A,lower,kinds=problems[i];target=track_acceleration(target_velocity,env.velocities[i],cfg.dt,bound_accel)
                    control,diag=solve_soc(target,A,lower,bound_accel)
                final.append(control);step_events[i]['second']=diag
            final=np.asarray(final);info=env.advance(final)
            positions.append(env.positions.copy());velocities.append(env.velocities.copy());headings.append(env.headings.copy())
            actions.append(final);desireds.append(desired);nominal.append(first_nom);safe.append(first);active.append(labels)
            rewrites.append(np.linalg.norm(first-first_nom,axis=-1)>HardProjectionConfig().intervention_tol)
            clear.append(info['swept_clearance']);events.append(dict(solvers=step_events,sampled_speed_violation=info['sampled_speed_violation']))
            termination=info['termination']
            if env.done:break
        except Exception as exc:
            import traceback
            error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());termination='unknown';break
    violation_steps=sum(any(d[which]['maximum_violation']>1e-7 for d in e['solvers'] for which in ('first','second')) for e in events)
    speed_steps=sum(e['sampled_speed_violation']>1e-7 for e in events)
    result=dict(success=int(termination=='success' and not env.collided),collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(error is not None),outcome=termination,episode_length=len(actions),error=error,elapsed_seconds=time.monotonic()-started,switch_step=switch,
        constraint_violation_steps=violation_steps,sampled_speed_violation_steps=speed_steps,minimum_swept_clearance=min(clear) if clear else None,
        final_goal_distances=np.linalg.norm(env.goals-env.positions,axis=-1),paper_velocity_convergence=int(np.all(np.linalg.norm(env.velocities,axis=-1)<.1)),
        stage='paper_dynamics',global_sampled_safety_guarantee=False)
    result["reservation_no_goal_certificates"]=getattr(getattr(actor,"planner",None),"certificates",[])
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),headings=np.asarray(headings),goals=env.goals,
        actions=np.asarray(actions).reshape(-1,env.num_agents,2),desired=np.asarray(desireds).reshape(-1,env.num_agents,2),nominal=np.asarray(nominal).reshape(-1,env.num_agents,2),safe=np.asarray(safe).reshape(-1,env.num_agents,2),
        projection_rewrite=np.asarray(rewrites).reshape(-1,env.num_agents),local_active_json=np.asarray(json.dumps(active)),events_json=np.asarray(json.dumps(clean(events))),swept_clearance=np.asarray(clear))
    return result,arrays
