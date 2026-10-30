"""Local fair-input/error adapters. Author source files are never edited."""
from core import *
import ast,copy,inspect,textwrap,types,traceback
import sys,os

class MethodFailure(RuntimeError):pass

class ErrorContext:
    def __init__(self,n,unified=True):
        self.n=n;self.unified=unified;self.step=0;self.agent=None;self.events=[];self.errors={};self.failed=set();self.first=None
    def begin(self,t):self.step=t;self.agent=None;self.failed=set()
    def fail(self,agent,exc):
        if isinstance(exc,PermissionError):raise exc
        agents=list(range(self.n))if agent is None else[int(agent)]
        key=type(exc).__name__+': '+str(exc)
        if key not in self.errors:self.errors[key]=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        fresh=[i for i in agents if i not in self.failed]
        if fresh:self.events.append(dict(step=self.step,agents=fresh,error_key=key))
        self.failed.update(agents)
        if self.first is None:self.first=dict(step=self.step,agents=agents,error_key=key)
        if not self.unified:raise exc

def protect_loop(cls,context,loop_style):
    """Keep successful per-agent statements literal; catch failures per robot."""
    source=textwrap.dedent(inspect.getsource(cls.action));tree=ast.parse(source);fn=tree.body[0];found=0
    for node in fn.body:
        if not isinstance(node,ast.For):continue
        target=ast.unparse(node.target);iterator=ast.unparse(node.iter)
        match=(loop_style=='formula'and target.startswith('(i,')and'enumerate(env.positions)'in iterator)or(loop_style=='agents'and target=='a'and iterator in('self.agents','sim.agents'))
        if not match:continue
        if loop_style=='agents':
            node.target=ast.parse('for _v2_index,a in enumerate([]):\n pass').body[0].target
            node.iter=ast.Call(func=ast.Name(id='enumerate',ctx=ast.Load()),args=[node.iter],keywords=[])
            index='_v2_index'
        else:index='i'
        old=node.body
        assign=ast.parse(f'self.v2_context.agent=int({index})').body[0]
        fallback=ast.parse(f'self.v2_context.fail({index},_v2_exception)\nout.append(np.zeros(2))').body
        handler=ast.ExceptHandler(type=ast.Name(id='Exception',ctx=ast.Load()),name='_v2_exception',body=fallback)
        node.body=[assign,ast.Try(body=old,handlers=[handler],orelse=[],finalbody=[])];found+=1
    assert found==1,(cls.__name__,loop_style,found)
    tree=ast.fix_missing_locations(tree);namespace=dict(cls.action.__globals__)
    exec(compile(tree,__file__,'exec'),namespace)
    return namespace['action']

def install_status_checks(context):
    """Leave solver returns to the unchanged author call sites.

    SolverMonitor still records nonoptimal returns. Native exceptions still
    propagate to the per-agent error wrapper. No solver option is changed.
    """
    return None

def formula_roots(module,method):
    original=module.solve_qp
    if method.startswith('wang'):
        def without_port_speed(P,q,A,b,speed=None,velocity_offset=None,dt=1.):
            # Paper acceleration/CBF inequalities stay untouched. The extra
            # common-plant speed SOC is removed; episode applies common vmax.
            return original(P,q,A,b,None,None,dt)
        module.solve_qp=without_port_speed
    elif method=='zhang2025_risk_si':
        source=inspect.getsource(original)
        old='settings.max_iter=200;settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9'
        assert source.count(old)==1
        source=source.replace(old,'settings.max_iter=200')
        ns=dict(module.__dict__);exec(compile(source,__file__,'exec'),ns);module.solve_qp=ns['solve_qp']

def risk_plain_action(self,env,t):
    """Paper Eq6: ordinary CLF–CBF, no risk/rotation/auxiliary deadlock CBF."""
    import formula_baselines as module
    output=[];prm=self.params;cfg=env.config
    for i,p in enumerate(env.positions):
        self.v2_context.agent=i;start=__import__('time').monotonic()
        try:
            error=p-env.goals[i];A=[np.r_[2*error,-1.]];b=[-prm['gamma']*(error@error)]
            for j,q in enumerate(env.positions):
                if i==j:continue
                d=p-q;h=d@d-(2*cfg.agent_radius+cfg.agent_collision_margin)**2
                A.append(np.r_[-2*d,0.]);b.append(.5*prm['alpha']*h)
            for wall in env.walls:
                q,_=module.closest_segment(p,wall);d=p-q;h=d@d-(cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin)**2
                A.append(np.r_[-2*d,0.]);b.append(prm['alpha']*h)
            self.calls+=1
            solution,_=module.solve_qp(np.diag([2.,2.,2*prm['p']]),np.zeros(3),A,b,cfg.max_speed)
            output.append(solution[:2]);self.zetas.append(0.)
        except Exception as exc:
            self.failures.append(dict(step=t,agent=i,error=str(exc)));self.v2_context.fail(i,exc);output.append(np.zeros(2))
        self.seconds+=__import__('time').monotonic()-start
    return np.asarray(output)

def deterministic_impc_constructor(cls):
    source=textwrap.dedent(inspect.getsource(cls.__init__))
    anchor='    import uav,run\n'
    assert source.count(anchor)==1
    replacement='''    from ompl import base as _ob,geometric as _og
    _original_solve=_og.ABITstar.solve
    def _fixed_iterations(planner,budget):
        count=max(1,int(round(float(budget)/.04*400)))
        condition=_ob.IterationTerminationCondition(count)
        callback=_ob.PlannerTerminationConditionFn(condition.eval)
        return _original_solve(planner,_ob.PlannerTerminationCondition(callback))
    _og.ABITstar.solve=_fixed_iterations
    import uav,run
'''
    source=source.replace(anchor,replacement)
    namespace=dict(cls.__init__.__globals__);exec(compile(source,__file__,'exec'),namespace)
    cls.__init__=namespace['__init__']

def make(env,seed,method,options,context,mechanism=True,root_only=False):
    sys.path.insert(0,str(HERE/'ports'))
    import paper_adapters as base
    import formula_baselines as formulas
    options=dict(options or{})
    if method.startswith('wang')or method=='zhang2025_risk_si':
        formula_roots(formulas,method)
        cls=base.METHODS[method];obj=cls(env,seed,options);obj.v2_context=context
        if not mechanism:
            if method.startswith('wang'):obj.classify=lambda *args:0
            else:
                obj.v2_mechanism_disabled=True
                obj.action=types.MethodType(risk_plain_action,obj)
                return obj
        if not root_only:obj.action=types.MethodType(protect_loop(cls,context,'formula'),obj)
        return obj
    if method=='dac_author_si':
        legacy=dict(threshold_profile=options['threshold_profile'])if options.get('threshold_profile')else{}
        obj=base.DAC(env,seed,legacy);obj.v2_context=context
        for a in obj.agents:
            a.alpha=options.get('alpha',a.alpha)
            a.cbf_boundary_hold=dict(agentCA=options.get('agent_threshold',a.cbf_boundary_hold['agentCA']),obsCA=options.get('obstacle_threshold',a.cbf_boundary_hold['obsCA']))
            a.CLF_shaping_flag=bool(mechanism)
            if not mechanism:
                a.deadlock_solver.detect_deadlock=lambda param:dict(deadlock_type=1,active_cbf_value=[],clf_cone_dist=np.inf,cone_theta=np.inf)
        if not root_only:
            action=protect_loop(base.DAC,context,'agents')
            def fair_action(self,env,t):
                for i,a in enumerate(self.agents):a.tar_pos=env.goals[i].copy()
                return action(self,env,t)
            obj.action=types.MethodType(fair_action,obj);install_status_checks(context)
        return obj
    if method=='mgr_author_unicycle_tracking':
        obj=base.MGRNative(env,seed,dict(tau=options.get('tau',.5)));obj.v2_context=context
        import agent as agent_module
        agent_module.K_P=options.get('kp',.05)
        if not mechanism:obj.sim.createAvoidanceCircle=lambda *args,**kwargs:None
        if not root_only:
            action=protect_loop(base.MGR,context,'agents')
            def fair_action(self,env,t):
                for i,a in enumerate(self.sim.agents):
                    target=self.Q(*(env.goals[i]+self.shift));a.target=target;a.original_target=self.Q(target)
                    if not a.goal_rotated and a.avoidance_circle is None:a.temp_target=self.Q(target)
                return action(self,env,t)
            obj.action=types.MethodType(fair_action,obj);install_status_checks(context)
        return obj
    if method=='chen_impc_ob_si_tracking':
        # The v1 module already contains the occupancy-domain revision2 fix.
        if not root_only:deterministic_impc_constructor(base.IMPC)
        obj=base.IMPC(env,seed,{k:v for k,v in options.items()if k in('K','scale_lengths')});obj.v2_context=context
        import SET,run,uav
        SET.plot_range=dict(SET.map_range,size=(8,8));SET.format='.png'
        gain=options.get('epsilon_gain',1.)
        for a in obj.agents:a.epsilon*=gain;a.E*=gain;a.W*=gain
        if not mechanism:
            uav.uav2D.deadlock_resolution=lambda self,share_data:None
            for a in obj.agents:a.priority=2;a.eta=0.;a.term_overlap=False;a.term_overlap_again=False
        if not root_only:
            original=run.run_one_agent
            def guarded(item):
                before=copy.deepcopy(item[0]);index=int(before.index);context.agent=index
                try:return original(item)
                except Exception as exc:context.fail(index,exc);return before
            run.run_one_agent=guarded;install_status_checks(context)
            if getattr(env,'_paper_guide_variant',False):
                original_action=obj.action
                def guided_action(self,env,t):
                    for i,a in enumerate(self.agents):
                        context.agent=i
                        if not np.array_equal(a.target,env.goals[i]):a.get_new_target(env.goals[i].copy())
                    return original_action(env,t)
                obj.action=types.MethodType(guided_action,obj)
        return obj
    raise ValueError(method)
