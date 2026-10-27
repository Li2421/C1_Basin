"""Only the authorized safety-layer changes; frozen physical stack unchanged."""
from dynamics_safety import *
from dynamics_safety import acceleration_problem as original_acceleration_problem

GAMMAS={'lit_narrow_passage':.3,'lit_crowded_random':.2}

def acceleration_problem(p,v,i,c,wall_segments,gamma=None,circle_discrete=False):
    A,lower,labels=original_acceleration_problem(p,v,i,c,wall_segments)
    cfg=HardProjectionConfig();d=2*c.agent_radius+c.agent_collision_margin+cfg.separation_buffer
    epsilon=.01*d
    p=np.asarray(p,float);v=np.asarray(v,float)
    for k,label in enumerate(labels):
        if label[0]=='pair':
            j=label[2]
            if c.scene=='lit_circle_five' and not circle_discrete:
                # Only enlarge the distance in the original h term.
                lower[k]+=.5*6.*((d+epsilon)**2-d*d)
            else:
                predicted=p if circle_discrete else p+c.dt*v
                delta=predicted[i]-predicted[j];distance=np.linalg.norm(delta)
                if distance<=1e-14:raise ValueError('Undefined predicted pair normal')
                n=delta/distance;g=distance-d-epsilon
                rate=gamma if gamma is not None else GAMMAS[c.scene]
                A[k]=(.5 if circle_discrete else 1.)*c.dt*c.dt*n
                lower[k]=-.5*rate*g-c.dt*(n@v[i])
        elif label[0]=='wall':
            lower[k]+=(6. if c.scene=='lit_circle_five' else 1.)*epsilon
    return A,lower,labels

def solve_soc(target,A,lower,acceleration_bound=None):
    """Lexicographic: original, relax only speed, then safety minimax.

    Speed is the final row, as in the immutable v2 problem builder. All
    certificates refer to the original rows; effective rows are saved too.
    """
    target=np.asarray(target,float);A=np.asarray(A,float).reshape(-1,2);lower=np.asarray(lower,float)
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
    m=len(A);physical=np.arange(m-1);speed=m-1
    def fit(ids,effective,slack_ids=None):
        minimax=slack_ids is not None;dim=3 if minimax else 2
        P=sparse.diags([0.,0.,0.] if minimax else [1.,1.],format='csc')
        q=np.array([0.,0.,1.]) if minimax else -target
        mat=-A[ids]
        if minimax:mat=np.c_[mat,-np.isin(ids,slack_ids).astype(float)]
        rhs=-effective[ids]
        if minimax:mat=np.vstack([mat,[0.,0.,-1.]]);rhs=np.r_[rhs,0.]
        blocks=[sparse.csc_matrix(mat)];values=[rhs];cones=[clarabel.NonnegativeConeT(len(rhs))]
        if acceleration_bound is not None:
            ball=np.zeros((3,dim));ball[1:,:2]=-np.eye(2)
            blocks.append(sparse.csc_matrix(ball));values.append(np.array([acceleration_bound,0.,0.]));cones.append(clarabel.SecondOrderConeT(3))
        return clarabel.DefaultSolver(P,q,sparse.vstack(blocks,format='csc'),np.concatenate(values),cones,settings).solve()
    solved=lambda s:str(s.status) in ('Solved','AlmostSolved')
    ids=np.arange(m);effective=lower.copy();phase='original';relax=0.;speed_relax=0.
    if np.all(A@target>=lower-1e-10) and (acceleration_bound is None or np.linalg.norm(target)<=acceleration_bound+1e-10):
        x=target.copy();status='Feasible_unconstrained_minimizer'
    else:
        sol=fit(ids,effective)
        if str(sol.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
            phase='speed_only';first=fit(ids,effective,[speed])
            if solved(first):
                speed_relax=max(0.,float(first.x[-1]));effective[speed]-=speed_relax+1e-8
                sol=fit(ids,effective)
            elif str(first.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
                phase='safety_minimax';ids=physical
                first=fit(ids,effective,physical)
                if not solved(first):raise RuntimeError('Safety minimax SOCP '+str(first.status))
                relax=max(0.,float(first.x[-1]));effective[physical]-=relax+1e-8
                sol=fit(ids,effective)
            else:raise RuntimeError('Speed slack SOCP '+str(first.status))
        if not solved(sol):raise RuntimeError('Acceleration SOCP '+str(sol.status))
        x=np.asarray(sol.x[:2]);status=str(sol.status)
    violation=np.maximum(lower-A@x,0.)
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite acceleration')
    if acceleration_bound is not None and np.linalg.norm(x)>acceleration_bound+1e-6:raise RuntimeError('Acceleration bound certificate failed')
    if np.max(effective[ids]-A[ids]@x,initial=0.)>2e-6:raise RuntimeError('Effective row certificate failed')
    return x,dict(minimax_relaxation=relax,maximum_violation=float(violation.max(initial=0.)),
        safety_maximum_violation=float(violation[physical].max(initial=0.)),
        speed_violation=float(violation[speed]),speed_relaxation=speed_relax,
        restored=phase!='original',restoration_phase=phase,status=status,
        effective_lower=effective.tolist(),included_rows=ids.tolist())
