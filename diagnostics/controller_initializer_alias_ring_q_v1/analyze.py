"""Independent-seed, state-bootstrap test of a controller descriptor alias."""
import itertools
import numpy as np
from .study import HERE, SCENE, FAMILIES, read, write, sha, freeze
from shared_rollout_db.src.rollout_db import connect
from diagnostics.controller_flow_endpoint_probe_v1.analyze import energy


def run():
    p=read(HERE/'protocol.json');sids=sorted(s['state_uid'] for s in p['states'])
    si={s:i for i,s in enumerate(sids)};ci={s:i for i,s in enumerate(FAMILIES)};ei={j:i for i,j in enumerate(p['eta_indices'])}
    y=np.zeros((2,32,4,4,2));u=np.zeros_like(y);seen=np.zeros_like(y,bool)
    flags={k:np.zeros_like(y,bool) for k in ('collision','deadlock','timeout','numerical_failure')}
    ruids=np.empty(y.shape,'<U80');failures=[]
    with connect(True) as db:
        for arm,name in enumerate(('baseline_cases.json','cases.json')):
            for c in read(HERE/name):
                r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                assert r is not None and not r['conflict_quarantined']
                assert all(r[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                ix=(arm,si[c['state_uid']],ci[c['controller_name']],ei[c['eta_index']],c['seed'])
                y[ix]=r['success'];u[ix]=r['numerical_failure'];seen[ix]=True;ruids[ix]=c['rollout_uid']
                for k in flags:flags[k][ix]=r[k]
                if r['numerical_failure'] or r['collision']:
                    failure=dict(arm=arm,case=c,original_source_file=r['original_source_file'],outcome=r['outcome'])
                    if arm:failure['error']=read(HERE/'raw'/(c['rollout_uid']+'.json'))['error']
                    failures.append(failure)
    assert seen.all()
    np.savez_compressed(HERE/'raw_labels.npz',success=y,unknown=u,rollout_uid=ruids,state_uid=np.asarray(sids),**flags)
    # New - old, preserving every numerical unknown as an interval.
    lo=y[1]-(y[0]+u[0]);hi=y[1]+u[1]-y[0]
    bootstrap=np.random.default_rng(2026101039).integers(32,size=(20000,32))
    boot={'state_indices':bootstrap};rows=[]
    targets=(('raw_Q_difference',()),('eta_preference_difference',(2,)),
             ('state_eta_difference',(0,2)),('state_controller_eta_difference',(0,1,2)))
    for label,axes in targets:
        for group,indices in [('all',list(range(4)))]+[(c,[i]) for i,c in enumerate(FAMILIES)]:
            if len(indices)==1 and label=='state_controller_eta_difference':continue
            a,b=lo[:,indices],hi[:,indices];point=energy(a,b,axes);lower=[];upper=[]
            for start in range(0,len(bootstrap),500):
                jj=bootstrap[start:start+500]
                _,l,h=energy(a[jj],b[jj],tuple(x+1 for x in axes));lower.append(l);upper.append(h)
            lower=np.concatenate(lower);upper=np.concatenate(upper);tag=label+'__'+group
            boot[tag+'__lower']=lower;boot[tag+'__upper']=upper
            rows.append(dict(target=label,controller=group,physical_states=32,
                point_bounds=[float(point[1]),float(point[2])],
                CI95=[float(np.quantile(lower,.025)),float(np.quantile(upper,.975))]))
    # This paired mean change is descriptive. An alias can matter even when
    # opposite effects cancel its average; independent-half energy is primary.
    mean_lo=lo.mean((1,2,3));mean_hi=hi.mean((1,2,3))
    mean_diff=dict(bounds=[float(mean_lo.mean()),float(mean_hi.mean())],
        CI95=[float(np.quantile(mean_lo[bootstrap].mean(-1),.025)),float(np.quantile(mean_hi[bootstrap].mean(-1),.975))])
    # All eta pairs: discovery seed defines opposite strict preferences;
    # second independent seed tests BOTH directions, never changes grouping.
    reversal=[]
    for a,b in itertools.combinations(range(4),2):
        known=~u[:,:,:,[a,b],0].astype(bool).any((0,3))
        disc=(y[:,:,:,a,0]-y[:,:,:,b,0]);group=known&(disc[0]*disc[1]<0)
        evaluation=y[:,:,:,a,1]-y[:,:,:,b,1]
        eval_known=~u[:,:,:,[a,b],1].astype(bool).any((0,3))
        follows=(evaluation[0]*disc[0]>0)&(evaluation[1]*disc[1]>0)
        lower=group&eval_known&follows;upper=lower|(group&~eval_known)
        reversal.append(dict(eta_indices=[p['eta_indices'][a],p['eta_indices'][b]],
            discovery_cases=int(group.sum()),physical_states=int(group.any(1).sum()),
            independent_seed_confirmed=int(lower.sum()),independent_seed_upper=int(upper.sum())))
    raw=next(r for r in rows if r['target']=='raw_Q_difference' and r['controller']=='all')
    interaction=next(r for r in rows if r['target']=='state_eta_difference' and r['controller']=='all')
    result=dict(energy=rows,paired_mean_change=mean_diff,reversal=reversal,
        outcomes=[dict(arm=name,successes=int(y[i].sum()),rollouts=y[i].size,
            **{k:int(v[i].sum()) for k,v in flags.items()}) for i,name in enumerate(('original','altered_initializer'))],
        Q_alias_supported=raw['CI95'][0]>0,state_eta_alias_supported=interaction['CI95'][0]>0,
        equal_input_mixture_MSE_floor_bounds=[v/4 for v in raw['point_bounds']],
        equal_input_mixture_MSE_floor_CI95=[v/4 for v in raw['CI95']],
        boundary='Broader-controller-class information counterexample on32DEVELOPMENT TRAINstates. Not a causal explanation of existing native-distribution LOSO or a validated repair. Q2 independent halves, all unknowns bounded, physical-state bootstrap; no trained model or fresh confirmation.',
        root_cause_closed=False,new_rollouts=1024,old_rollouts_reused=1024,new_fits=0,
        statistics='20000pairedphysicalstatebootstraps; fourfamilies andallK4 retained. Primaryaggregate predeclared. Per-family CIs descriptive, not multiplicity-adjusted proof.',
        code_sha256=sha(__file__))
    np.savez_compressed(HERE/'bootstrap.npz',**boot)
    write(HERE/'failure_cases.json',failures);write(HERE/'results.json',result)
    (HERE/'REPORT.txt').write_text('CONTROLLER INITIALIZER ALIAS — TERMINAL Q DIAGNOSTIC\n'+
        '\n'.join(str(r) for r in rows if r['controller']=='all')+'\n'+str(result['outcomes'])+'\n'+
        str(mean_diff)+'\nQ_alias_supported='+str(result['Q_alias_supported'])+
        ';state_eta_alias_supported='+str(result['state_eta_alias_supported'])+'\n'+result['boundary']+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_INITIALIZER_ALIAS_Q_DIAGNOSTIC',root_cause_closed=False,
        new_rollouts=1024,new_fits=0,report_sha256=sha(HERE/'REPORT.txt')))
    print({k:result[k] for k in ('Q_alias_supported','state_eta_alias_supported','outcomes')})


if __name__=='__main__':run()
