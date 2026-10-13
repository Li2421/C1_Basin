"""Locate failure before FF integration: learn the actual deterministic lift."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_local_transfer_localization_v1';EVAL=HERE
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    assert (HERE/'history_audit.json').exists()
    p=copy.deepcopy(read(BASE/'protocol.json'))
    d=np.load(BASE/'dataset.npz');wanted=set(d['state_uid'])
    old=read(ROOT/'controller_four_scene_mechanism_v1/protocol.json')
    states=[s for s in old['states'] if s['state_uid'] in wanted]
    assert len(states)==160
    (HERE/'learn.py').write_text((BASE/'learn.py').read_text())
    (HERE/'review.py').write_text((BASE/'review.py').read_text())
    # Bound all components without label-fitted scales. Bgoal/Brel<=vmax;
    # orthogonal safety residual<=vmax times the frozen3.303687... scale.
    bound=float(np.max(np.abs(d['eta'])@np.array([1.,3.303687238760696,1.])))
    p.update(schema='deterministic_correction_transfer_localization_v1',states=states,arms=ARMS,folds=FOLDS,
        correction_bound_vmax=bound,eta_indices=[0,5,10,15])
    p['experiment']=dict(
        question='Does learned state_eta transfer fail already in the deterministic runtime correction delta(s,eta), before the endpoint FF program/integration? If yes, local integration noise and futurecontrollerresponse cannot be necessary for this specific failure.',
        target='Exact existing basis.compute(physical, safety(nominal_goal_reference)).correction(eta); allgoal-frameactivecomponents, affinebyfixedanalyticbound to[0,1]. Not Q, not a replacement controller, never added as Qinput.',
        scope='All160oldnaturalstates,4existinggoal-referenceMfamilies,K4. Delta is identicalacrossM/initlaw; retain4Crowstofixoldtrainingdraws/exposure. Source-only4folds,state72andfaireta x3seeds=24fits,6000steps,sourceVALstate_etaMSEcheckpoint. Inputfeaturesunchangedbitwise.',
        cache='Reuse32matchingstates basis fromexistingexcitationaudit; only128missingdeterministicbasis evaluations. No Env.step,no FF integration,no labelresimulation.',
        statistics='Same physicalstate20k bootstrap/mismatch/sourceheldseparation asStage35. Targetdeterministic, fouridenticalstoredcopies are formatcompatibility, NOTfourindependentrollouts. Trueenergy is exactwithinthispanel; no half-seedreliability claim. Allsource/heldmetrics andspuriousCdependence retained.',
        gate='If sourceANDheld delta state_eta skill positiveCI acrossscenes, moveoneoperationforwardtoFFcomposition. Ifheldfailswhereexactnonzero targetsignal existsandsourcelearns, localizefailuretolearningphysical/interventionmapping already; nextisdeterministicgeometryfeature-sufficiency/learnedcomposition,notmorecontrollerrollouts. No Qsuccessclaim.',
        boundary='OpenedDEVELOPMENT. Actualdelta uses safety-projectednominal so this doesnotyetseparatepreparatorysafety fromgeometry/basis. Globalaffinetargetscale differsfromactiontarget; no directcross-target rawMSEperformance comparison. Matchedstatebundle includesstate-dependentCfeatures.',
        preparation_sha256=sha(__file__),code_sha256=sha(HERE/'learn.py'),old_dataset_sha256=sha(BASE/'dataset.npz'),
        literature=[dict(url='https://arxiv.org/abs/1806.01261',relevance='Relational/compositionalgeneralization motivates execution-graph localization,not a claimthatGNNautomaticallysolvesourtask.')])
    freeze(HERE/'protocol.json',p)
    print(dict(cached_basis_states=32,new_basis_states=128,new_FF_actions=0,new_rollouts=0,diagnostic_fits=24))


def worker(index):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');assert sha(__file__)==p['experiment']['preparation_sha256']
    scene=tuple(FOLDS.values())[index];states=[s for s in p['states'] if s['scenario']==scene]
    old=ROOT/'controller_flow_endpoint_probe_v1'
    rr=read(old/'basis_excitation_audit.json')['rows'];bb=np.load(old/'basis_fields.npz')['fields']
    cache={}
    for r,b in zip(rr,bb):
        if r['reference']!='goal':continue
        if r['state_uid'] in cache:np.testing.assert_array_equal(cache[r['state_uid']],b)
        cache[r['state_uid']]=b
    base=native_runtime(scene);rt=EndpointMatchedField(base,'isotropic')
    def forbidden(*a,**k):raise AssertionError('No Env.step or FF integrate in correction target audit')
    base.env_cls.step=forbidden;rt.integrate=forbidden
    jax.config.update('jax_enable_x64',False)
    rows=[]
    for s in states:
        dest=HERE/'basis'/(s['state_uid']+'.npz')
        if dest.exists():rows.append(read(dest.with_suffix('.json')));continue
        n=len(s['physical']['positions'])
        if s['state_uid'] in cache:
            b=cache[s['state_uid']][:n];reused=True
        else:
            env=rt.make_env(s);_,_,_,safe,_,_=rt.prepare(env,key_for(s,0,0),np.zeros(3))
            b=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,-1);reused=False
        pos=np.asarray(s['physical']['positions']);e=np.asarray(s['physical']['goals'])-pos
        e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
        frame=np.stack((e,np.stack((-e[:,1],e[:,0]),-1)),-1)
        correction=np.einsum('ndk,ek->end',b,np.asarray(p['eta']))
        local=np.einsum('end,ndf->enf',correction,frame)/rt.config.max_speed
        assert abs(local).max()<=p['correction_bound_vmax']+1e-8
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,basis_world=b,correction_world=correction,correction_goal_vmax=local,
            label=.5+.5*local/p['correction_bound_vmax'],vmax=rt.config.max_speed)
        r=dict(state_uid=s['state_uid'],scene=scene,reused_basis=reused,sha256=sha(dest));write(dest.with_suffix('.json'),r);rows.append(r)
    write(HERE/f'worker{index}.json',dict(rows=rows,physical_steps=0,FF_actions=0))
    print(dict(scene=scene,reused=sum(r['reused_basis'] for r in rows),new=len(rows)-sum(r['reused_basis'] for r in rows)))


def assemble():
    d=np.load(BASE/'dataset.npz');ss=np.zeros((640,16,8));ff=ss.copy();raw=np.full((640,16,4,8),np.nan)
    files=[]
    for i,sid in enumerate(d['state_uid']):
        f=HERE/'basis'/(str(sid)+'.npz');z=np.load(f);r=read(f.with_suffix('.json'));assert sha(f)==r['sha256']
        y=z['label'].reshape(16,-1);no=y.shape[-1]
        for j in (0,5,10,15):
            ss[i,j,:no]=4*y[j];ff[i,j,:no]=4*(1-y[j]);raw[i,j,:,:no]=y[j]
        files.append(r)
    (HERE/'dataset.npz').symlink_to((BASE/'dataset.npz').resolve())
    np.savez_compressed(HERE/'motion_raw.npz',vector_motion=raw,state_uid=d['state_uid'])
    t=HERE/'targets';t.mkdir(exist_ok=True);h={}
    for arm in ARMS:
        np.savez_compressed(t/(arm+'.npz'),success=ss,failure=ff);h[arm]=sha(t/(arm+'.npz'))
    assert len(set(h.values()))==1
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),target_sha256=h,
        deterministic_target=True,format_copies_not_independent_rollouts=4,physical_states=160,
        cached_basis_states=32,new_basis_queries=128,new_rollouts=0,files=files))
    freeze(HERE/'source_gate.json',dict(passed=True,reason='Mechanism diagnostic authorization; actual source learning assessed beforeinterpretingheldfailure. Gatefielddoesnotassertlearnedsourceperformance.'))
    print(dict(ready=True,new_rollouts=0,new_basis_queries=128))


def fit(index):
    from . import learn
    for arm in ARMS:learn.fit(learn.JOBS.index(dict(**RUNS[index],arm=arm)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','assemble','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    globals()[a.stage](a.index) if a.stage in ('worker','fit') else globals()[a.stage]()
