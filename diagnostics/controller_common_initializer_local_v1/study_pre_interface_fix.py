"""Common-initializer local transfer control, using existing natural states."""
import argparse, copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read, write, freeze, sha, SEEDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'controller_local_transfer_localization_v1'
EVAL = HERE
FOLDS = {'toy':'toy_give_way', 'db':'double_bottleneck', 'four':'four_way_intersection', 'ring':'ring_exchange'}
FAMILIES = ('isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative')
ARMS = ('motion_complete', 'motion_eta_only')
RUNS = [dict(fold=f, seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    assert (HERE/'history_audit.json').exists()
    old = read(BASE/'protocol.json')
    d = np.load(BASE/'dataset.npz')
    wanted = set(d['state_uid'])
    parent = read(ROOT/'controller_four_scene_mechanism_v1/protocol.json')
    states = [s for s in parent['states'] if s['state_uid'] in wanted]
    assert len(states) == 160 and set(s['state_uid'] for s in states) == wanted
    np.testing.assert_array_equal(np.asarray(parent['eta']), d['eta'])
    (HERE/'learn.py').write_text((BASE/'learn.py').read_text())
    (HERE/'review.py').write_text((BASE/'review.py').read_text())
    p = copy.deepcopy(old)
    p.update(schema='common_initializer_local_transfer_control_v1', states=states, arms=ARMS, folds=FOLDS,
        families=FAMILIES, eta_indices=[0,5,10,15], static_key=2026101204,
        initializer=dict(mean='0', scale='vmax', chart='Cartesian world',
            reference='unchanged original goal-reference law', matrix='unchanged four uncoupled operators'))
    p['experiment'] = dict(
        question='Does removing scene-confounded physical Gaussian initialization/chart repair local state x eta transfer while keeping physical geometry, nominal goal reference and FF/safety unchanged?',
        history_audit_sha256=sha(HERE/'history_audit.json'),
        contrast='Original Stage35 scene-native initializer versus common zero-mean/vmax-isotropic world Gaussian. Source-only retraining needed because targets/field descriptors change. Old24fits reused, not rerun.',
        unchanged='All160 old natural states, split32TRAIN8VAL/scene, K4indices0/5/10/15 from originalK16, shared goal reference/M, FF/safety, no physical step, input72/encoder/optimizer/budget/sourceVALrule. Nominal14 and physical features reused. Exact old initializer8 formula.',
        labels='TRAIN common4cached-key identities root2026101204 folds0..3; independent UID evaluation seeds key_for(s,k,0), TRAIN-origin2 and VAL-origin4. SOURCEVAL only original UID keys on source scenes. No old-label invalidation from re-encoding; this specific new program changes actions.',
        budget='16384 new static FF action queries;640 field descriptors recomputed. Zero terminal rollouts. Numerical failures retained; no dropping rows or repeated seeds. Allfour folds x3seeds xstate/faireta =24fits if labels complete.',
        checkpoint='Identical Stage35 SOURCEVAL state_etaMSE, then fullactionMSE, NLL, earliest. Source-only normalization. No held-label parameter/model selection.',
        statistics='Exact Stage35 independent-half power, zero-interaction skill, state-bundle and controller mismatches, physical-state bootstrap20000 with recentering and donor-pool resampling. All40 heldstates plus origin subsets. Same bootstrap indices pair contrasts; compare normalized skill/true-energy as sensitivity because law changes target power.',
        gate='No Q experiment authorized by positive local fullaction gain alone. Evidence for repair requires positive held state_eta skill CI AND paired gain CI; controller-specific repair additionally positive triple skill where true independent-half power supported. Report all scenes/seeds/source learning; weak power inconclusive. If negative prune initializer-shift-alone explanation, no scale/mean sweep.',
        boundary='Opened DEVELOPMENT, not fresh confirmation or Q/eta-selection success. Common law also changes clipping chart in Ring; treatment is whole initializer/chart subsystem, not Gaussian scale alone. Same-scene state permutation moves full S and state-dependent C descriptor, not explicit S channels alone.',
        literature=[dict(url='https://arxiv.org/abs/1702.02453', relevance='UP-OSI varies dynamics in training alongside parameter conditioning; motivates separating source program support from input presence. Different task, no transferred theorem.')],
        code_sha256=sha(HERE/'learn.py'), runtime_sha256=sha(HERE/'runtime.py'), preparation_sha256=sha(__file__),
        old_protocol_sha256=sha(BASE/'protocol.json'), old_dataset_sha256=sha(BASE/'dataset.npz'),
        hardware='Same GPU highestfloat32 precision as Stage35; no CPU/GPU paired replay discrepancy.')
    freeze(HERE/'protocol.json', p)
    print(dict(states=160, action_queries=16384, terminal_rollouts=0, old_fits_reused=24))


def worker(index):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime, controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    from .runtime import CommonInitializerField
    p = read(HERE/'protocol.json')
    assert sha(__file__) == p['experiment']['preparation_sha256']
    assert sha(HERE/'runtime.py') == p['experiment']['runtime_sha256']
    scene = tuple(FOLDS.values())[index//4]; family = FAMILIES[index%4]
    base = native_runtime(scene); rt = CommonInitializerField(base, family)
    mean, scale = base.mean.copy(), base.scale.copy()
    states = sorted([s for s in p['states'] if s['scenario']==scene], key=lambda s:s['state_uid'])
    def forbidden(*a, **k): raise AssertionError('Physical steps prohibited in static diagnostic')
    base.env_cls.step = forbidden
    records=[]
    for si,s in enumerate(states):
        out = HERE/'static'/s['state_uid']/(family+'.npz')
        if out.exists():
            records.append(read(out.with_suffix('.json'))); continue
        jax.config.update('jax_enable_x64', scene=='toy_give_way')
        old = EndpointMatchedField(base, family)
        cache = np.load(ROOT/'controller_four_scene_mechanism_v1/features'/s['state_uid']/(family+'.npz'))
        nominal = cache['nominal']
        env = rt.make_env(s)
        np.testing.assert_array_equal(rt.target(env), old.target(env))
        np.testing.assert_array_equal(rt.matrix(env), old.matrix(env))
        if si==0:
            np.testing.assert_allclose(controller_context(rt,s), nominal, atol=1e-6, rtol=0)
        field = field_context(rt,s)
        compact = np.asarray(compact_field(np.concatenate((nominal,field),-1)))
        n = len(env.positions); e = env.goals-env.positions
        e /= np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
        lat = np.stack((-e[:,1],e[:,0]),-1)
        # Same eight known-initializer quantities, now analytic world-isotropic.
        init = np.column_stack((np.zeros((n,2)),np.ones(n),np.zeros(n),np.ones(n),
                               e[:,0],lat[:,0],np.full(n,1/rt.config.max_speed)))
        actions = np.full((2,16,4,n,2),np.nan); valid = np.zeros((2,16,4),bool)
        errors=[];keys=np.zeros((2,4,2),np.uint32)
        jax.config.update('jax_enable_x64',False)
        nt=2 if s['study_split']=='train' else 4
        for mode,count in ((0,4),(1,nt)):
            for k in range(count):
                key = jax.random.fold_in(jax.random.PRNGKey(p['static_key']),k) if mode==0 else key_for(s,k,0)
                keys[mode,k]=np.asarray(key)
                for j in p['eta_indices']:
                    try:
                        a=rt.action(env,key,np.asarray(p['eta'][j]),'field')[0]
                        assert np.isfinite(a).all()
                        actions[mode,j,k]=a;valid[mode,j,k]=True
                    except Exception as ex:
                        errors.append(dict(mode=mode,eta=j,seed=k,type=type(ex).__name__,message=str(ex)[:400]))
        vector = np.stack(((actions*e).sum(-1),(actions*lat).sum(-1)),-1)/rt.config.max_speed
        assert np.max(np.abs(vector[valid]),initial=0) <= 1+1e-5
        vector=.5+.5*np.clip(vector.reshape(2,16,4,2*n),-1,1)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,compact=compact,initializer=init,nominal=nominal,field=field,
            raw_actions=actions,vector_motion=vector,valid=valid,keys=keys,physical_steps=0)
        r=dict(state_uid=s['state_uid'],scene=scene,family=family,split=s['study_split'],
            planned_actions=4*(4+nt),known_actions=int(valid.sum()),errors=errors,sha256=sha(out))
        write(out.with_suffix('.json'),r);records.append(r)
        np.testing.assert_array_equal(base.mean,mean);np.testing.assert_array_equal(base.scale,scale)
        if (si+1)%10==0:print(dict(scene=scene,family=family,states=si+1,errors=sum(len(r['errors']) for r in records)),flush=True)
    write(HERE/f'static_worker{index}.json',dict(rows=records,physical_steps=0,new_terminal_rollouts=0))


def assemble():
    d=dict(np.load(BASE/'dataset.npz'))
    common=np.full((640,16,4,8),np.nan);uid=np.full_like(common,np.nan)
    field=np.zeros_like(d['feature']);files=[];errors=[]
    for i,(sid,c) in enumerate(zip(d['state_uid'],d['controller'])):
        f=HERE/'static'/str(sid)/(str(c).removeprefix('goal_')+'.npz');z=np.load(f);r=read(f.with_suffix('.json'))
        assert sha(f)==r['sha256'];errors.extend(r['errors']);files.append(dict(path=str(f),sha256=sha(f)))
        n=int(d['entity_agent_mask'][i].sum());assert n==len(z['compact'])
        field[i,:n]=np.concatenate((z['compact'],z['initializer']),-1)
        common[i,:,:,:2*n]=z['vector_motion'][0];uid[i,:,:,:2*n]=z['vector_motion'][1]
    freeze(HERE/'static_audit.json',dict(files=files,planned_action_queries=16384,failures=errors,
        old_physical_dataset_sha256=sha(BASE/'dataset.npz'),physical_steps=0,new_terminal_rollouts=0))
    assert not errors, 'Numerical failures retained; no silent missing-action fit'
    d['feature']=field
    active=np.repeat(d['entity_agent_mask'].astype(bool),2,-1)
    tr=d['split']=='train';ss=np.zeros((640,16,8));ff=ss.copy()
    for i in range(640):
        raw=common[i] if tr[i] else uid[i]
        nt=4
        for j in (0,5,10,15):
            a=raw[j,:nt,active[i]]
            assert np.isfinite(a).all()
            ss[i,j,active[i]]=a.sum(-1);ff[i,j,active[i]]=(1-a).sum(-1)
    # Preserve all physical/identity fields bitwise; targets change only because
    # this is an explicit controller-program intervention.
    np.savez_compressed(HERE/'dataset.npz',**d)
    np.savez_compressed(HERE/'motion_raw.npz',vector_motion=uid,state_uid=d['state_uid'])
    np.savez_compressed(HERE/'independent_static_actions.npz',vector_motion=common,state_uid=d['state_uid'])
    out=HERE/'targets';out.mkdir(exist_ok=True);hashes={}
    for arm in ARMS:
        np.savez_compressed(out/(arm+'.npz'),success=ss,failure=ff);hashes[arm]=sha(out/(arm+'.npz'))
    assert len(set(hashes.values()))==1
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),target_sha256=hashes,
        raw_sha256=sha(HERE/'motion_raw.npz'),static_sha256=sha(HERE/'independent_static_actions.npz'),
        rows=640,physical_states=160,new_rollouts=0,static_action_queries=16384))
    freeze(HERE/'source_gate.json',dict(passed=True,reason='Fixed diagnostic design after native-source learning already verified Stage35. All labels numerically complete. Re-evaluate source learning in every fold before interpreting held failure; no inference of learning from this bookkeeping flag.'))
    print(dict(assembled=True,errors=len(errors),states=160,new_rollouts=0))


def fit(index):
    from . import learn
    for arm in ARMS:
        learn.fit(learn.JOBS.index(dict(**RUNS[index],arm=arm)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','assemble','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    globals()[a.stage](a.index) if a.stage in ('worker','fit') else globals()[a.stage]()
