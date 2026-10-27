"""200 full paired episodes, 3000 random keys, real chains, 10000 CSC checks."""
from dispatch import *
from acceleration import prepare,accelerated_case,functions
from equivalence import KEYS,OUTCOMES
from discovery import specs_s1
import importlib.util

def units(cohort):
    spec=importlib.util.spec_from_file_location('threefry_cert',HERE/'accel_inputs/np_threefry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(2026102803)
    for i in range(3000):
        k=rng.integers(0,2**32,2,dtype=np.uint32);d=int(rng.integers(0,2**32,dtype=np.uint64))
        assert np.array_equal(module.fold_in(k,d),np.asarray(RN.jax.random.fold_in(k,d)))
    chain=0
    for c in cohort:
        root=RN.jax.random.fold_in(RN.jax.random.fold_in(RN.jax.random.PRNGKey(2026102004),int(hashlib.sha256(c['rng_state_uid'].encode()).hexdigest()[:8],16)),0)
        for t in range(15):
            a=module.fold_in(root,t);b=RN.jax.random.fold_in(root,t);assert np.array_equal(a,np.asarray(b));chain+=1
            for i in range(c['state']['N']):assert np.array_equal(module.fold_in(a,i),np.asarray(RN.jax.random.fold_in(b,i)));chain+=1
    import shared_control.hard_projection as HP
    import clarabel
    from scipy import sparse
    ns=dict(np=np,HP=HP,clarabel=clarabel,sparse=sparse,_P={});functions(HERE/'accel_inputs/fastproj.py',{'fast_project_velocity'},ns)
    errors=0
    for j in range(10000):
        n=1+j%3;matrix=rng.normal(size=(j%9+1,2*n));matrix[rng.random(matrix.shape)<.2]=0.
        lower=-rng.uniform(.1,1.,len(matrix));target=rng.normal(size=(n,2));target[0]=[1.,0.]
        try:ns['fast_project_velocity'](target,matrix,lower,.5,HP.HardProjectionConfig(),check=True)
        except HP.CBFSolverError:errors+=1
    return dict(random_keys=3000,actual_chain_comparisons=chain,csc_array_checks=10000,synthetic_solver_unknown=errors,passed=True)

def main():
    p=HERE/'ACCELERATION_CERTIFICATE.json'
    if p.exists():return read(p)
    prepare();frozen=read(HERE/'SCENES_FROZEN.json');refs=[]
    eligible=[e for e in frozen['entries']if 'scene'in e]
    # First include 2 states x both controllers x 2 settings in every valid type.
    for states in ([0,1],[2,3]):
        for e in eligible:
            for ctl in controllers():
                for i in states:
                    for sp in [dict(eta=[0.,0.,0.],coefficients={}),specs_s1()[0]]:
                        refs.append(case(e['scene'],ctl,i,sp,purpose='acceleration_certificate'))
    refs=refs[:200];assert len(refs)>=200
    unit=units(refs[:4]);dump(HERE/'ACCELERATION_UNITS.json',unit)
    rr={r['rollout_uid']:r for r in execute(refs,'accel_reference')}
    fast=[]
    for c in refs:
        cc=json.loads(json.dumps(c));cc['controller_payload']['exact_acceleration']=dict(inputs=read(HERE/'ACCELERATION_INPUTS.json'),check=False)
        cc['controller_uid']=db.uid('ctl',dict(scenario=cc['scenario_uid'],payload=cc['controller_payload']))
        cc['rollout_uid']=db.uid('roll',dict(state=cc['state_uid'],eta=cc['eta_uid'],controller=cc['controller_uid'],seed=cc['seed_key']));fast.append(cc)
    ff={r['rollout_uid']:r for r in execute(fast,'accel_paired',worker_entry='worker_accel.py')};checks=[]
    for c,d in zip(refs,fast):
        a,b=c['rollout_uid'],d['rollout_uid']
        if a not in rr or b not in ff:checks.append(dict(a=a,b=b,equal=False,reason='incomplete'));continue
        with np.load(rr[a]['trace'])as x,np.load(ff[b]['trace'])as y:
            arrays={k:x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes()for k in KEYS}
        outcomes={k:rr[a][k]==ff[b][k]for k in OUTCOMES}
        checks.append(dict(a=a,b=b,type_id=c['type_id'],controller=c['controller_payload']['nominal']['type_code'],equal=all(arrays.values())and all(outcomes.values()),arrays=arrays,outcomes=outcomes))
    passed=all(x['equal']for x in checks)and unit['passed']
    result=dict(passed=passed,enabled=['same_CSC_construction','numpy_threefry_fold','numpy_far_wall_cost_pruning','local_far_wall_prescreen']if passed else[],units=unit,pairs=len(checks),checks=checks,inputs=read(HERE/'ACCELERATION_INPUTS.json'),original_seconds=sum(r['wall_seconds']for r in rr.values()),accelerated_seconds=sum(r['wall_seconds']for r in ff.values()),geometry_gap='Requested forest cannot physically be instantiated; its required spacing is inconsistent. All other valid new wall types included.',policy='On any discrepancy all four optimizations are disabled and the original main path continues. Route adapter and dictionary formulas never accelerated.')
    dump(p,result);print('ACCELERATION',passed,'pairs',len(checks),flush=True);return result

if __name__=='__main__':main()
