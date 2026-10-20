"""32 actual requery prefix-equivalent executions, stratified over all19 types."""
from common import *
from actions_v2 import action,case
from early_reuse import proof
from registry import register

def design():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'];base=read(HERE/'BASE_PROTOCOL.json');old=D/'crosstype4_no_repair_schedule_diagnosis_v1';sources=read(old/'EARLY_VALIDATION_POOL.json');means={}
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:means.update({str(s):z['proposals'][i,0] for i,s in enumerate(z['state_uids'])})
    means.update({r['state_uid']:np.asarray(r['proposals'][0]) for r in read(old/'A_OBJECTS.json')['records']});pool=[];seen=set()
    with rollout_db.connect(True) as db:
        for r in sources:
            c0=r['case'];a=r['source']['record']['eta']
            if any(np.array_equal(a,base['eta'][j]) for j in (10,13)):continue
            c,cp=case(base,c0['base_controller_name'],r['state'],action(a,means[r['state']['state_uid']],c0['switch_mode'],600,'early_requery_equivalence','requery'),0)
            if c['rollout_uid'] in seen or db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():continue
            assert sha(r['source']['trace_path'])==r['source']['trace_sha256'];pr=proof(c,r['source'])
            if pr is None:continue
            pool.append(dict(case=c,profile=cp,state=r['state'],source=r['source'],proof=pr));seen.add(c['rollout_uid'])
    types=sorted(set(base['types'].values()));available={t:sum(r['case']['family']==t for r in pool) for t in types};write(HERE/'EARLY_VALIDATION_ELIGIBILITY.json',dict(utc=utc(),available=available,eligible=len(pool),artificial_eta13_eta10_excluded=True));assert all(available.values()),available;assert len(pool)>=32
    rng=np.random.default_rng(202610195103);selected=[];used=set()
    for t in types:
        choices=[r for r in pool if r['case']['family']==t];r=choices[int(rng.integers(len(choices)))];selected.append(r);used.add(r['case']['rollout_uid'])
    rest=[r for r in pool if r['case']['rollout_uid'] not in used];rng.shuffle(rest);selected.extend(rest[:32-len(selected)]);rng.shuffle(selected);assert len(selected)==32
    rows=[r['case'] for r in selected];profiles={r['profile']['controller_uid']:r['profile'] for r in selected};states={r['state']['state_uid']:r['state'] for r in selected};register('early_validation_v2',dict(base,states=list(states.values()),phase='equivalence'),rows,profiles);assert len(read(HERE/'early_validation_v2_MISSING_CASES.json'))==32
    write(HERE/'EARLY_VALIDATION_CHECKS.json',dict(utc=utc(),seed=202610195103,selected=selected,all19_types=types));seal('early_validation_v2_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('EARLY_VALIDATION_CHECKS.json','EARLY_VALIDATION_ELIGIBILITY.json','early_validation_v2_protocol.json','early_validation_v2_CASES.json','early_validation_v2_PREFLIGHT.receipt.json','early_validation_v2.py','early_reuse.py','requery_execution_v2.py','requery_generator_v2.py','actions.py','actions_v2.py','execution_v2.py','snapshot_conditioning.py')],before_submission=True)
def run(worker,workers):
    from execution_v2 import execute
    for path,h in read(HERE/'early_validation_v2_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(path)==h,path
    rows=read(HERE/'early_validation_v2_MISSING_CASES.json');p=read(HERE/'early_validation_v2_protocol.json');cache={}
    for i in range(worker,len(rows),workers):
        c=rows[i];path=HERE/'early_validation_v2/raw'/f"{c['rollout_uid']}.json";assert not path.exists();out=execute(c,p,cache,'early_validation_v2');write(path,out);cache_writer.append_journal([out],EXP,f'early_validation_v2_{worker}')
def collect():
    results=[]
    for r in read(HERE/'EARLY_VALIDATION_CHECKS.json')['selected']:
        c=r['case'];ref=r['source']['record'];out=read(HERE/'early_validation_v2/raw'/f"{c['rollout_uid']}.json");fields={k:out[k]==ref[k] for k in ('outcome','episode_length','physical_steps','success','collision','timeout','numerical_failure')}
        with np.load(r['source']['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
            for key in ('positions','velocities','actions','nominal_actions','goals','projection_active','swept_clearance'):fields[key]=a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes()
        fields['no_requery_before_termination']=out['late_eta_requery'] is None;results.append(dict(controller=c['base_controller_name'],type=c['family'],checks=fields,passed=all(fields.values())))
    passed=all(r['passed'] for r in results);types=sorted({r['type'] for r in results});assert len(types)==19;write(HERE/'EARLY_TERMINATION_EQUIVALENCE_v2.json',dict(utc=utc(),passed=passed,checks=results,validated_types=types if passed else [],actual_executions=32,requery_cases=32,any_failure_globally_disables_reuse=True));seal('EARLY_TERMINATION_EQUIVALENCE_v2.receipt.json',[HERE/n for n in ('EARLY_TERMINATION_EQUIVALENCE_v2.json','early_validation_v2_COMPLETE.receipt.json','early_validation_v2_EXECUTION_FROZEN.receipt.json')],reuse_enabled=passed)
if __name__=='__main__':run(int(sys.argv[2]),int(sys.argv[3])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
