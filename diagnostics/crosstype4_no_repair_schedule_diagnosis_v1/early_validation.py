"""Randomized 32 actual executions before permitting any inferred local label."""
from common import *
from actions import action,case
from records import actual
from early_reuse import proof
from registry import register
from shared_rollout_db.src.rollout_db import eta_identity,uid

def design():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'];base=read(HERE/'BASE_PROTOCOL.json');train=read(HERE/'TRAINING_STATES.json');A=read(HERE/'A_OBJECTS.json');states={s['state_uid']:s for s in train+A['states']}
    proposals={r['state_uid']:r['proposals'] for r in read(D/'crosstype3_unseen_controller_schedule_v1/CANDIDATES.json')['states'][:48]}
    proposals.update({r['state_uid']:r['proposals'] for r in A['records']})
    rng=np.random.default_rng(202610194401);pool=[];coverage={};rejected={};types=sorted(set(base['types'].values()))
    with rollout_db.connect(True) as db:
        for typ in types:
            candidates=[]
            for name in base['controller_order']:
                if base['types'][name]!=typ:continue
                cid=uid('ctl',base['profiles'][name]['payload'])
                for row in db.execute('SELECT * FROM rollout WHERE controller_uid=? AND episode_length<600 AND conflict_quarantined=0 AND outcome IN (\'success\',\'collision\',\'numerical_failure\')',(cid,)):
                    if row['state_uid'] not in states or row['seed_key']!='{"future_index":0}':continue
                    candidates.append((name,dict(row)))
            rng.shuffle(candidates);candidates.sort(key=lambda x:x[1]['outcome']=='success');kept=[]
            # Randomized source order and alternating target modes; at most six
            # sources/type suffice for the prespecified stratified random draw.
            for name,row in candidates:
                source=actual(row)
                if source is None:continue
                a=source['record']['eta'];s=states[row['state_uid']];late=next(e for e in proposals[s['state_uid']] if not np.array_equal(a,e))
                for mode in ('progress','fixed'):
                    c,cp=case(base,name,s,action(a,late,mode,600,'early_termination_validation'),0)
                    if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():continue
                    p=proof(c,source)
                    if p is None:continue
                    zero=bool(np.all(np.asarray(a)==0))
                    if sum(r['zero_early']==zero and r['case']['switch_mode']==mode for r in kept)<3:kept.append(dict(case=c,profile=cp,state=s,source=source,proof=p,zero_early=zero,type=typ))
                if len(kept)>=6 and any(r['zero_early'] for r in kept) and any(r['case']['switch_mode']=='progress' for r in kept):break
            coverage[typ]=len(kept);pool.extend(kept)
            print(typ,len(candidates),'sources;',len(kept),'eligible',flush=True)
    write(HERE/'EARLY_VALIDATION_POOL.json',pool)
    print('Eligible modes',[(m,sum(r['case']['switch_mode']==m for r in pool)) for m in ('fixed','progress')],'zero',sum(r['zero_early'] for r in pool),flush=True)
    assert len(pool)>=32 and sum(n>0 for n in coverage.values())>=4
    selected=[];used=set()
    def take(options):
        options=[r for r in options if r['case']['rollout_uid'] not in used]
        if not options:return
        r=options[int(rng.integers(len(options)))];selected.append(r);used.add(r['case']['rollout_uid'])
    # Test every eligible type before allowing any reuse for that type.
    for typ in types:take([r for r in pool if r['type']==typ])
    for mode in ('fixed','progress'):take([r for r in pool if r['zero_early'] and r['case']['switch_mode']==mode])
    assert any(r['zero_early'] for r in selected) and any(r['case']['switch_mode']=='progress' for r in selected)
    while len(selected)<32:take(pool)
    rng.shuffle(selected);rows=[r['case'] for r in selected];assert len(used)==32
    profiles={r['profile']['controller_uid']:r['profile'] for r in selected};ss={r['state']['state_uid']:r['state'] for r in selected}
    register('early_validation',dict(base,states=list(ss.values()),phase='equivalence'),rows,profiles)
    assert len(read(HERE/'early_validation_MISSING_CASES.json'))==32
    write(HERE/'EARLY_VALIDATION_CHECKS.json',dict(utc=utc(),seed=202610194401,source_pool_coverage=coverage,selected=selected,all_sources_actual=True))
    seal('early_validation_EXECUTION_FROZEN.receipt.json',[HERE/'EARLY_VALIDATION_CHECKS.json',HERE/'early_validation_protocol.json',HERE/'early_validation_CASES.json',HERE/'early_validation_PREFLIGHT.receipt.json',HERE/'early_validation.py',HERE/'early_reuse.py',HERE/'actions.py'],before_submission=True,no_inferred_labels_yet=True)

def run(worker,workers):
    from new_execution import execute
    rows=read(HERE/'early_validation_MISSING_CASES.json');p=read(HERE/'early_validation_protocol.json');cache={}
    for i in range(worker,len(rows),workers):
        c=rows[i];path=HERE/'early_validation/raw'/f"{c['rollout_uid']}.json";assert not path.exists();out=execute(c,p,cache,'early_validation');write(path,out);cache_writer.append_journal([out],EXP,f'early_validation_{worker}')

def collect():
    checks=read(HERE/'EARLY_VALIDATION_CHECKS.json');results=[]
    for check in checks['selected']:
        c=check['case'];source=check['source'];r=source['record'];path=HERE/'early_validation/raw'/f"{c['rollout_uid']}.json";out=read(path)
        assert sha(source['trace_path'])==source['trace_sha256']
        fields={k:out[k]==r[k] for k in ('outcome','episode_length','physical_steps','success','collision','timeout','numerical_failure')}
        with np.load(source['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
            for k in ('positions','velocities','actions','nominal_actions','projection_active','swept_clearance','goals'):
                fields[k]=a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()
        results.append(dict(controller=c['base_controller_name'],type=c['family'],target_uid=c['rollout_uid'],source_uid=r['rollout_uid'],zero_early=check['zero_early'],switch_mode=c['switch_mode'],checks=fields,passed=all(fields.values())))
    passed=all(r['passed'] for r in results);types=sorted({r['type'] for r in results})
    write(HERE/'EARLY_TERMINATION_EQUIVALENCE.json',dict(utc=utc(),passed=passed,actual_executions=32,validated_types=types if passed else [],tested_types=types,checks=results,any_failure_globally_disables_reuse=True))
    seal('EARLY_TERMINATION_EQUIVALENCE.receipt.json',[HERE/'EARLY_TERMINATION_EQUIVALENCE.json',HERE/'early_validation_COMPLETE.receipt.json',HERE/'early_validation_EXECUTION_FROZEN.receipt.json'],local_reuse_enabled=passed)
if __name__=='__main__':run(int(sys.argv[2]),int(sys.argv[3])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
