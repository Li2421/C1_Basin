"""Fixed historical coverage cohort, new held-type-out proposals and exact cache."""
from common import *
from actions_v3 import action,case
from action_encoding import encode
from registry import register
from local_reuse import infer

def design():
    from requery_client import FrozenMean
    from schedule_generator import predict as generate
    from inference import predict
    import jax
    assert (HERE/'COVERAGE_BOOTSTRAP_IMPLEMENTATION.receipt.json').exists()
    base=read(HERE/'BASE_PROTOCOL.json');old=D/'crosstype4_no_repair_schedule_diagnosis_v1';objects=[dict(r,original_object_index=i) for i,r in enumerate(read(old/'A_OBJECTS.json')['records']) if r['coverage']];assert len(objects)==240
    models=read(HERE/'SELECTED_MODELS.json');assert (HERE/'SELECTED_GATES.receipt.json').exists()
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:features={str(n):v for n,v in zip(z['names'],z['features'])}
    ref=FrozenMean();condition={};states=list({r['state_uid']:r['state'] for r in objects}.values());state_index={s['state_uid']:i for i,s in enumerate(states)};rows={};profiles={};links=[]
    with jax.default_matmul_precision('highest'):
        for obj in objects:
            suid=obj['state_uid'];name=obj['controller'];typ=obj['type'];si=state_index[suid]
            if suid not in condition:condition[suid]=ref.query(suid,obj['state']['physical'])
            cond=condition[suid];assert np.array_equal(cond['eta_b'],obj['proposals'][0])
            data=dict(names=np.asarray([name]),types=np.asarray([typ]),features=np.asarray([features[name]]),h=np.asarray(cond['h'],np.float32)[None],c=np.asarray(cond['c'],np.float32)[None]);model=models['outer'][typ];assert typ not in model['generator']['source_types']
            proposal=generate(model['generator'],data,[suid])[0,0];aa={}
            for i,pair in enumerate(proposal):aa[f'schedule_{i}']=action(pair[0],pair[1],'progress',700)
            for i,early in enumerate(obj['proposals']):aa[f'requery_{i}']=action(early,obj['proposals'][0],'progress',700,late_source='requery')
            aa['frozen_mean']=action(obj['proposals'][0],obj['proposals'][0]);vectors=np.asarray([[encode(a) for a in aa.values()]],np.float32)[None];p=predict(model['full'],dict(data,action=vectors))[:,0,0]
            method_links={}
            for key,a in aa.items():
                c,cp=case(base,name,obj['state'],a,0,si);rid=c['rollout_uid'];rows.setdefault(rid,c);profiles[cp['controller_uid']]=cp;method_links[key]=rid
            links.append(dict(controller=name,type=typ,state_uid=suid,state_index=si,original_object_index=obj['original_object_index'],actions=aa,methods=method_links,critic_probability=p,old_56_reused=True))
    register('coverage',dict(base,states=states,phase='coverage'),list(rows.values()),profiles);write(HERE/'coverage_ACTION_LINKS.json',links);missing=read(HERE/'coverage_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for c in missing:
            a=infer(c,base,db)
            if a:aliases[c['rollout_uid']]=a
    execution=[c for c in missing if c['rollout_uid'] not in aliases];mean_uids={r['methods']['frozen_mean'] for r in links};mean_missing=[c for c in missing if c['rollout_uid'] in mean_uids]
    assert len(rows)<=4320
    write(HERE/'coverage_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'coverage_EXECUTION_CASES.json',execution)
    write(HERE/'coverage_BUDGET.json',dict(utc=utc(),new_proposal_request_cap=4080,frozen_mean_reference_request_cap=240,total_unique_requests=len(rows),exact_cached=len(rows)-len(missing),early_equivalent=len(aliases),actual_execution_cap=len(execution),mean_reference_DB_missing=len(mean_missing),mean_reference_real_execution_cap=sum(c['rollout_uid'] not in aliases for c in mean_missing),historical_56_progress_labels_reused=13440,no_coverage_labels_used_for_selection=True))
    seal('coverage_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('coverage_protocol.json','coverage_CASES.json','coverage_PREFLIGHT.receipt.json','coverage_ACTION_LINKS.json','coverage_LOCAL_EQUIVALENCE.json','coverage_EXECUTION_CASES.json','coverage_BUDGET.json','coverage.py','worker.py','actions_v3.py','execution_v3.py','requery_execution_v3.py','requery_client.py','requery_helper.py','snapshot_conditioning.py','requery_generator_v2.py','SELECTED_MODELS.receipt.json','SELECTED_GATES.receipt.json')],before_submission=True)

def assemble():
    import csv
    assert (HERE/'coverage_COMPLETE.receipt.json').exists();links=read(HERE/'coverage_ACTION_LINKS.json');old=D/'crosstype4_no_repair_schedule_diagnosis_v1';raw=list(csv.DictReader((old/'A_ALL_ACTION_OUTCOMES.csv').open()));lookup={}
    for r in raw:
        if r['seed']=='0':lookup[r['controller'],r['state_uid'],r['method_or_pair']]=r
    oldlinks={r['object_index']:r for r in read(old/'A_ACTION_LINKS.json') if r['seed']==0 and r['coverage']};aliases=read(HERE/'coverage_ALL_LOCAL_EQUIVALENCE.json');lo=[];hi=[];uids=[];probs=[]
    with rollout_db.connect(True) as db:
        for obj in links:
            lower=[];upper=[];ids=[]
            original=oldlinks[obj['original_object_index']]
            pair_keys=list(original['proposal_pairs']);assert len(pair_keys)==56
            for key in [*pair_keys,'human_progress']:
                r=lookup[obj['controller'],obj['state_uid'],key];lower.append(float(r['success_lower']));upper.append(float(r['success_upper']));ids.append(r['rollout_uid'])
            for key,rid in obj['methods'].items():
                r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();r=dict(r) if r is not None else aliases[rid]['outcome'];unknown=bool(r['numerical_failure']);lower.append(0. if unknown else float(r['success']));upper.append(1. if unknown else float(r['success']));ids.append(rid)
            assert len(lower)==75  # 56 original + human + 9 schedule + 8 requery + mean.
            lo.append(lower);hi.append(upper);uids.append(ids);probs.append(obj['critic_probability'])
    np.savez_compressed(HERE/'COVERAGE_BUNDLE.npz',names=[r['controller'] for r in links],types=[r['type'] for r in links],state_uids=[r['state_uid'] for r in links],lower=lo,upper=hi,uids=uids,critic_probability=probs)
    seal('COVERAGE_BUNDLE.receipt.json',[HERE/'COVERAGE_BUNDLE.npz',HERE/'coverage_COMPLETE.receipt.json',HERE/'coverage.py',old/'A_ALL_ACTION_OUTCOMES.csv',old/'A_ACTION_LINKS.json'])
if __name__=='__main__':globals()[sys.argv[1]]()
