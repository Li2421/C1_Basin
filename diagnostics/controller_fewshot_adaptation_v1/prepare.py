from core import *

def main():
    prereg();p4=read(R4/'protocol.json');pb=read(R4/'stage_b/protocol.json');pv=read(P1/'P1_EXECUTION_PROTOCOL.json');names=p4['controller_order']+pb['controller_order'];a=np.load(R4/'LABELS.npz');b=np.load(R4/'stage_b/LABELS.npz');v=np.load(P1/'P1_LABELS.npz')
    assert list(a['controller_names'])==names[:7] and list(b['controller_names'])==names[7:]
    assert list(a['state_uids'])==[s['state_uid'] for s in p4['states']];assert list(v['state_uids'])==[s['state_uid'] for s in pv['states']]
    vn=v['controller_names'].tolist();assert set(vn)==set(names)
    Y=v['Y'];U=v['U'];assert np.isin(Y,[0,1]).all() and np.isin(U,[0,1]).all() and (Y+U<=1).all()
    p1=np.where(U==1,-1,Y).astype(np.int8);p1=np.array([p1[vn.index(n)] for n in names]);train=np.concatenate([a['outcomes'][:,:128,:,:4],b['outcomes']]);assert np.isin(train,[-2,-1,0,1]).all()
    requested=(train!=-2).any(-1);assert (requested.sum(2)==8).all() and (requested.sum(1)==64).all()
    assert [s['state_uid'] for s in pb['states']]==[s['state_uid'] for s in p4['states'][:128]]
    def identity(states):return {hashlib.sha256(json.dumps(s['physical'],sort_keys=True,separators=(',',':')).encode()).hexdigest() for s in states}
    groups=[p4['states'][:128],p4['states'][128:],pv['states']];physical=[identity(s) for s in groups];uids=[{s['state_uid'] for s in g} for g in groups]
    assert all(not (physical[i]&physical[j] or uids[i]&uids[j]) for i in range(3) for j in range(i+1,3))
    assert p4['eta']==pv['eta']==pb['eta'];assert p4['dt']==pv['dt']==pb['dt'] and p4['horizon']==pv['horizon']==pb['horizon']==700 and p4['placement']==pv['placement']==pb['placement']=='FF'
    compat=[]
    for i,n in enumerate(names):
        c=(p4 if i<7 else pb)['controllers']['ring_exchange'][n];c1=pv['controllers']['ring_exchange'][n]
        # Preserve exact identities even if experimental UIDs differ.
        x=c['payload'];y=c1['payload'];fields=['flow_checkpoint_sha256','horizon','dt','success_semantics','conditioning']
        equal={k:x.get(k)==y.get(k) for k in fields};assert all(equal.values()),(n,equal)
        compat.append(dict(controller=n,train_controller_uid=c['controller_uid'],P1_controller_uid=c1['controller_uid'],matched_fields=equal,train_profile=x,P1_profile=y))
    np.savez_compressed(HERE/'SOURCE_TRAIN_ONLY.npz',outcomes=train,controller_names=names,state_uids=[s['state_uid'] for s in groups[0]])
    rows=[]
    for i,n in enumerate(names):
        pool=np.concatenate([train[i],p1[i]]) if i<7 else train[i];where=np.argwhere(pool!=-2).astype(np.int16);label=pool[tuple(where.T)];state_uids=[s['state_uid'] for s in groups[0]]+([s['state_uid'] for s in groups[2]] if i<7 else [])
        dest=HERE/'pools'/f'{i}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,cells=where,outcomes=label,controller_name=n,state_uids=state_uids)
        eta_rows=[dict(eta=k,requests=int((where[:,1]==k).sum()),valid=int(((where[:,1]==k)&(label>=0)).sum()),unknown=int(((where[:,1]==k)&(label==-1)).sum()),states=int(len(np.unique(where[where[:,1]==k,0])))) for k in range(16)]
        rows.append(dict(target=i,controller=n,bed='main' if i<7 else 'stageB_descriptive',available=int(len(label)),unknown=int((label==-1).sum()),per_eta=eta_rows,TRAIN_count=4096,P1_count=1536 if i<7 else 0))
    write(HERE/'POOL_AUDIT.json',dict(utc=utc(),passed=True,rows=rows,controller_order=names,UID_and_physical_disjoint=True,TRAIN_states=128,endpoint_states=64,P1_states=24,compatibility=compat,no_fake_labels=True,new_rollouts=0,shared_db_writes=0))
    write(HERE/'PROTOCOL.json',dict(controller_order=names,main_sources=[[n for n in names[:7] if n!=h] for h in names[:7]],stageB_sources=[[n for n in names if n!=h] for h in names[7:]],budgets=BUDGETS,strategies=STRATEGIES,repeats=200,methods=METHODS,main_endpoint_seeds=[4,5,6,7],hindsight_seeds=[0,1,2,3],source_prior_eta=10,source_train_only=True,preregistration_sha256=sha(HERE/'PREREGISTRATION.md')))
    files=set([ROOT/'AGENTS.md',ROOT/'docs/prompt_round8_fewshot_adaptation_20261013.md',ROOT/'docs/prompt_fewshot_controller_adaptation_20261013.md',P1/'P1_LABELS.npz',P1/'P1_EXECUTION_PROTOCOL.json',P1/'state_generator.py',ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz'])
    for base in (R4,R6,R7):
        for pattern in ('*.md','*.py','*.json','*.npz','fits/*/*','static_features/*.npz','stage_b/*.json','stage_b/*.npz','stage_b/static_features/*.npz','controller_prior/*.npz'):
            files.update(f for f in base.glob(pattern) if f.is_file())
    files.update([R5/'REPORT.md',R5/'STAGE_0_RESULTS.json',R5/'stage_2/TRAIN_LABELS.npz',R5/'stage_2/protocol.json'])
    write(HERE/'HISTORICAL_ASSET_INDEX.json',dict(utc=utc(),files=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(files)],compatibility='same cached execution labels and frozen REAL6 critic',evidence_gap='pool predictions and preregistered repeated acquisition/adaptation only'))
    print(json.dumps(dict(passed=True,targets=len(rows),main_pool_size=5632,main_unknown=[r['unknown'] for r in rows[:7]])))
if __name__=='__main__':main()
