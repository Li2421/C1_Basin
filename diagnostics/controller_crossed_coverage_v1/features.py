"""Reuse physical/nominal inputs; measure only missing static field responses."""
import argparse,time
import numpy as np
from .study import HERE,BASE,read,write,freeze,sha


def run(index,workers):
    import jax
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    p=read(HERE/'protocol.json');d=dict(np.load(BASE/'dataset.npz'))
    for path,h in p['source_code_hashes'].items():assert sha(path)==h
    native={str(s):i for i,s in enumerate(d['state_uid']) if d['controller'][i]=='native_flow' and d['scene'][i]=='ring_exchange'}
    base=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    def forbidden(*a,**kw):raise RuntimeError('Physical simulator step forbidden in static features')
    base.env_cls.step=forbidden
    runtimes={c:FlowEndpointMatchedField(base,c.removeprefix('flow_')) for c in p['controllers']['ring_exchange']}
    rows=[];start=time.monotonic()
    for state in p['states'][index::workers]:
        sid=state['state_uid'];nominal=d['feature'][native[sid],:,:14]
        for name,rt in runtimes.items():
            dest=HERE/'features'/sid/(name+'.npz')
            if dest.exists():continue
            class Interface:
                def __getattr__(self,key):return getattr(rt,key)
            field=field_context(Interface(),state)
            compact=np.asarray(compact_field(np.concatenate((nominal,field),-1)))
            assert compact.shape==(4,64) and np.isfinite(compact).all()
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(temp,nominal=nominal,field=field,compact=compact,physical_steps=0)
            temp.replace(dest);rows.append(dict(state_uid=sid,controller=name,sha256=sha(dest)))
    write(HERE/'feature_jobs'/f'{index:03d}.json',dict(index=index,workers=workers,rows=rows,
        physical_steps=0,seconds=time.monotonic()-start,nominal_reused=True,code_sha256=sha(__file__)))
    print(dict(index=index,created=len(rows),physical_steps=0),flush=True)


def assemble():
    from shared_rollout_db.src.rollout_db import connect
    p=read(HERE/'protocol.json');old=dict(np.load(BASE/'dataset.npz'))
    assert sha(BASE/'dataset.npz')==read(BASE/'dataset_audit.json')['sha256']
    rows=[(s,c) for s in p['states'] for c in sorted(p['controllers']['ring_exchange'])]
    ix={(s['state_uid'],c):i for i,(s,c) in enumerate(rows)}
    n=len(rows);assert n==480
    raw={k:np.zeros((n,16,4),np.int8) for k in ('success','unknown','collision','deadlock','timeout','observed')}
    raw['termination']=np.full((n,16,4),'','<U48');raw['rollout_uid']=np.full((n,16,4),'','<U90')
    with connect(True) as db:
        for c in read(HERE/'cases.json'):
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert a is not None and not a['conflict_quarantined'],c['rollout_uid']
            assert all(a[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
            i,j,k=ix[c['state_uid'],c['controller_name']],c['eta_index'],c['seed']
            assert not raw['observed'][i,j,k]
            for key in ('success','collision','deadlock','timeout'):raw[key][i,j,k]=a[key]
            raw['unknown'][i,j,k]=a['numerical_failure'];raw['observed'][i,j,k]=1
            raw['termination'][i,j,k]=a['outcome'];raw['rollout_uid'][i,j,k]=c['rollout_uid']
    assert (raw['observed'][:,p['eta_indices']]==1).all() and raw['observed'].sum()==7680
    native={str(s):i for i,s in enumerate(old['state_uid']) if old['controller'][i]=='native_flow' and old['scene'][i]=='ring_exchange'}
    extra={k:[] for k in old if k!='eta'}
    for i,(s,c) in enumerate(rows):
        origin=native[s['state_uid']];assert old['split'][origin]=='train'
        f=np.load(HERE/'features'/s['state_uid']/(c+'.npz'));assert int(f['physical_steps'])==0
        np.testing.assert_array_equal(f['nominal'],old['feature'][origin,:,:14])
        ss,uu,nn=[raw[k][i].sum(-1) for k in ('success','unknown','observed')]
        overrides=dict(feature=f['compact'],success=ss,unknown=uu,failure=nn-ss-uu,
            controller=c,source='crossed_ring128',source_row=i)
        for key in extra:extra[key].append(overrides.get(key,old[key][origin]))
    data={k:np.concatenate((v,np.asarray(extra[k],dtype=v.dtype)),0) if k!='eta' else v.copy() for k,v in old.items()}
    for k in old:np.testing.assert_array_equal(data[k] if k=='eta' else data[k][:len(old[k])],old[k])
    assert not set(data['physical_hash'][data['split']=='train'])&set(data['physical_hash'][data['split']=='validation'])
    assert set(old['physical_hash'])==set(data['physical_hash'])
    raw.update(state_uid=np.array([s['state_uid'] for s,c in rows]),controller=np.array([c for s,c in rows]),eta=old['eta'])
    for name,value in [('dataset.npz',data),('raw_labels.npz',raw)]:
        assert not (HERE/name).exists(),'Do not overwrite immutable assembled evidence'
        np.savez_compressed(HERE/name,**value)
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),base_sha256=sha(BASE/'dataset.npz'),
        old_rows=len(old['state_uid']),added_rows=n,total_rows=len(data['state_uid']),
        total_outcomes=int(sum(data[k].sum() for k in ('success','failure','unknown'))),
        added_outcomes=int(raw['observed'].sum()),added_unknown=int(raw['unknown'].sum()),
        added_collision=int(raw['collision'].sum()),added_deadlock=int(raw['deadlock'].sum()),added_timeout=int(raw['timeout'].sum()),
        unique_physical_states_unchanged=True,old_arrays_bit_identical=True,train_val_overlap=0,
        original_raw_evidence=str(BASE/'dataset_audit.json'),new_raw_sha256=sha(HERE/'raw_labels.npz'),
        static_features_only=True,feature_code_sha256=sha(__file__)))
    print(read(HERE/'dataset_audit.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','assemble']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    run(a.index,a.workers) if a.stage=='features' else assemble()
