from core import *
import time

def audit():
    check_prereg();d=R5/'stage_2/probes';p=read(d/'protocol.json');cases=read(d/'CASES.json');index=read(d/'RAW_TRACE_INDEX.json')
    z=np.load(d/'R5_FEATURES.npz');names=z['controller_names'].tolist();old=z['features'];re=np.full((26,16,2,5),np.nan);bad=[];rawerr=0.;counts={n:0 for n in names};files=set()
    for r,idx in zip(cases,index):
        f=d/'raw'/f"{r['rollout_uid']}.json";rec=read(f);tr=Path(rec['trace_path']);files.update((f,tr))
        assert rec['case']==r and sha(f)==idx['raw_sha256'] and sha(tr)==rec['trace_sha256']==idx['trace_sha256']
        assert not rec['terminal_labels_read'] and rec['protocol_sha256']==sha(d/'protocol.json')
        if not rec['complete']:
            bad.append(dict(controller=r['controller_name'],state=r['state_index'],eta=r['eta_index'],steps=rec['physical_steps'],error=rec['error'],statistics=rec['statistics']));continue
        t=np.load(tr);pos=t['positions'];act=t['actions'];goals=t['goals'];active=t['projection_active']
        assert len(act)==200 and len(pos)==201 and len(active)==200
        initial=np.linalg.norm(goals-pos[0],axis=-1).sum();final=np.linalg.norm(goals-pos[-1],axis=-1).sum()
        dist=np.linalg.norm(pos[:,:,None,:]-pos[:,None,:,:],axis=-1)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
        speed=np.linalg.norm(act,axis=-1)
        stats=np.array([(initial-final)/initial,active.mean(),dist.min(),speed.mean(),(speed.mean(-1)<.01).sum()],float)
        rawerr=max(rawerr,float(np.max(np.abs(stats-rec['statistics']))));counts[r['controller_name']]+=1
        re[names.index(r['controller_name']),r['state_index'],[10,13].index(r['eta_index'])]=stats
    reconstructed=re.reshape(26,160);err=float(np.nanmax(np.abs(reconstructed-old)));same=np.array_equal(reconstructed.astype(np.float32),old,equal_nan=True)
    assert names==p['anchor_order']+read(R5/'stage_2/protocol.json')['controller_order']
    assert np.argwhere(np.isnan(old)).tolist()==[[9,j] for j in range(105,110)]
    assert len(bad)==1 and bad[0]['controller']=='base_u_v3_macflow' and bad[0]['state']==10 and bad[0]['eta']==13
    distribution={}
    for j,n in enumerate(['progress','projection_fraction','minimum_pair_distance','mean_speed','stalled_steps']):
        a=re[...,j];valid=a[np.isfinite(a)];distribution[n]=dict(n=len(valid),min=valid.min(),max=valid.max(),mean=valid.mean(),std=valid.std(),quantiles=np.quantile(valid,[0,.25,.5,.75,1]),legitimate_zero_count=int((valid==0).sum()))
    p4=read(R4/'protocol.json');p5=read(R5/'stage_2/protocol.json');uid=lambda s:{r['state_uid'] for r in s}
    disjoint=not(uid(p['states']) & (uid(p4['states'])|uid(p5['states'])))
    train_subset=uid(p5['states'][:96])<=uid(p4['states'][:128]);train_test_disjoint=not(uid(p5['states'][:96])&uid(p4['states'][128:]))
    np.savez_compressed(HERE/'R5_RECONSTRUCTED.npz',features=re,controller_names=names)
    result=dict(utc=utc(),passed=bool(same and rawerr<1e-12 and disjoint and train_subset and train_test_disjoint),raw_complete=sum(counts.values()),valid_per_controller=counts,nan_indices=np.argwhere(np.isnan(old)),failures=bad,max_abs_error_vs_float32_features=err,max_abs_error_vs_raw_statistics=rawerr,float32_exact=same,ordering='controller,state,eta10/13,progress/projection/min_pair_distance/mean_speed/stalled_steps',no_missing_zero_fill=True,distribution=distribution,probe_disjoint=disjoint,blend_train_subset_real_train=train_subset,blend_train_disjoint_real_endpoint=train_test_disjoint)
    write(HERE/'R5_AUDIT.json',result)
    # Prepare TRAIN-only artifacts now; stage1 never opens endpoint containers.
    orig=np.load(R4/'LABELS.npz')['outcomes'][:,:128,:,:4];extra=np.load(R4/'stage_b/LABELS.npz')['outcomes'];blend=np.load(R5/'stage_2/TRAIN_LABELS.npz')['outcomes']
    labels=np.full((26,128,16,4),-2,np.int8);labels[:14]=np.concatenate([orig,extra]);labels[14:,:96]=blend
    np.savez_compressed(HERE/'TRAIN_ONLY.npz',outcomes=labels,controller_names=names,lengths=[128]*14+[96]*12)
    for base in (R4,R5,R6):
        for pattern in ('*.py','*.md','*.json','*.npz','fits/*/*.npz','fits/*/*.json','fits/*/*.msgpack','shared_sparse_fits/*/*','logistic/*/*','static_features/*.npz','stage_b/*.json','stage_b/*.npz','stage_b/static_features/*.npz','stage_2/*.json','stage_2/*.npz','stage_2/static_features/*.npz','stage_2/fits/*/*','stage_2/probes/*.json','stage_2/probes/*.npz'):
            files.update(f for f in base.glob(pattern) if f.is_file())
    files.update([ROOT/'AGENTS.md',ROOT/'docs/prompt_round7_behavior_repr_20261013.md',P1/'P1_LABELS.npz',P1/'P1_EXECUTION_PROTOCOL.json',P1/'state_generator.py']);files.update((P1/'p1_features').glob('*.npz'))
    write(HERE/'HISTORICAL_ASSET_INDEX.json',dict(utc=utc(),files=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(files)],compatibility='same execution labels; no new rollout needed',missing='R5/coverage critic and classifier fits only'))
    print(json.dumps(clean(result)),flush=True);assert result['passed']

def replay():
    check_prereg()
    import jax,jax.numpy as jnp,flax.linen as nn
    from flax import serialization
    from scipy.special import softmax
    sys.path.insert(0,str(R6));from util import Data,frozen_predict
    from fit import r0_inputs
    jax.config.update('jax_enable_x64',False);assert any(d.platform=='gpu' for d in jax.devices())
    class Residual(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.silu(nn.Dense(16,name='dense1')(x));x=nn.silu(nn.Dense(16,name='dense2')(x))
            return jnp.tanh(nn.Dense(16,name='out',kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x))
    net=Residual();fn=jax.jit(lambda p,x:net.apply(p,x));template=net.init(jax.random.PRNGKey(0),np.zeros((1,536),np.float32));rows=[]
    for t in tasks():
        start=time.time();data=Data(t);ns=len(data.test_states);nh=len(data.held);out=HERE/'replay'/old_name(t,'all');out.mkdir(parents=True,exist_ok=True)
        base=r0_folder(t);bx=np.concatenate([r0_inputs(base,data.test_states) for n in data.held]);b=frozen_predict(base,bx).reshape(3,nh,ns,16)
        cached0=np.load(base/'predictions.npz')['held'];cached0=cached0[:,None] if t['bed']=='real' else cached0
        f=R6/'fits'/old_name(t,'F2_PCA2');norm=np.load(f/'normalization.npz');x=np.array([(data.inputs(n,data.test_states,'F2_PCA2')-norm['mean'])/norm['scale'] for n in data.held],np.float32)
        meta=read(f/'complete.json');p=[]
        for i,(path,h) in enumerate(zip(meta['checkpoints'],meta['checkpoint_sha256'])):
            assert sha(path)==h;pa=serialization.from_bytes(template,Path(path).read_bytes());p.append(np.asarray(fn(pa,jnp.asarray(x.reshape(-1,536)))).reshape(nh,ns,16)+b[i])
        p=np.array(p);cached=np.load(f/'predictions.npz')['held']
        cf=R6/'logistic'/old_name(t,'F4');c=np.load(cf/'classifier.npz');h=np.array([data.coordinates(n,'F1_PCA2') for n in data.held]);probs=softmax(np.c_[np.ones(nh),(h-c['mean'])/c['scale']]@c['weights'],axis=1)
        choice=np.where(probs.max(1)>=.6,c['classes'][probs.argmax(1)],data.src);direct=np.tile(choice[None,:,None],(3,1,ns));saved4=np.load(cf/'predictions.npz')['choices']
        hist=read(R6/f"{t['bed']}_results.json")['rows']
        if t['bed']=='real':lab=np.load(R4/'LABELS.npz')['outcomes'][:,128:,:,4:8];names=data.p4['controller_order']
        else:lab=np.load(R5/'stage_2/TEST_ENDPOINT_LABELS.npz')['outcomes'];names=data.p5['controller_order']
        comparisons=[]
        for arm,policy,a,old in [('B0','argmax',b.argmax(-1),cached0.argmax(-1)),('F2_PCA2','pessimistic1',choose(p,'pessimistic1',data.src),choose(cached,'pessimistic1',data.src)),('F2_PCA2','conservative',choose(p,'conservative',data.src),choose(cached,'conservative',data.src)),('F4','argmax',direct,saved4)]:
            for j,n in enumerate(data.held):
                y=(lab[names.index(n)]==1).mean(-1)[None];u=(lab[names.index(n)]==-1).mean(-1)[None];success=picked(y,a[:,j]).mean();unknown=picked(u,a[:,j]).mean()
                prev=next(r for r in hist if r['coverage']==t['coverage'] and r['controller']==n and r['arm']==arm and r['policy']==policy)
                sl,sh=contrast(y,u,a[:,j],np.full_like(a[:,j],data.src));rlo,rhi=contrast(y,u,a[:,j],b.argmax(-1)[:,j])
                vals=dict(success_lower=success,success_upper=success+unknown,gain_src_lower=sl.mean(),gain_src_upper=sh.mean(),gain_R0_lower=rlo.mean(),gain_R0_upper=rhi.mean())
                differences={k:float(v-prev[k]) for k,v in vals.items()};identical=np.array_equal(a[:,j],old[:,j]);rate_same=max(abs(v) for v in differences.values())<1e-10
                comparisons.append(dict(controller=n,arm=arm,policy=policy,choices_identical=identical,selection_mismatches=int((a[:,j]!=old[:,j]).sum()),report_differences=differences,success_identical=rate_same))
        rec=dict(task=t,passed=all(r['choices_identical'] and r['success_identical'] for r in comparisons),comparisons=comparisons,R0_logit_max_abs_error=float(abs(b-cached0).max()),F2_logit_max_abs_error=float(abs(p-cached).max()),elapsed_seconds=time.time()-start,utc=utc(),devices=[str(d) for d in jax.devices()])
        np.savez_compressed(out/'predictions.npz',R0=b,F2=p,F4_choices=direct);write(out/'receipt.json',rec);rows.append(rec);write(HERE/'STAGE0_REPRODUCTION.json',dict(passed=all(r['passed'] for r in rows),tasks=len(rows),expected_tasks=15,rows=rows))
        print(json.dumps(clean(dict(task=t,passed=rec['passed'],seconds=rec['elapsed_seconds']))),flush=True)
        if not rec['passed']:raise RuntimeError('Mandatory reproduction gate failed; see receipt. No new analysis/fits allowed.')
    assert len(rows)==15

def finalize():
    a=read(HERE/'R5_AUDIT.json');r=read(HERE/'STAGE0_REPRODUCTION.json');p=a['passed'] and r['passed'] and r['tasks']==15
    write(HERE/'STAGE0.json',dict(utc=utc(),passed=p,R5_audit_sha256=sha(HERE/'R5_AUDIT.json'),reproduction_sha256=sha(HERE/'STAGE0_REPRODUCTION.json'),new_rollouts=0,shared_db_writes=0));assert p
if __name__=='__main__':globals()[sys.argv[1]]()
