"""Prespecified prediction and input-use tests on sealed source confirmation."""
import os,csv,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import pearsonr,binomtest
from . import support_confirmation as conf
from . import support_train as train
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite,sha
read=conf.read;write=conf.write;OUT=conf.OUT

def evaluate():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    from ring_exchange.local_frame import local_observation
    from ring_exchange.environment import Config
    frozen=read(train.OUT/'models_frozen.json');d=dict(np.load(OUT/'dataset.npz'));x0=dict(np.load(OUT/'entities.npz'))
    assert d['valid'].all(),'An invalid confirmation input needs a prespecified fallback, not dropping the state'
    states=read(OUT/'states.json');cfg=Config(**read(OUT.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config'])
    native=np.array([local_observation(*[np.array(s['physical'][k]) for k in ('positions','velocities','goals')],cfg).reshape(-1) for s in states])
    pairs=read(OUT/'pairs.json');si=d['state_index'];ii=np.arange(len(pairs));assert len(pairs)==128
    lookup={(int(s),int(e)):i for i,(s,e) in enumerate(zip(si,d['eta_index']))}
    shifted=np.array([lookup[((int(s)+1)%64,int(e))] for s,e in zip(si,d['eta_index'])])
    rows=[];decisions=[];predictions={};conditions=('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller')
    for kind in (*train.KINDS,'native_state_controller'):
      m=train.model(kind);predict=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
      for entry in [e for e in frozen['models'] if e['kind']==kind]:
        dest=Path(entry['path']);assert sha(dest/'best.msgpack')==entry['checkpoint_sha256'];assert sha(dest/'normalization.json')==entry['normalization_sha256']
        params=serialization.msgpack_restore((dest/'best.msgpack').read_bytes());norm=read(dest/'normalization.json');x=dict(x0)
        eta=(d['eta']-norm['eta_center'])/norm['eta_scale']
        c=(d['context']-norm['context_center'])/norm['context_scale'];r=(d['agent_response']-norm['agent_center'])/norm['agent_scale']
        context=np.concatenate([c,r.reshape(2,len(pairs),-1)],-1)
        if kind=='native_state_controller':
            x['native']=((native-norm['native_center'])/norm['native_scale']).astype(np.float32)
            context=np.concatenate([context,np.broadcast_to(np.eye(2)[:,None,:],(2,len(pairs),2))],-1)
        for condition in conditions:
            state_idx=(si+1)%64 if condition in ('state_shuffle','joint_state_context_shuffle') else si
            context_idx=shifted if condition in ('context_state_shuffle','joint_state_context_shuffle') else ii
            z=np.array([np.asarray(predict(params,gather(x,state_idx),jnp.asarray(eta,dtype=jnp.float32),jnp.asarray(context[1-ci if condition=='wrong_controller' else ci,context_idx],dtype=jnp.float32))) for ci in range(2)])
            met,dec=train.metrics(z,d,ii)
            rows.append(dict(size=entry['size'],kind=kind,seed=entry['seed'],condition=condition,split='sealed_source_confirmation',**met))
            decisions.extend(dict(size=entry['size'],kind=kind,seed=entry['seed'],condition=condition,**v) for v in dec)
            predictions[f"{entry['size']}__{kind}__{entry['seed']}__{condition}"]=z
    # Controller identities are known only in this source diagnostic. This
    # empirical prior is a stronger diagnostic baseline, not unseen-controller
    # or cross-scene deployment evidence.
    td=np.load(train.OUT/'dataset.npz')
    for size in train.SIZES:
        mask=(td['split']=='train') & ((td['source']=='old46') if size=='old46' else True)
        z=np.zeros((2,len(pairs)))
        for ci in range(2):
          for e in (10,15):
            m=mask&(td['eta_index']==e);s=float(td['success'][ci,m].sum());f=float(td['failure'][ci,m].sum());p=(s+.5)/(s+f+1)
            z[ci,d['eta_index']==e]=np.log(p/(1-p))
        predictions[f'{size}__controller_eta_constant__0__correct']=z
        met,dec=train.metrics(z,d,ii);rows.append(dict(size=size,kind='controller_eta_constant',seed=0,condition='correct',split='sealed_source_confirmation',**met))
        decisions.extend(dict(size=size,kind='controller_eta_constant',seed=0,condition='correct',**v) for v in dec)
    csvwrite(OUT/'prediction_metrics.csv',rows);csvwrite(OUT/'decisions.csv',decisions)
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    summarize()

def summarize():
    frozen=read(train.OUT/'models_frozen.json');d=np.load(OUT/'dataset.npz');zz=np.load(OUT/'predictions.npz')
    ss=d['success'].reshape(2,64,2);ff=d['failure'].reshape(2,64,2);q=ss/(ss+ff);good=ss>=15;bad=ff>=2
    rng=np.random.default_rng(20261004071);boot=rng.integers(64,size=(10000,64))
    def array(z):return z.reshape(2,64,2)
    def loss(z):return ((ss*np.logaddexp(0,-z)+ff*np.logaddexp(0,z))/(ss+ff)).mean((0,2))
    def choose(z):return z.argmax(-1)
    def chosen(a,ch):return np.take_along_axis(a,ch[...,None],-1)[...,0]
    def compare(a,b):
        ca=choose(a);cb=choose(b);ga=chosen(good,ca);gb=chosen(good,cb);ba=chosen(bad,ca);bb=chosen(bad,cb)
        delta=loss(a)-loss(b);qdelta=(chosen(q,ca)-chosen(q,cb)).mean(0)
        ci=np.quantile(delta[boot].mean(1),[.025,.975]);qi=np.quantile(qdelta[boot].mean(1),[.025,.975])
        rescue=int((ga&bb).sum());brk=int((ba&gb).sum())
        return dict(NLL_delta=float(delta.mean()),NLL_family_CI_low=float(ci[0]),NLL_family_CI_high=float(ci[1]),
            selected_Q_delta=float(qdelta.mean()),selected_Q_family_CI_low=float(qi[0]),selected_Q_family_CI_high=float(qi[1]),
            rescue=rescue,breaks=brk,net_rescue=rescue-brk,paired_binomial_p=float(binomtest(rescue,rescue+brk,.5).pvalue) if rescue+brk else 1.,
            warning='Paired binomial p treats controller-state cases independently; family-bootstrap CI is the clustered primary uncertainty',
            changed_top1=int((ca!=cb).sum()),mean_abs_probability_change=float(abs(expit(a)-expit(b)).mean()))
    input_rows=[];comparisons=[];contrasts=[]
    for entry in frozen['models']:
        size,kind,seed=(entry[k] for k in ('size','kind','seed'));prefix=f'{size}__{kind}__{seed}'
        z=array(zz[prefix+'__correct']);eta=array(zz[f'{size}__eta_only__{seed}__correct'])
        comparisons.append(dict(size=size,kind=kind,seed=seed,reference='eta_only',**compare(z,eta)))
        baseline_size=size.replace('_pair_equal','')
        prior=array(zz[f'{baseline_size}__controller_eta_constant__0__correct'])
        comparisons.append(dict(size=size,kind=kind,seed=seed,reference='controller_eta_constant',**compare(z,prior)))
        if size.startswith('expanded206'):
            old=size.replace('expanded206','old46');ref=array(zz[f'{old}__{kind}__{seed}__correct'])
            comparisons.append(dict(size=size,kind=kind,seed=seed,reference='matched_old46',**compare(z,ref)))
        for condition in ('state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller'):
            altered=array(zz[prefix+'__'+condition]);input_rows.append(dict(size=size,kind=kind,seed=seed,condition=condition,**compare(altered,z)))
        pred=expit(z);t=q[...,0]-q[...,1];p=pred[...,0]-pred[...,1]
        # Remove controller-level eta preference in this descriptive statistic.
        tr=t-t.mean(1,keepdims=True);pr=p-p.mean(1,keepdims=True)
        corr=float(pearsonr(tr.ravel(),pr.ravel()).statistic) if np.std(pr)>1e-9 else None
        contrasts.append(dict(size=size,kind=kind,seed=seed,state_residual_contrast_correlation=corr,
            state_residual_contrast_MAE=float(abs(tr-pr).mean()),true_state_contrast_std=float(tr.std()),predicted_state_contrast_std=float(pr.std())))
    csvwrite(OUT/'paired_comparisons.csv',comparisons);csvwrite(OUT/'input_use.csv',input_rows);csvwrite(OUT/'state_contrasts.csv',contrasts)
    write(OUT/'confirmation_audit.json',dict(families=64,controller_state_cases=128,controller_state_eta_cells=256,
        valid_trials=int((ss+ff).sum()),numerical=int(d['numerical'].sum()),oracle_B15=int(good.any(-1).sum()),
        fixed_eta10_B15=int(good[...,0].sum()),fixed_eta15_B15=int(good[...,1].sum()),
        all_candidates_B15=int(good.all(-1).sum()),no_known_B15=int((~good.any(-1)).sum()),
        models_frozen_sha256=sha(train.OUT/'models_frozen.json'),target_LOSO_labels_used=False,
        scope='Known source controllers, previously unseen native state families, same exact eta10/15. Not unseen-eta or strict cross-scene generalization.',
        bootstrap_unit='source family, preserving both controllers and both eta',generator_modified=False))
    print(json.dumps(read(OUT/'confirmation_audit.json'),indent=2))

if __name__=='__main__':evaluate()
