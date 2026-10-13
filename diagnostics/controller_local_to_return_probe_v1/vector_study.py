"""Full-vector target revision, preserving the unexecuted scalar protocol."""
import argparse, copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read, write, freeze, sha, SEEDS
PARENT=Path(__file__).resolve().parent
HERE=PARENT/'vector_control'
FOLDS=read(PARENT/'protocol.json')['folds']
ARMS=('return_current','return_complete','return_eta_only','motion_current','motion_complete','motion_eta_only')
RUNS=[dict(fold=f,representation=a,seed=s) for f in FOLDS for s in SEEDS for a in ('current','complete','eta_only')]


def prepare():
    HERE.mkdir(exist_ok=True)
    d=dict(np.load(PARENT/'dataset.npz'));raw=dict(np.load(PARENT/'raw_labels.npz'))
    actions=dict(np.load(PARENT/'goal_frame_actions.npz'))
    known=raw['observed'].astype(bool)&~raw['unknown'].astype(bool)
    y=np.full((len(d['scene']),16,4,8),np.nan)
    for sc in sorted(set(d['scene'])):
        for split in ('train','validation'):
            tag=str(sc)+'__'+split
            ix=np.flatnonzero((d['scene']==sc)&(d['split']==split))
            ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            np.testing.assert_array_equal(d['state_uid'][ix][::4],actions[tag+'__states'])
            a=actions[tag+'__goal_frame_action'];n,nc,ne,nt,na,nd=a.shape
            a=a.reshape(n*nc,ne,nt,na*nd);assert np.max(np.abs(a))<=1+1e-5
            for k,j in enumerate((0,5,10,15)):y[ix,j,:nt,:na*nd]=.5+.5*np.clip(a[:,k],-1,1)
    active=np.repeat(d['entity_agent_mask'].astype(bool),2,-1)
    train_mask=known[...,None]&active[:,None,None,:];assert np.isfinite(y[train_mask]).all()
    d['motion_success']=np.where(train_mask,y,0).sum(2)
    d['motion_failure']=np.where(train_mask,1-y,0).sum(2)
    np.testing.assert_allclose(d['motion_success']+d['motion_failure'],
        (d['success']+d['failure'])[...,None]*active[:,None,:],atol=1e-12,rtol=0)
    np.savez_compressed(HERE/'dataset.npz',**d)
    np.savez_compressed(HERE/'motion_raw.npz',vector_motion=y,**raw)
    p=copy.deepcopy(read(PARENT/'protocol.json'));p['schema']='local_execution_vs_return_full_vector_v2'
    p['experiment']['revision']=dict(
        reason='Before any scalar fit, half-signal weak. Full-vector audit finds repeatable state x eta signal in all4scenes. Retain every component; no best-coordinate selection.',
        original_protocol_sha256=sha(PARENT/'protocol.json'),original_models_fit=0,
        full_vector_audit_sha256=sha(PARENT/'vector_signal_audit.json'),
        target='All first-action goal-forward/lateral components / vmax, affine to[0,1]. Every active APIagent retained; inactive outputs masked.',
        output='Same encoder/eta/head trunk; local output8 versus Q output1. Shared-trunk initialization/input/sampling identical. Output size and checkpoint criterion necessarily differ; localization diagnostic, not an isolated single-factor performance claim.',
        local_checkpoint='SOURCEVAL state_etaMSE, then complete_actionMSE, thenNLL; noheldcriterion.',
        gate='Joint first. Continue localLOSO onlyif complete-inputlocal improvesSOURCEVAL state_etaMSE overeta-only in>=2/3fits AND independent-half mean-scene interactionskill has positive physical-statebootstrapCI. Reportevery scene andthreewaypower; state_eta alonecannotprovecontrollerinteractiontransfer.',
        boundary='OpenedDEVELOPMENT only. Action is TARGET, never Qinput. Localtransfer/returnfailure localizes failure, notcomplete rootcause. No newrollout.')
    p['experiment']['motion_target']=p['experiment']['revision']['target']
    p['experiment']['gate']=p['experiment']['revision']['gate']
    p['experiment']['evaluation']='Sourcegate before justified heldfolds. Fullvector, physicalstatebootstrap, independenthalves. OriginalQaudits unchanged. Noaction-as-Qcontext.'
    code=(PARENT/'learn.py').read_text()
    code=code.replace('from .study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha',
                      'from .vector_study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha')
    code=code.replace('from diagnostics.controller_q_diversity_v1.train import Critic','from .vector_net import Critic')
    code=code.replace('def metrics(z,d,ix):','def scalar_metrics(z,d,ix):')
    pos=code.index('\ndef fit(index):')
    code=code[:pos]+'''

def metrics(z,d,ix):
    if z.ndim==2:return scalar_metrics(z,d,ix)
    q=1/(1+np.exp(-np.clip(z,-80,80)))
    s,f=d['success'][ix],d['failure'][ix];count=s+f;valid=count>0
    truth=s/np.maximum(count,1)
    groups=[];scene_means=[]
    for sc in sorted(set(d['scene'][ix])):
        sub=[]
        for c in sorted(set(d['controller'][ix][d['scene'][ix]==sc])):
            ii=np.flatnonzero((d['scene'][ix]==sc)&(d['controller'][ix]==c))
            support=valid[ii].all(0);delta=q[ii]-truth[ii]
            aa=q[ii]-q[ii].mean(0,keepdims=True);bb=truth[ii]-truth[ii].mean(0,keepdims=True)
            ee=np.flatnonzero(support.any(-1));cc=np.flatnonzero(support.any(0))
            aa=aa[:,ee][:,:,cc];bb=bb[:,ee][:,:,cc]
            aa-=aa.mean(1,keepdims=True);bb-=bb.mean(1,keepdims=True)
            g=dict(scene=str(sc),controller=str(c),N=len(ii),
                state_eta_MSE=float(np.mean((aa-bb)**2)),
                Q_MSE=float(np.mean(delta[valid[ii]]**2)),
                nll=float(np.sum(s[ii]*np.logaddexp(0,-z[ii])+f[ii]*np.logaddexp(0,z[ii]))/max(count[ii].sum(),1)))
            groups.append(g);sub.append(g)
        scene_means.append({k:float(np.mean([g[k] for g in sub])) for k in ('state_eta_MSE','Q_MSE','nll')})
    answer={k:float(np.mean([g[k] for g in scene_means])) for k in ('state_eta_MSE','Q_MSE','nll')}
    answer.update(groups=groups,regret=answer['state_eta_MSE'],ranking=0.)
    return answer
'''+code[pos:]
    code=code.replace('eligible=s+f>0','eligible=s+f>0\n    if eligible.ndim==3:eligible=eligible.any(-1)')
    code=code.replace("net=Critic('eta_only' if job['arm'].endswith('eta_only') else 'state_field_compact')",
        "net=Critic('eta_only' if job['arm'].endswith('eta_only') else 'state_field_compact',8 if job['arm'].startswith('motion_') else 1)")
    old='return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1))'
    new='''value=(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1)
                if value.ndim==2:
                    active=(ss+ff)>0
                    return jnp.mean(jnp.sum(value*active,-1)/jnp.maximum(active.sum(-1),1))
                return jnp.mean(value)'''
    assert old in code;code=code.replace(old,new)
    code=code.replace('return np.concatenate(parts).reshape(-1,16)',
        "return np.concatenate(parts).reshape((-1,16,8) if job['arm'].startswith('motion_') else (-1,16))")
    code=code.replace("key=(m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)",
        "key=(m['state_eta_MSE'],m['Q_MSE'],m['nll'],step) if job['arm'].startswith('motion_') else (m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)")
    (PARENT/'vector_learn.py').write_text(code)
    net_src=(PARENT.parent/'controller_q_diversity_v1/train.py').read_text()
    body=net_src[net_src.index('class Critic(nn.Module):'):net_src.index('\ndef normalization')]
    body=body.replace('    kind:str','    kind:str\n    outputs:int=1')
    body=body.replace("return nn.Dense(1,name='out')(h)[...,0]",
        "z=nn.Dense(self.outputs,name='out')(h)\n        return z[...,0] if self.outputs==1 else z")
    assert body.count('self.outputs')==2
    (PARENT/'vector_net.py').write_text('import jax.numpy as jnp\nimport flax.linen as nn\nfrom diagnostics.orthoflow3_unified_representation_v1 import representation as rep\n\n'+body)
    p['experiment']['code_sha256']=sha(PARENT/'vector_learn.py')
    p['experiment']['network_sha256']=sha(PARENT/'vector_net.py')
    p['experiment']['preparation_sha256']=sha(__file__)
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),source_sha256=sha(PARENT/'dataset.npz'),
        matching_training_mask=True,physical_states=160,rows=640,new_rollout=0,new_query=0,original_scalar_fits=0))
    print(dict(stage='prepared_full_vector',joint_fits=18,scalar_fits_pruned=True,new_rollout=0))


def fit(index):
    from . import vector_learn as learn
    run=RUNS[index]
    for target in ('return','motion'):
        job=dict(fold=run['fold'],arm=target+'_'+run['representation'],seed=run['seed'])
        learn.fit(learn.JOBS.index(job))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();prepare() if a.stage=='prepare' else fit(a.index)

