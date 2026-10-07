"""Source-controller CV local controls; opened targets are diagnostic only.

Same exact eta regression is not continuous-eta or cross-scene validation.
Native slot flattening is a privileged diagnostic, not deployable invariance.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .goal_response_cv import DATA, OUT, load, folds, csvwrite, read, write
from .simple_response import pooled, loss
from .function_local import predict


def physical(x,kind):
    if kind=='pooled_physical': return pooled(x)
    if kind=='native_slot_physical':
        return np.concatenate([x[k].reshape(len(x[k]),-1) for k in ('agents','pairs','obstacles','globals')],1)
    return np.zeros((len(x['agents']),1))


def normalize_features(d,x,fit,controllers,kind):
    h=physical(x,kind);fs=np.unique(d['state_index'][fit]);hc=h[fs].mean(0);hs=np.maximum(h[fs].std(0),.05)
    c=d['input_context'];cc=c[controllers][:,fit].mean((0,1));cs=np.maximum(c[controllers][:,fit].std((0,1)),.2)
    hn=(h[d['state_index']]-hc)/hs/np.sqrt(h.shape[-1]);cn=(c-cc)/cs/np.sqrt(c.shape[-1])
    z=np.concatenate([np.broadcast_to(hn,(12,*hn.shape)),cn],-1)
    return z,dict(hc=hc,hs=hs,cc=cc,cs=cs)


def metrics(p,s,f):
    q=s/(s+f);p=np.clip(p,1e-5,1-1e-5);pp=p.reshape(-1,2);qq=q.reshape(-1,2);g=s.reshape(-1,2)>=15;ch=pp.argmax(1);ix=np.arange(len(ch))
    return dict(NLL=float(loss(p,q).mean()),MAE=float(abs(p-q).mean()),B15=int(g[ix,ch].sum()),oracle=int(g.any(1).sum()),selected_Q=float(qq[ix,ch].mean()),cases=len(ch),
        contrast_correlation=float(np.corrcoef(qq[:,0]-qq[:,1],pp[:,0]-pp[:,1])[0,1]) if np.std(pp[:,0]-pp[:,1])>1e-9 else None)


def main():
    dest=OUT/'local_diagnostic';dest.mkdir(exist_ok=True)
    assert not (dest/'complete.json').exists(),'Already frozen; do not retune'
    kinds=('context_only','pooled_physical','native_slot_physical');rows=[]
    for fold,held in enumerate(folds()):
        fitc=[c for c in range(12) if c not in held];d,x,tr,va,norm=load(fitc);q=d['success']/(d['success']+d['failure'])
        for kind in kinds:
            z,_=normalize_features(d,x,tr,fitc,kind)
            tz=z[fitc][:,tr].reshape(-1,z.shape[-1]);te=np.tile(d['eta_index'][tr],8);tq=q[fitc][:,tr].ravel()
            vz=z[held][:,va].reshape(-1,z.shape[-1]);ve=np.tile(d['eta_index'][va],4)
            for k in (5,20,100):
                pr=predict(tz,te,tq,vz,ve,k);met=metrics(pr,d['success'][held][:,va].ravel(),d['failure'][held][:,va].ravel())
                rows.append(dict(fold=fold,kind=kind,k=k,**met))
    csvwrite(dest/'source_cv.csv',rows)
    selected={kind:min((np.mean([r['NLL'] for r in rows if r['kind']==kind and r['k']==k]),k) for k in (5,20,100))[1] for kind in kinds}
    write(dest/'source_selection.json',dict(selected_k=selected,criterion='Pooled3sourcecontrollerCVNLL only',targets_already_opened=True,target_role='Post-hoc diagnostic; no new independentgeneralizationclaim'))
    d,x,tr,va,norm=load(list(range(12)));q=d['success']/(d['success']+d['failure']);out=[];predictions={}
    for kind in kinds:
        z,n=normalize_features(d,x,tr,list(range(12)),kind);tz=z[:,tr].reshape(-1,z.shape[-1]);te=np.tile(d['eta_index'][tr],12);tq=q[:,tr].ravel()
        for ri,name in enumerate(('goal_response_confirmation','goal_response_confirmation_b')):
            target=OUT.parent/name;dd=np.load(target/'dataset.npz');xx=dict(np.load(target/'entities.npz'));gg=np.load(target/'goal_held.npz')['goal_response']
            c=np.concatenate([(dd['context']-norm['context_center'])/norm['context_scale'],((dd['agent_response']-norm['agent_center'])/norm['agent_scale']).reshape(128,-1),(gg-norm['goal_center'])/norm['goal_scale']],-1)
            h=physical(xx,kind);hh=(h[dd['state_index']]-n['hc'])/n['hs']/np.sqrt(h.shape[-1]);cc=(c-n['cc'])/n['cs']/np.sqrt(c.shape[-1]);vz=np.concatenate([hh,cc],-1)
            pr=predict(tz,te,tq,vz,np.tile([10,15],64),selected[kind]);predictions[f'{kind}__{ri}']=pr
            out.append(dict(kind=kind,replicate=ri,k=selected[kind],**metrics(pr,dd['success'],dd['failure'])))
    csvwrite(dest/'diagnostic_target_metrics.csv',out);np.savez_compressed(dest/'diagnostic_predictions.npz',**predictions)
    write(dest/'complete.json',dict(source_cv=rows,selected=selected,target_diagnostic=out,new_rollouts=0,model_changes=0,
        scope='Post-hoc localization of neuralvsrepresentation failure on two already-opened controllers; k chosenonlysourceCV. Native-slotbaselineprivileged, notpermutation-compatible crossscene architecture.'))
    print(out)


if __name__=='__main__':main()
