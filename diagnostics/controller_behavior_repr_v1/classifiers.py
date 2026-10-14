from core import *
from data import Data,Representations,name,logistic
import shutil

def main():
    gate();r=Representations();records=[]
    for t in read(HERE/'TASKS.json'):
        d=Data(t);out=HERE/'classifiers'/name(t);out.mkdir(parents=True,exist_ok=True)
        if t['reuse']:
            ot=dict(t,coverage='LOCO' if t['bed']=='real' else t['coverage']);f=R6/'fits'/old_name(ot,'F2_PCA2');oldcf=R6/'logistic'/old_name(ot,'F4');dest=HERE/'fits'/name(t);dest.mkdir(parents=True,exist_ok=True)
            for file in ('predictions.npz','normalization.npz'):shutil.copyfile(f/file,dest/file)
            meta=read(f/'complete.json');write(dest/'complete.json',dict(meta,task=t,reused_from=str(f),optimization_performed=False,reuse_prediction_sha256=sha(f/'predictions.npz')))
            for file in ('predictions.npz','classifier.npz'):shutil.copyfile(oldcf/file,out/file)
            write(out/'complete.json',dict(read(oldcf/'complete.json'),task=t,reused_from=str(oldcf)))
        else:
            x=np.array([d.coords[n] for n in d.sources]);h=np.array([d.coords[n] for n in d.held]);y=np.array([d.regimes[n] for n in d.sources]);p,model=logistic(x,y,h);choice=np.where(p.max(1)>=.6,model['classes'][p.argmax(1)],d.src);choices=np.tile(choice[None,:,None],(3,1,len(d.test_states)))
            np.savez_compressed(out/'predictions.npz',choices=choices,regime_probabilities=p,classes=model['classes'],held_controller_names=d.held);np.savez_compressed(out/'classifier.npz',**{k:v for k,v in model.items() if k not in ('success','iterations')});write(out/'representation.json',d.rep_meta)
            write(out/'complete.json',dict(status='COMPLETE',utc=utc(),task=t,source_regimes=d.regimes,source_prior_eta=d.src,model=model,choice=choice,threshold=.6,L2=1.,test_labels_read=False,target_labels_used=False))
        _,delta,meta=r.behavior(d.sources,t['sensitivity']);target_delta=np.array([delta[r.names.index(n),0] for n in d.held]);a=np.where(target_delta>0,13,10) if d.src in (10,13) else np.full(len(d.held),d.src)
        choices=np.tile(a[None,:,None],(3,1,len(d.test_states)));np.savez_compressed(out/'P_rule.npz',choices=choices,raw_delta_progress=target_delta,source_prior_eta=d.src)
        records.append(dict(task=t,source_prior=d.src,regimes=d.regimes,P_rule_choices=a,P_rule_delta=target_delta))
    write(HERE/'CLASSIFIERS_INDEX.json',records)
if __name__=='__main__':main()
