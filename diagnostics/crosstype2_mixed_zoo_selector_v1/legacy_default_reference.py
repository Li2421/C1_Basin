"""Read-only F10 reference where exact historical training evidence exists."""
from common import *
from zoo_design import profiles,case
from statistics_pipeline import evaluate_group,equivalence


def main():
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in ('names','types','labels','uids','collision','timeout','unknown')}
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    base=profiles();states=read(HERE/'TRAINING_STATES.json');reference=np.full((len(data['names']),48),-2,np.int8);uids=np.full(reference.shape,'',dtype='U96')
    events={k:np.zeros(reference.shape,np.int8) for k in ('collision','timeout','unknown')}
    with rollout_db.connect(True) as db:
        for i,name in enumerate(data['names']):
            for s,state in enumerate(states):
                c,_=case(base,str(name),state,17);row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                if row is None:continue
                reference[i,s]=-1 if row['numerical_failure'] else row['success'];uids[i,s]=c['rollout_uid']
                for key in events:events[key][i,s]=row['numerical_failure'] if key=='unknown' else row[key]
    complete=np.all(reference!=-2,axis=1);result={}
    for kind in sorted(set(data['types'])):
        mask=data['types']==kind;group=dict(controllers=int(mask.sum()),complete_controller_evidence=int(np.sum(complete&mask)),historical_only=True,methods={})
        if np.all(complete[mask]):
            subset={k:data[k][mask] for k in ('names','types')}
            subset['labels']=np.concatenate((data['labels'][mask,...,:1],reference[mask,:,None,None]),axis=2)
            subset['uids']=np.concatenate((data['uids'][mask,...,:1],uids[mask,:,None,None]),axis=2)
            for key in events:subset[key]=np.concatenate((data[key][mask,...,:1],events[key][mask,:,None,None]),axis=2)
            subset['uid_class']=equivalence(subset['uids']);nc=int(mask.sum())
            fixed={k:methods[k][mask] for k in ('pipeline','all_Sprog','all_safety_baseline')};fixed['all_F10']=np.full((nc,48),16,np.int32)
            group['reference_metrics']=evaluate_group(subset,fixed,True,5000)['methods']['all_F10']
            group['reference_metrics'].update(disasters_vs_F10=0,disasters_vs_F10_certain=0,disaster_unavailable_reason=None,reference_evidence_level='DEVELOPMENT exact historical reference')
        for key,choice in methods.items():
            if not np.all(complete[mask]):group['methods'][key]=dict(possible=None,certain=None,reason='Not all type instances have full48-state historical F10 evidence; no extra rollout requested');continue
            selected=np.take_along_axis(data['labels'][mask,...,0],choice[mask,...,None],-1)[...,0];selected_uid=np.take_along_axis(data['uids'][mask,...,0],choice[mask,...,None],-1)[...,0]
            same=selected_uid==uids[mask];lower=selected==1;upper=selected!=0;lo=reference[mask]==1;hi=reference[mask]!=0
            possible=np.where(same,0,hi.astype(float)-lower).mean(1);certain=np.where(same,0,lo.astype(float)-upper).mean(1)
            group['methods'][key]=dict(possible=int(np.sum(possible>=.1-1e-12)),certain=int(np.sum(certain>=.1-1e-12)))
        result[str(kind)]=group
    write(HERE/'LEGACY_F10_REFERENCE.json',dict(utc=utc(),types=result,complete_controllers=data['names'][complete].tolist(),missing_controllers=data['names'][~complete].tolist(),new_physical_rollouts=0,new_evaluation_reference_available=False))
    seal('LEGACY_F10_REFERENCE.receipt.json',[HERE/'LEGACY_F10_REFERENCE.json',HERE/'legacy_default_reference.py'],exact_historical_UIDs_only=True)
    print('Historical F10 disaster counts available for',int(complete.sum()),'of',len(complete),'controllers; no new rollouts')


if __name__=='__main__':main()
