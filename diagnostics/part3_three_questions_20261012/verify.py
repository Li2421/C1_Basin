"""Verify split/label firewall and independent linear solve for one saved fit."""
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from run import HERE,ROOT,old,transform,choose,fit_predict,indices

def main():
    for line in (HERE/'frozen_inputs.sha256').read_text().splitlines():
        h,path=line.split(None,1)
        assert hashlib.sha256((ROOT/path.strip()).read_bytes()).hexdigest()==h
    names,blocks,y,u,lo,hi,mid=old.load()
    p=json.loads((old.P1/'P1_PROTOCOL.json').read_text())
    state=np.array([np.concatenate([np.array(s['physical'][k]).ravel() for k in ('positions','velocities','goals')]) for s in p['states']])
    parts=[np.broadcast_to(state,(14,24,24)).reshape(336,24)]+[a.reshape(336,-1) for a in blocks['R5']]
    q=mid.reshape(336,16)
    folds=json.loads((HERE/'folds.json').read_text())
    coverage=np.zeros((14,24),int)
    for fold in folds:
        held=names.index(fold['held'])
        assert held not in fold['train_controllers']
        assert not set(fold['train_states'])&set(fold['test_states'])
        coverage[held,fold['test_states']]+=1
    assert np.all(coverage==1)
    fold=folds[0];src=fold['train_controllers'];ss=fold['train_states']
    train=indices(src,ss);test=indices([0],fold['test_states'])
    ta=q.copy();excluded=np.setdiff1d(np.arange(336),train);ta[excluded]=1-ta[excluded]
    key,scores=choose(parts,q,src,ss,'R5');key2,scores2=choose(parts,ta,src,ss,'R5')
    assert key==key2 and scores==scores2
    a,_,_,_=fit_predict(parts,q,train,test,'R5',key)
    b,_,_,_=fit_predict(parts,ta,train,test,'R5',key)
    assert np.array_equal(a,b)
    x,v=transform(parts,train,test,'R5');bw,reg=key
    mean=q[train].mean(0);k=np.exp(-cdist(x,x,'sqeuclidean')/(2*bw*bw))
    coef=np.linalg.solve(k+len(train)*reg*np.eye(len(train)),q[train]-mean)
    direct=np.clip(mean+np.exp(-cdist(v,x,'sqeuclidean')/(2*bw*bw))@coef,0,1)
    err=float(np.max(np.abs(a-direct)));assert err<1e-9
    pred=np.load(HERE/'predictions.npz')['prediction']
    assert np.max(np.abs(a-pred[1,0,fold['test_states']]))<1e-12
    out=dict(frozen_input_hashes_verified=True,outer_folds=len(folds),complete_controller_state_coverage=True,all_outer_controller_and_state_disjoint=True,one_fold_held_controller_and_state_label_tamper_invariant=True,independent_direct_solve_max_error=err,new_rollouts=0)
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
