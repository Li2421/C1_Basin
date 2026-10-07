"""Separation is necessary, not sufficient: family-held-out linear interaction probe.

Small counterfactual panel only. Fixed regularization; never tunes LOSO features.
"""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SWAP=ROOT/'diagnostics/orthoflow3_controller_conditioning_probe_v1'
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 contexts=load(OUT/'swap_contexts.json');labels=list(csv.DictReader((SWAP/'controller_pair_results.csv').open()))
 label={(r['state_uid'],r['eta_uid']):r for r in labels}
 hh=np.load(ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/cohort_features.npz')['h_raw']
 pairs=[r for r in contexts if r['candidate']=='C1'];cc={k:[r for r in contexts if r['candidate']==k] for k in ('C1','C3')}
 families=sorted({r['source_group'] for r in pairs},key=lambda s:hashlib.sha256(s.encode()).hexdigest());fold={f:i%4 for i,f in enumerate(families)}
 h=np.repeat(np.array([hh[r['episode_index']] for r in pairs]),2,axis=0).astype(float)
 e=np.repeat(np.array([r['eta'] for r in pairs]),2,axis=0)
 y=np.array([[float(label[(r['state_uid'],r['eta_uid'])]['base_Q16']),float(label[(r['state_uid'],r['eta_uid'])]['alternate_Q16'])] for r in pairs]).reshape(-1)
 split=np.repeat([fold[r['source_group']] for r in pairs],2)
 contexts2={k:np.array([r['contexts'] for r in cc[k]]).reshape(-1,10) for k in cc}
 base=np.c_[h,e,e**2];design={'C0':base,'eta_only':np.c_[e,e**2]}
 for k,c in contexts2.items():design[k]=np.c_[base,c,(c[:,:,None]*e[:,None,:]).reshape(len(c),-1)]
 predictions={};results=[]
 for kind,xx in design.items():
  pred=np.zeros(len(y));wrong=np.zeros(len(y));details=[]
  for f in range(4):
   tr=split!=f;te=split==f;mean=xx[tr].mean(0);scale=np.maximum(xx[tr].std(0),.05)
   x=np.c_[np.ones(len(xx)),(xx-mean)/scale];xt=x[tr];yt=y[tr]
   def fun(w):
    z=xt@w;loss=np.mean(np.logaddexp(0,z)-yt*z)+.01*np.sum(w[1:]**2)/2
    g=xt.T@(expit(z)-yt)/len(yt);g[1:]+=.01*w[1:];return loss,g
   fit=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1500,'gtol':1e-7})
   pred[te]=expit(x[te]@fit.x)
   wrong[te]=expit(x.reshape(-1,2,x.shape[1])[:,::-1,:].reshape(x.shape)[te]@fit.x)
   details.append({'fold':f,'source_families':12,'test_families':4,'converged':bool(fit.success)})
  delta=y.reshape(-1,2)[:,1]-y.reshape(-1,2)[:,0];pd=pred.reshape(-1,2)[:,1]-pred.reshape(-1,2)[:,0];m=abs(delta)>=.5
  correct=np.where(abs(pd[m])<1e-8,.5,(pd[m]*delta[m]>0).astype(float))
  nll=lambda p:float(np.mean(-y*np.log(np.clip(p,1e-8,1-1e-8))-(1-y)*np.log1p(-np.clip(p,1e-8,1-1e-8))))
  results.append({'kind':kind,'heldout_NLL':nll(pred),'heldout_MAE':float(abs(pred-y).mean()),'wrong_controller_NLL':nll(wrong),
   'large_Q_change_pairs':int(m.sum()),'controller_ordering_accuracy_ties_half':float(correct.mean()),'folds':details})
  predictions[kind]=pred.tolist()
 separation=[]
 for k in cc:
  dist=np.array([r['context_distance'] for r in cc[k]]);delta=abs(y.reshape(-1,2)[:,1]-y.reshape(-1,2)[:,0])
  separation.append({'kind':k,'pairs':len(dist),'separated':int((dist>1e-6).sum()),'context_distance_min':float(dist.min()),
    'context_distance_median':float(np.median(dist)),'distance_abs_deltaQ_spearman':float(spearmanr(dist,delta).statistic)})
 dump(OUT/'swap_audit.json',{'separation':separation,'heldout_probe':results,'predictions':predictions,
  'protocol':'4 source-family folds, all eta+controller conditions per family kept together; ridge .01 fixed; no model selection; linear h/eta/quadratic eta + context*eta interactions',
  'scope':'small controller-counterfactual diagnostic; not unseen-scene evidence or sufficiency proof',
  'baseline_identical_input_max_error_from_original_probe':5.967e-16,'new_rollouts':0})
 print(json.dumps({'separation':separation,'heldout_probe':results},indent=2))
if __name__=='__main__':main()
