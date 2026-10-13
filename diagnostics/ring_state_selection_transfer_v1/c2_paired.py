"""Paired Q-risk and descriptor-use diagnostic, physical-state bootstrap."""
import numpy as np
from scipy.special import expit
from study import OUT,read,write
from analyze_learning import ci
p=np.load(OUT/'C2_FALLBACK_PREDICTIONS.npz');y=np.load(OUT/'c1/labels.npz')['Y'];ix=np.array(read(OUT/'C2_SPLITS.json')['joint_test_states']);s=(y[[1,4]][:,ix,:,4:]==1).sum(-1);n=(y[[1,4]][:,ix,:,4:]>=0).sum(-1);q=np.divide(s,n,out=np.full(s.shape,np.nan),where=n>0);loss={};rows=[]
for arm in ['R0','R1','R2','R3','R4','R4_action','RB']:
 a=expit(p[arm][:,[1,4]][:,:,ix]);b=expit(p[arm+'_mismatched'][:,[1,4]][:,:,ix]);l=np.nanmean((a-q)**2,axis=(0,1,3));m=np.nanmean((b-q)**2,axis=(0,1,3));loss[arm]=l;gain=m-l
 rows.append(dict(arm=arm,mean_Q_MSE=float(l.mean()),mismatched_minus_matched_MSE=float(gain.mean()),descriptor_use_ci95=ci(gain),R0_minus_arm_MSE=float((loss['R0']-l).mean()),R0_minus_arm_MSE_ci95=ci(loss['R0']-l)))
write(OUT/'C2_PAIRED_Q_DIAGNOSTIC.json',dict(rows=rows,physical_test_states=len(ix),shared_across_held_controllers=True,scope='Exploratory Q error only; same eight physical test states average controllers/fits before bootstrap. Not selection success.'));print(rows)
