from core import *
from collections import Counter
import csv
K=[1,2,4,8,16];R=1000
def tied(values,indices,prior):
 best=np.max(values);tie=[indices[j] for j in np.flatnonzero(values==best)];return prior if prior in tie else tie[0]
def analyze_test(tag,ti):
 meta=read(HERE/'INPUT_IDENTITIES.json');groups=meta['groups'];source=np.load(HERE/'reconstructed/R15.npz');src=source['success'].mean((0,1,3));srcarms=source['arms'].tolist();cohort={'T1':'R14A','T2':'R10','T3':'R12','T4':'R17','T5':'R18TEST'}[tag];z=np.load(HERE/'reconstructed'/f'{cohort}.npz');allnames=z['names'].tolist();names=groups[tag];ci=[allnames.index(n) for n in names];arms=z['arms'].tolist();states=z['states'].tolist();Y=z['success'][ci];U=z['unknown'][ci];available=z['available'][ci];pi=np.argsort(np.asarray(states));cal=pi[:len(pi)//2];ev=pi[len(pi)//2:];assert not set(cal)&set(ev);P=np.load(HERE/('T2_ALIGNED_PREDICTIONS.npz' if tag=='T2' else f'{tag}_PREDICTIONS.npz'));assert P['controller_names'].tolist()==names;psi=P['state_uids'].tolist();porder=[psi.index(s) for s in states];pa=P['arms'].tolist();pred={m:P[m][:,:,porder,:] for m in ['M3','M4']};aliases={}
 if tag=='T2':
  for case in read(D/'controller_universal_eta_schedule_v1/main_CASES.json'):
   if case.get('arm')=='Freg':aliases[case['base_controller_name']]='F'+str(case['eta_index'])
 nc=len(names);na=len(arms);nm={};choice_records=[];simulation={};menus=[]
 def put(method,choices):
  weights=np.zeros((nc,na),float)
  for i,cc in enumerate(choices):
   cc=np.asarray(cc,int).reshape(-1);weights[i]=np.bincount(cc,minlength=na)/len(cc)
   if len(cc)==1:choice_records.append(dict(test=tag,controller=names[i],method=method,selected_arm=arms[cc[0]],target_labels=0))
  nm[method]=weights
 zero={m:[] for m in ['M0','M1','M2','M3_fixed','M3_full','M4_fixed','M4_full','M5']}
 for i,n in enumerate(names):
  order=lambda j:(srcarms.index(arms[j]) if arms[j] in srcarms else len(srcarms),arms[j]);fixed=sorted([j for j,a in enumerate(arms) if a.startswith('F') and available[i,j]],key=order);full=sorted([j for j in range(na) if available[i,j]],key=order);head=lambda j:aliases.get(n,arms[j]) if arms[j]=='Freg' else arms[j];supported=[j for j in full if head(j) in pa];sf=[j for j in fixed if head(j) in pa];source_value=lambda j:src[srcarms.index(head(j))] if head(j) in srcarms else -np.inf;prior={False:max(sf,key=source_value),True:max(supported,key=source_value)};menus.append(dict(fixed=fixed,full=full,supported_fixed=sf,supported_full=supported,head_alias={str(j):head(j) for j in full}));zero['M0'].append(arms.index('F10'));zero['M1'].append(prior[False]);zero['M2'].append(prior[True]);zero['M5'].append(arms.index(str(P['GE_choices'][i])))
  for m in ['M3','M4']:
   q=pred[m][:,i,ev].mean((0,1))
   for menu,jj in [('fixed',sf),('full',supported)]:zero[m+'_'+menu].append(max(jj,key=lambda j:q[pa.index(head(j))]))
 for m,cc in zero.items():put(m,cc)
 budget=np.zeros((nc,2),int);simchoices={}
 for menu_index,menu in enumerate(['fixed','full']):
  for method in ['M6_E','M6_Thompson','M7']:
   for k in K:simchoices[f'{method}_{menu}_k{k}']=np.zeros((nc,R),np.int16)
  for i,n in enumerate(names):
   jj=menus[i][menu];sj=menus[i]['supported_'+menu];budget[i,menu_index]=len(jj)
   for repeat in range(R):
    rng=np.random.default_rng(202610152600+10000*ti+100*i+repeat+menu_index*1000000);trng=np.random.default_rng(202610152600+10000*ti+100*i+repeat+menu_index*1000000+90000000);bank={}
    for j in jj:
     valid=np.argwhere(U[i,cal,j]==0);assert len(valid)>=16;order=rng.permutation(len(valid))[:16];pairs=valid[order];cs=cal[pairs[:,0]];seed=pairs[:,1];vals=Y[i,cs,j,seed];bank[j]=(cs,vals)
    prior=zero['M1' if menu=='fixed' else 'M2'][i]
    for k in K:
     means=np.array([bank[j][1][:k].mean() for j in jj]);ec=tied(means,jj,prior);success=np.array([bank[j][1][:k].sum() for j in jj]);tc=jj[int(trng.beta(1+success,1+k-success).argmax())];simchoices[f'M6_E_{menu}_k{k}'][i,repeat]=ec;simchoices[f'M6_Thompson_{menu}_k{k}'][i,repeat]=tc;modified=[]
     for j in sj:
      cs,vals=bank[j];a=pa.index(menus[i]['head_alias'][str(j)]);pool=pred['M4'][:,i,cs[:k],a];offset=(vals[:k][None]-pool).sum(1)/(k+8);q=np.clip(pred['M4'][:,i,ev,a]+offset[:,None],0,1).mean();modified.append(q)
     simchoices[f'M7_{menu}_k{k}'][i,repeat]=tied(np.asarray(modified),sj,prior)
   print(tag,n,'simulation',menu,flush=True)
 for method,cc in simchoices.items():put(method,cc)
 np.savez_compressed(HERE/f'{tag}_SIMULATIONS.npz',**simchoices,names=names,arms=arms,calibration_state_uids=np.asarray(states)[cal],evaluation_state_uids=np.asarray(states)[ev]);methods=list(nm);W=np.stack([nm[m] for m in methods],axis=1);lo=Y[:,ev].mean(-1);hi=(Y[:,ev]+U[:,ev]).mean(-1);rates=lo.mean(1);rates_hi=hi.mean(1);oracle=np.where(available,rates,-np.inf).max(1);oraclehi=np.where(available,rates_hi,-np.inf).max(1);fixedmask=available&np.array([a.startswith('F') for a in arms])[None];fo=np.where(fixedmask,rates,-np.inf).max(1);policy=np.einsum('nma,na->nm',W,rates);policyhi=np.einsum('nma,na->nm',W,rates_hi);reg=oracle[:,None]-policy;regmin=oracle[:,None]-policyhi;regmax=oraclehi[:,None]-policy;f10=rates[:,arms.index('F10')];point=[]
 for j,m in enumerate(methods):
  labels=np.zeros(nc,int) if not m.startswith(('M6','M7')) else budget[:,0 if '_fixed_' in m else 1]*int(m.split('_k')[1])
  if m.startswith('M7'):
   labels=np.array([len(x['supported_fixed' if '_fixed_' in m else 'supported_full']) for x in menus])*int(m.split('_k')[1])
   # All M7 draws for unsupported arms are ignored: charge only its supported menu.
  catastrophe=policy[:,j]<=f10-.10+1e-14
  expected_cat=0.
  for i in range(nc):expected_cat+=float(np.sum(W[i,j]*(rates[i]<=f10[i]-.10+1e-14)))
  point.append(dict(test=tag,method=m,mean_regret=float(reg[:,j].mean()),maximum_regret=float(reg[:,j].max()),mean_success=float(policy[:,j].mean()),mean_success_upper=float(policyhi[:,j].mean()),regret_unknown_bounds=[float(regmin[:,j].mean()),float(regmax[:,j].mean())],catastrophe_count=int(catastrophe.sum()),expected_per_repeat_catastrophes=expected_cat,target_labels_min=int(labels.min()),target_labels_max=int(labels.max()),target_labels_mean=float(labels.mean()),fixed_oracle_mean_regret=float((fo-policy[:,j]).mean())))
 for m,w in nm.items():
  if m.startswith(('M6','M7')):
   kk=int(m.split('_k')[1]);full='_full_' in m
   for i,n in enumerate(names):
    nlabels=kk*len(menus[i][('supported_' if m.startswith('M7') else '')+('full' if full else 'fixed')])
    for a in np.flatnonzero(w[i]):choice_records.append(dict(test=tag,controller=n,method=m,selected_arm=arms[a],choice_probability=float(w[i,a]),target_labels=nlabels))
 # Pooled arrays pad controller state samples with per-controller metadata below.
 np.savez_compressed(HERE/f'{tag}_ANALYSIS.npz',lo=lo,hi=hi,weights=W,available=available,fixedmask=fixedmask,methods=methods,names=names,arms=arms,policy=policy,regret=reg,budget=budget)
 write(HERE/f'{tag}_POINT_RESULTS.json',dict(test=tag,rows=point,menus=menus,arms=arms,controllers=names,unknown_calibration=int(U[:,cal].sum()),unknown_evaluation=int(U[:,ev].sum()),evaluation_state_count=len(ev),training_source_arm_order=srcarms))
 return point,choice_records
def main():
 prereg();assert read(HERE/'RECONSTRUCTION_GATE.json')['passed'];assert (HERE/'MODEL_METHOD_FREEZE.receipt.json').exists();points=[];choices=[]
 for ti,tag in enumerate(['T1','T2','T3','T4','T5']):
  if (HERE/f'{tag}_POINT_RESULTS.json').exists() and (HERE/f'{tag}_ANALYSIS.npz').exists():
   r=read(HERE/f'{tag}_POINT_RESULTS.json');pp=r['rows'];zz=np.load(HERE/f'{tag}_ANALYSIS.npz');cc=[]
   for j,m in enumerate(zz['methods'].tolist()):
    for i,n in enumerate(zz['names'].tolist()):
     labels=0 if not m.startswith(('M6','M7')) else int(m.split('_k')[1])*len(r['menus'][i][('supported_' if m.startswith('M7') else '')+('fixed' if '_fixed_' in m else 'full')])
     for a in np.flatnonzero(zz['weights'][i,j]):cc.append(dict(test=tag,controller=n,method=m,selected_arm=zz['arms'].tolist()[a],choice_probability=float(zz['weights'][i,j,a]),target_labels=labels))
  else:pp,cc=analyze_test(tag,ti)
  points+=pp;choices+=cc
 write(HERE/'POINT_RESULTS.json',points)
 with (HERE/'METHOD_CHOICES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['test','controller','method','selected_arm','choice_probability','target_labels']);w.writeheader();w.writerows(choices)
 print('Simulations complete',flush=True)
if __name__=='__main__':main()
