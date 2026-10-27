"""Predeclared comparisons, paired two-level uncertainty and exact8! scene matching."""
from common import *
from design import *
from stages import read_result
import numpy as np,itertools
METHODS=['v2','v2_49','v2_49_no_fallback','global_fixed','knn','scene_xstate','safety','generic_input']
NAMES={'v2':'学习选η v2','v2_49':'v2·只在49个旧候选里选','v2_49_no_fallback':'v2·不退回','global_fixed':'一个全局η（训练场景里最好）','knn':'按场景近邻选η','scene_xstate':'每个场景自己的最好η（参照）','safety':'只加安全投影','generic_input':'通用输入版本'}
P=200;B=5000;SEED=2026102652

def own_rates(counts,rng):
 M,S,A=counts.shape;order=np.argsort(rng.random((M,P,S)),axis=-1,kind='stable')[:,:,:S//2];tr=counts[np.arange(M)[:,None,None],order].sum(2);te=counts.sum(1)[:,None]-tr;which=tr.argmax(-1)
 return np.take_along_axis(te,which[...,None],-1)[...,0].mean(1)/(3*(S//2))

def bootstrap(arr,counts,rng,hierarchical):
 M,S,K=arr.shape;out=np.empty((B,K))
 for b in range(0,B,50):
  n=min(50,B-b);scene=rng.integers(M,size=(n,M))if hierarchical else np.broadcast_to(np.arange(M),(n,M));state=rng.integers(S,size=(n,M,S))
  sampled=arr[scene[:,:,None],state];rates=sampled.mean(2);lab=counts[scene[:,:,None],state]
  order=np.argsort(rng.random((n,M,P,S)),axis=-1,kind='stable')[:,:,:,:S//2];tr=lab[np.arange(n)[:,None,None,None],np.arange(M)[None,:,None,None],order].sum(3,dtype=np.int16);te=lab.sum(2,dtype=np.int16)[:,:,None]-tr;idx=tr.argmax(-1)
  own=np.take_along_axis(te,idx[...,None],-1)[...,0].mean(2)/(3*(S//2));rates[:,:,METHODS.index('scene_xstate')]=own
  out[b:b+n]=rates.mean(1)
 return out

def matching(matrix):
 m=np.asarray(matrix,float);M=len(m);perms=np.asarray(list(itertools.permutations(range(M))),int);diag=float(np.trace(m));traces=m[np.arange(M)[None,:],perms].sum(1);observed=float((M*diag-m.sum())/(M*(M-1)))
 # Integer-success inputs have denominator48; quantize only exact trace comparison to avoid float-order ties.
 scaled=np.rint(m*48).astype(np.int64);observed_trace=int(np.trace(scaled));null=scaled[np.arange(M)[None,:],perms].sum(1);p=float(np.count_nonzero(null>=observed_trace)/len(perms))
 return dict(D=observed,p_one_sided=p,permutations=len(perms),diagonal_rate=diag/M,other_choice_rate=float((m.sum()-diag)/(M*(M-1))),matrix=m.tolist(),exact_integer_success_trace_test=True)

def assemble():
 tables=read(HERE/'LABEL_TABLES.json');evals=read(HERE/'EVALUATION_STATES.json')['scenes'];folds=read(HERE/'FOLDS.json');chosen={};refs={}
 for f in folds:chosen.update(read(HERE/'folds'/f['name']/'CHOICES.json'))
 for c in read(HERE/'proposal_CASES.json'):refs[(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'])]=read_result('proposal',c)
 names=list(tables);arr=np.zeros((len(names),16,len(METHODS)));upper=np.zeros_like(arr);counts=[];rows={}
 for ni,n in enumerate(names):
  row=tables[n];positions=[row['state_indices'].index(i)for i in evals[n]];y=np.asarray(row['y'])[:,positions];u=np.asarray(row['unknown'])[:,positions];counts.append(y.sum(0).astype(np.int16));ch=chosen[n]
  for mi,m in enumerate(METHODS):
   if m=='scene_xstate':continue
   a=ch['methods'][m]['selected_index']
   if a<49:values=y[:,:,a];uv=u[:,:,a]
   else:
    assert m=='v2';values=np.zeros((3,16));uv=np.zeros_like(values)
    for ci,ctl in enumerate(CONTROL_IDS):
     for si,state in enumerate(evals[n]):
      r=refs[(n,ctl,state)];values[ci,si]=r['success'];uv[ci,si]=r['numerical_failure']
   arr[ni,:,mi]=values.mean(0);upper[ni,:,mi]=(values+uv).mean(0)
  rows[n]=dict(state_indices=evals[n],methods=ch['methods'],guard=ch['guard'],shrinkage=ch['shrinkage'],candidate_keys=ch['candidate_keys'],candidate_etas=ch['candidate_etas'])
 return names,arr,upper,np.stack(counts),rows,tables

def main():
 from requests_corrected import verify_freezes
 verify_freezes();names,arr,upper,counts,rows,tables=assemble();rng=np.random.default_rng(SEED);own=own_rates(counts,rng);oi=METHODS.index('scene_xstate');arr[:,:,oi]=own[:,None];upper[:,:,oi]=own_rates(np.stack([(np.asarray(tables[n]['y'])+np.asarray(tables[n]['unknown']))[:,[tables[n]['state_indices'].index(i)for i in rows[n]['state_indices']]].sum(0).astype(np.int16)for n in names]),np.random.default_rng(SEED))[:,None]
 point=arr.mean((0,1));gi=METHODS.index('global_fixed');states=bootstrap(arr,counts,rng,False);hier=bootstrap(arr,counts,rng,True);np.savez_compressed(HERE/'METHOD_BOOTSTRAP.npz',states=states,scene_states=hier,methods=np.array(METHODS),scenes=np.array(names))
 comparisons={};per_scene={}
 for mi,m in enumerate(METHODS):
  ds=states[:,mi]-states[:,gi];dh=hier[:,mi]-hier[:,gi]
  comparisons[m]=dict(rate=float(point[mi]),unknown_success_upper=float(upper[:,:,mi].mean()),difference_from_global=float(point[mi]-point[gi]),ci_states=np.percentile(ds,[2.5,97.5]).tolist(),ci_scene_states=np.percentile(dh,[2.5,97.5]).tolist(),rate_ci_states=np.percentile(states[:,mi],[2.5,97.5]).tolist(),rate_ci_scene_states=np.percentile(hier[:,mi],[2.5,97.5]).tolist())
 for i,n in enumerate(names):per_scene[n]=dict(rates={m:float(arr[i,:,mi].mean())for mi,m in enumerate(METHODS)},differences_from_global={m:float((arr[i,:,mi]-arr[i,:,gi]).mean())for mi,m in enumerate(METHODS)},**rows[n])
 matches={}
 for m in ['v2_49','v2_49_no_fallback','knn','generic_input']:
  selected=[rows[n]['methods'][m]['selected_index']for n in names];matrix=[]
  for n in names:
   row=tables[n];positions=[row['state_indices'].index(i)for i in rows[n]['state_indices']];q=np.asarray(row['y'])[:,positions].mean((0,1));matrix.append([float(q[a])for a in selected])
  matches[m]=dict(scenes=names,selected_indices=selected,**matching(matrix))
 transfer={}
 for n in names:
  peer=PAIRED[n]
  if peer not in tables:transfer[n]=dict(peer=peer,available=False);continue
  pq=np.asarray(tables[peer]['y']).mean((0,1));order=np.lexsort((np.arange(49),-pq));top=order[:3];cut=pq[top[-1]]
  selected=rows[n]['methods']['v2']['selected_index'];eta=rows[n]['candidate_etas'][selected]
  transfer[n]=dict(peer=peer,available=True,selected_candidate=dict(key=rows[n]['candidate_keys'][selected],eta=eta),peer_top3=[menu()[int(a)]for a in top],in_peer_canonical_top3=bool(selected<49 and selected in top),in_peer_top3_including_boundary_ties=bool(selected<49 and pq[selected]>=cut),v2_rate=per_scene[n]['rates']['v2'],difference_from_global=per_scene[n]['differences_from_global']['v2'])
 learned=[m for m in ['v2_49','v2_49_no_fallback']if matches[m]['p_one_sided']<.05 and matches[m]['D']>matches['generic_input']['D']];primary=comparisons['v2'];worst=min(per_scene[n]['differences_from_global']['v2']for n in names);phase1=read(HERE/'HETEROGENEITY_CHECK.json')
 decisions=dict(scene_set_established=phase1['established'],learned=bool(learned),learned_methods=learned,useful=bool(primary['difference_from_global']>0 and primary['ci_states'][0]>0),not_harmful=bool(worst>=-.1),worst_scene_difference=float(worst),interpretation_limited=not phase1['established'],all_thresholds_use_unrounded_values=True)
 write(HERE/'STATISTICS.json',dict(utc=utc(),methods=METHODS,method_names=NAMES,scenes=names,comparisons=comparisons,per_scene=per_scene,scene_matching=matches,same_type_transfer=transfer,decisions=decisions,bootstrap=dict(seed=SEED,replicates=B,paired_indices_across_methods=True,own_reference_half_splits=P),generator_selected_scenes=[n for n in names if rows[n]['methods']['v2']['selected_index']>=49],literature_states_not_new=['lit_circle_five','lit_swap8_free','lit_swap8_obstacles']))
 write(HERE/'TEST_DECISION.json',dict(utc=utc(),**decisions,primary=primary,scene_matching=matches));print('FINAL DECISIONS',decisions,flush=True)
if __name__=='__main__':main()
