"""Analysis only: archived labels and code; never import simulation or DB writers."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,hashlib,itertools,datetime,math
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'diagnostics/critic_pool_supplement_v1';V=R/'diagnostics/hetero_scene_pipeline_v1'
inputs={}
def read(p):
 p=Path(p);data=p.read_bytes();inputs[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
def write(n,d):(H/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
M=read(P/'ETA_MENU.json');K=M['keys'];T=read(V/'LABEL_TABLES.json');E=read(V/'EVALUATION_STATES.json')['scenes'];U=read(P/'LABEL_TABLES.json');Q={};YS={};metadata={}
for n,t in dict(T,**U).items():
 indices=E.get(n,t['state_indices']);ix=[t['state_indices'].index(i)for i in indices];y=np.asarray(t['y'],float)[:,ix];YS[n]=y;Q[n]=y.mean((0,1));order=np.lexsort((np.arange(49),-Q[n]));metadata[n]=dict(states=len(ix),state_indices=indices,source=str(V/'LABEL_TABLES.json'if n in T else P/'LABEL_TABLES.json'),best_candidates=[dict(key=K[i],eta=M['candidates'][i]['eta_a'],rate=float(Q[n][i]))for i in order[:5]],rates={k:float(Q[n][i])for i,k in enumerate(K)},within5pp=[K[i]for i in np.flatnonzero(Q[n]>=Q[n].max()-.05-1e-12)],controller_rates=y.mean(1).tolist())
anchors=['lit_swap8_obstacles','xsc_tr_swap_n09','xsc_tr_plus_w016','lit_circle_five']
q=np.stack([Q[n]for n in anchors]);L=math.lcm(*[3*YS[n].shape[1]for n in anchors]);full_counts=np.stack([YS[n].sum((0,1)).astype(np.int64)*(L//(3*YS[n].shape[1]))for n in anchors]);full_scores=full_counts.sum(0);global_i=int(full_scores.argmax());own=q.max(1);gap=float((full_counts.max(1).sum()-full_scores.max())/(4*L));cross={n:{m:float(Q[n].max()-Q[n][int(Q[m].argmax())])for m in anchors}for n in anchors}
# Cross-state description with scene-equal weights and differing state counts.
rng=np.random.default_rng(2026102721);Psplit=2000;tr=[];te=[]
LH=math.lcm(*[3*(YS[n].shape[1]//2)for n in anchors])
for n in anchors:
 y=YS[n].sum(0).astype(np.int64);S=len(y);ix=np.argsort(rng.random((Psplit,S)),axis=1,kind='stable');a=y[ix[:,:S//2]].sum(1)*(LH//(3*(S//2)));b=y[ix[:,S//2:]].sum(1)*(LH//(3*(S//2)));tr.append(a);te.append(b)
tr=np.stack(tr);te=np.stack(te)/LH;best=tr.argmax(-1);own_half=np.take_along_axis(te,best[...,None],-1)[...,0];gc=tr.sum(0,dtype=np.int64).argmax(-1);fixed=te[:,np.arange(Psplit),gc]
# Paired two-way losses across exactly the same target scoring half.
cross_half={n:{m:float(np.mean(own_half[i]-te[i,np.arange(Psplit),best[j]]))for j,m in enumerate(anchors)}for i,n in enumerate(anchors)}
# Near-optimal-set intersection diagnoses whether winner changes conceal a common compromise.
intersections={str(tol):[K[i]for i in np.flatnonzero(np.all(q>=q.max(1)[:,None]-tol-1e-12,axis=0))]for tol in [.05,.1]}
counts=read(P/'COLLISION_CLASSIFICATION.json')['records'];collision_summary={}
for n,t in U.items():
 if not n.startswith('xsc_tr_bigbox'):continue
 ids=t['rollout_uids'];summ={k:sum(int(counts[i][k])for i in ids)for k in ['agent','static','moving']};summ['episodes']=len(ids);summ['any_collision']=sum(any(counts[i].values())for i in ids);summ['overall_success_rate']=float(np.asarray(t['y']).mean());summ['best49_rate']=float(np.asarray(t['y']).mean((0,1)).max());collision_summary[n]=summ
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_only=True,new_rollouts=0,new_training=0,anchor_scenes=anchors,anchor_source_state_counts=[YS[n].shape[1]for n in anchors],candidate_menu_size=49,anchor_rates={n:metadata[n]for n in anchors},full_sample=dict(own_rate=float(own.mean()),best_global_rate=float(q.mean(0).max()),gap=gap,best_global=M['candidates'][global_i],all_tied_best_global=[M['candidates'][int(i)]for i in np.flatnonzero(full_scores==full_scores.max())],own_rates=own.tolist(),global_per_scene=q[:,global_i].tolist(),all49_global_rates=q.mean(0).tolist()),half_split_descriptive=dict(splits=Psplit,seed=2026102721,own_rate=float(own_half.mean()),global_rate=float(fixed.mean()),gap=float((own_half-fixed).mean()),per_scene={n:dict(own=float(own_half[i].mean()),global_rate=float(fixed[i].mean()),gap=float((own_half[i]-fixed[i]).mean()))for i,n in enumerate(anchors)},global_selection_counts=np.bincount(gc,minlength=49).tolist(),scope='Post-selected historical anchors, descriptive only, not new independent confirmation'),cross_use_losses_full=cross,cross_use_losses_half=cross_half,near_optimal_intersection=intersections,all_scene_profiles=metadata,bigbox_collision_summary=collision_summary,lateral_gain={a['key']:1+3.303687238760696*a['eta_a'][1]for a in M['candidates']},source_sha256=inputs)
write('EXISTING_PATTERN_ANALYSIS.json',out)
print(json.dumps({k:out[k]for k in ['full_sample','half_split_descriptive','cross_use_losses_half','near_optimal_intersection','bigbox_collision_summary']},indent=2,ensure_ascii=False))
