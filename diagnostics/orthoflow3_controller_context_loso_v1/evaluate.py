"""All source-selected models frozen before target scoring or cached Q access."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,csv,time
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import binomtest,spearmanr
from .train import ROOT,OUT,OLD,FOLDS,KINDS,Critic,load,dump,sha,gather,data
TARGET=ROOT/'diagnostics/orthoflow3_loso_root_cause_v1/targets'
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def target_context():
 assert (OUT/'models_frozen.json').exists(),'Do not acquire new target context before source selection freeze'
 from .context import Runtime,cached
 from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
 for fold,scene in FOLDS.items():
  path=OUT/f'target_context_{fold}.npz'
  if path.exists():continue
  manifest=load(TARGET/fold/'manifest.json');rt=Runtime(scene)
  if fold!='ring':physical=load(TARGET/fold/'physical.json')
  else:
   from diagnostics.orthoflow3_cross_scene_zero_shot_v1.evaluate import physical_inputs
   ps,ss,x,flat=physical_inputs();by={p['state_uid']:i for i,p in enumerate(ps)};physical=[]
   for p in manifest:
    uid=p['state_uid'];i=by[uid]
    physical.append(rep.parse({'scenario':scene,'structured_state':{**ss[uid]['physical'],'timestep':0},
     'conditioning':{'flat':flat[i].tolist()},'environment_descriptor':{'obstacle_center':[0.,0.]}}))
  x=rep.batch([rep.entities(p) for p in physical]);frozen=dict(np.load(TARGET/fold/'entities.npz'))
  for k in frozen:np.testing.assert_allclose(np.asarray(x[k]),frozen[k],atol=2e-6,rtol=2e-6)
  out1=[];out3=[];records=[]
  def encode(r):return np.r_[r['context'],1.] if r['valid'] else np.zeros(11)
  for p,s in zip(manifest,physical):
   state={'state_uid':p['state_uid'],'physical':s};a=cached(rt,state,[0.,0.,0.],'C1');out1.append(encode(a));records.append(a)
   for eta in p['eta']:
    a=cached(rt,state,eta,'C3');out3.append(encode(a));records.append(a)
  np.savez_compressed(path,C1=np.asarray(out1,np.float32),C3=np.asarray(out3,np.float32))
  dump(OUT/f'target_context_{fold}.json',{'physical':physical,'records':records,'input_replay_matches_frozen':True,
    'invalid':sum(not r['valid'] for r in records),'labels_used':False})

def normalized(c,norm):
 return np.c_[np.where(c[:,-1,None]>0,(c[:,:-1]-np.asarray(norm['context_center']))/np.asarray(norm['context_scale']),0),c[:,-1]].astype(np.float32)

def predict():
 if (OUT/'target_predictions.json').exists():return
 fr=load(OUT/'models_frozen.json');result={}
 for fold in FOLDS:
  man=load(TARGET/fold/'manifest.json');x=dict(np.load(TARGET/fold/'entities.npz'));n=len(man)
  eta=np.asarray([m['eta'] for m in man],np.float32).reshape(-1,3);ix=np.repeat(np.arange(n),16);raw=np.load(OUT/f'target_context_{fold}.npz')
  scores={};logits={}
  for kind in KINDS:
   norm=load(OUT/fold/kind/'normalization.json');c=normalized(raw['C1'][ix] if kind.startswith('C1') else raw['C3'],norm)
   e=(eta-np.asarray(norm['eta_center'],np.float32))/np.asarray(norm['eta_radius'],np.float32)
   model=Critic(kind.endswith('additive'));init=model.init(jax.random.PRNGKey(0),gather(x,[0]),jnp.zeros((1,3)),jnp.zeros((1,11)))
   for run in fr['folds'][fold][kind]['runs']:
    assert sha(run['checkpoint'])==run['sha256'] and not run['target_labels_used']
    params=serialization.from_bytes(init,open(run['checkpoint'],'rb').read());fn=jax.jit(lambda xx,ee,cc:model.apply(params,xx,ee,cc))
    def get(si,ee,cc):return np.concatenate([np.asarray(fn(gather(x,si[j:j+128]),jnp.asarray(ee[j:j+128]),jnp.asarray(cc[j:j+128]))) for j in range(0,len(ix),128)]).reshape(n,16)
    z=get(ix,e,c);key=f'{kind}_seed{run["seed"]}';logits[key]=z.tolist();scores[key]=expit(z).tolist()
    if run['seed']==fr['folds'][fold][kind]['selected']['seed']:
     # Fixed intervention RNG. State and context shuffled independently within scene.
     perm=np.random.default_rng(2026100320).permutation(n);per=np.repeat(perm,16)*16+np.tile(np.arange(16),n)
     logits[kind]=z.tolist();scores[kind]=expit(z).tolist()
     for label,si,ee,cc in [('context_shuffle',ix,e,c[per]),('state_shuffle',np.repeat(perm,16),e,c),
         ('eta_shuffle',ix,np.roll(e.reshape(n,16,3),1,axis=1).reshape(-1,3),c)]:
      zz=get(si,ee,cc);logits[kind+'_'+label]=zz.tolist();scores[kind+'_'+label]=expit(zz).tolist()
     # C3 c(h,eta) contains eta interaction itself: test an aligned eta+response permutation separately.
     if kind=='C3_full':
      ee=np.roll(e.reshape(n,16,3),1,axis=1).reshape(-1,3);cc=np.roll(c.reshape(n,16,11),1,axis=1).reshape(-1,11)
      zz=get(ix,ee,cc);logits[kind+'_eta_response_shuffle']=zz.tolist();scores[kind+'_eta_response_shuffle']=expit(zz).tolist()
   scores[kind+'_ensemble']=np.mean([scores[f'{kind}_seed{s}'] for s in (17,23,41)],axis=0).tolist()
  # Predeclared numerical diagnostic: sigmoid can round several distinct logits
  # to exactly 1 in float32. Primary still matches the historical probability
  # selector; logit-safe ranking is reported separately, never selected on TEST.
  for kind in KINDS:scores[kind+'_logit_rank']=logits[kind]
  chosen=fr['folds'][fold]['source_selected_kind'];scores['source_selected']=scores[chosen]
  result[fold]={'state_uids':[m['state_uid'] for m in man],'eta':eta.reshape(n,16,3).tolist(),'scores':scores,'logits':logits,'chosen':chosen}
 dump(OUT/'target_predictions.json',{'models_sha':sha(OUT/'models_frozen.json'),'target_labels_opened_for_new_predictions':False,'folds':result})

def evaluate():
 pred=load(OUT/'target_predictions.json');truth=load(OLD/'cached_truth.json');previous=load(OLD/'target_predictions.json')
 results=[];paired=[];details=[]
 for fold,pp in pred['folds'].items():
  tt=truth[fold];assert pp['state_uids']==[t['state_uid'] for t in tt]
  lo=np.array([t['lower'] for t in tt]);hi=np.array([t['upper'] for t in tt]);r=np.array([[v is True for v in t['robust']] for t in tt]);unk=np.array([[v is None for v in t['robust']] for t in tt]);n=len(r);avail=r.any(1)
  scores=dict(pp['scores']);prior=previous['folds'][fold]['scores']
  scores.update(C0=prior['partial_count_shared'],eta_only=prior['partial_count_eta_only'])
  scores['oracle']=lo.tolist();chosen={}
  for method,zz in scores.items():
   z=np.asarray(zz);sel=z.argmax(1);ii=np.arange(n);yes=r[ii,sel];unknown=unk[ii,sel];chosen[method]=(yes,unknown)
   p=expit(z[ii,sel]) if method.endswith('_logit_rank') else z[ii,sel];ql=lo[ii,sel];qh=hi[ii,sel]
   results.append({'fold':fold,'method':method,'N':n,'B15':int(yes.sum()),'unknown':int(unknown.sum()),'oracle_B15':int(avail.sum()),
    'oracle_gap_lower':int(avail.sum()-yes.sum()-unknown.sum()),'oracle_gap_upper':int(avail.sum()-yes.sum()),
    'selection_given_available_lower':float(yes[avail].mean()),'Q_lower':float(ql.mean()),'Q_upper':float(qh.mean()),
    'predicted_selected':float(p.mean()),'severe_fp_confirmed':int(((p>.9)&(qh<=.5)).sum()),
    'top_score_tied_states':int(((z==z.max(1)[:,None]).sum(1)>1).sum()),
    'overestimate_lower':float((p-qh).mean()),'overestimate_upper':float((p-ql).mean()),
    'p95_count':int((p>.95).sum()),'p95_B15_precision_lower':float(yes[p>.95].mean()) if (p>.95).any() else None,
    'regret_lower':float(np.maximum(0,lo.max(1)-qh).mean()),'regret_upper':float(np.maximum(0,hi.max(1)-ql).mean())})
   for i,j in enumerate(sel):details.append({'fold':fold,'method':method,'state_uid':tt[i]['state_uid'],'proposal':int(j),'p':float(p[i]),'Q_lower':float(ql[i]),'Q_upper':float(qh[i]),'B15':bool(yes[i]),'unknown':bool(unknown[i])})
  for main in ['C1_full','C3_full','source_selected']:
   for ctrl in ['C0','eta_only','C1_additive',main+'_context_shuffle' if main!='source_selected' else pp['chosen']+'_context_shuffle']:
    a,u=chosen[main];b,v=chosen[ctrl];valid=~(u|v);res=int((a&~b&valid).sum());br=int((b&~a&valid).sum())
    paired.append({'fold':fold,'main':main,'control':ctrl,'rescue':res,'break':br,'resolved':int(valid.sum()),'p_two_sided':float(binomtest(res,res+br).pvalue) if res+br else 1.})
 csvout('loso_results.csv',results);csvout('paired_comparisons.csv',paired);csvout('selected_proposals.csv',details)
 dump(OUT/'working_state.json',{'stage':'loso_evaluated','new_full_rollouts':0,'remaining':['source shuffle','swap attribution','latency','report']})

def source_audit():
 frozen=load(OUT/'models_frozen.json');out=[];rng=np.random.default_rng(2026100321)
 for fold in FOLDS:
  for kind in KINDS:
   rows,x,e,c,si,s,f,w,groups=data(fold,kind);model=Critic(kind.endswith('additive'))
   run=frozen['folds'][fold][kind]['selected'];params=serialization.from_bytes(model.init(jax.random.PRNGKey(0),gather(x,[0]),jnp.zeros((1,3)),jnp.zeros((1,11))),open(run['checkpoint'],'rb').read())
   fn=jax.jit(lambda xx,ee,cc:np_dummy(model,params,xx,ee,cc))
   for sc,g in groups.items():
    ix=g['validation'];ids=np.unique(si[ix]);mapping=dict(zip(ids,rng.permutation(ids)));permstate=np.array([mapping[i] for i in si[ix]])
    # Context shuffle permutes state-level blocks for C1; C3 uses candidate-wise permutation
    # (eta-conditioned response mismatch, explicitly not a pure controller swap).
    pc=c[ix][rng.permutation(len(ix))] if kind=='C3_full' else np.array([c[np.flatnonzero(si==i)[0]] for i in permstate])
    for label,ss,ee,cc in [('correct',si[ix],e[ix],c[ix]),('context_shuffle',si[ix],e[ix],pc),('state_shuffle',permstate,e[ix],c[ix]),('eta_shuffle',si[ix],e[ix][rng.permutation(len(ix))],c[ix])]:
     z=np.concatenate([np.asarray(fn(gather(x,ss[j:j+256]),jnp.asarray(ee[j:j+256]),jnp.asarray(cc[j:j+256]))) for j in range(0,len(ix),256)])
     p=expit(z);ll=float((s[ix]*np.logaddexp(0,-z)+f[ix]*np.logaddexp(0,z)).sum()/(s[ix]+f[ix]).sum())
     cert=np.array([rows[i]['b15_confirmed'] for i in ix]);nob=np.array([rows[i]['non_b15_confirmed'] for i in ix]);selected=[]
     for st in ids:
      jj=np.flatnonzero(si[ix]==st)
      if cert[jj].any():selected.append(bool(cert[jj[np.argmax(p[jj])]]))
     out.append({'fold':fold,'kind':kind,'scene':sc,'intervention':label,'NLL':ll,'MAE_observed_rate':float(abs(p-s[ix]/(s[ix]+f[ix])).mean()),
      'B15_available_states':len(selected),'B15_selected_confirmed':sum(selected),'severe_fp_Q16_confirmed':int(sum((p>.9)&np.array([rows[i]['strong_failure_Q16_upper_le_half'] for i in ix])))})
 csvout('source_heldout_diagnostics.csv',out)

def np_dummy(model,params,xx,ee,cc):return model.apply(params,xx,ee,cc)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['target_context','predict','evaluate','source_audit']);a=ap.parse_args();globals()[a.action]()
