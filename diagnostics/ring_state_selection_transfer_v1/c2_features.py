"""Reusable source/held-controller static descriptors without any outcome reads."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import OUT,read,write,sha
from diagnostics.ring_state_selection_transfer_v1.c1.runtime import make

def run():
 import jax
 from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
 from diagnostics.controller_q_diversity_v1.features import field_context
 from diagnostics.controller_q_diversity_v1.train import compact_field
 p=read(OUT/'c1/protocol.json');names=list(p['kappa'])[:5];old=np.load(OUT/'EFFECTIVE_BASIS_QUERIES.npz');base=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
 def forbidden(*a,**kw):raise RuntimeError('No physical step permitted')
 base.env_cls.step=forbidden
 dims=256;features={k:np.zeros((5,32,dims),np.float32) for k in ['R0','R1','R2','R3','R4','R4_action','RB']};fail=[];reuse=0;start=time.time();queries=0
 for si,s in enumerate(p['states']):
  pos=np.array(s['physical']['positions']);goal=np.array(s['physical']['goals']);e=goal-pos;e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12);lat=np.stack([-e[:,1],e[:,0]],-1)
  def local(a):
   a=np.asarray(a).reshape(4,2,-1);return np.stack([(a*e[:,:,None]).sum(1),(a*lat[:,:,None]).sum(1)],axis=1).ravel()/.52
  for ci,name in enumerate(names):
   if ci in [0,2,3]:features['R1'][ci,si,[0,2,3].index(ci)]=1
   features['R2'][ci,si,0]=p['kappa'][name];rt=make(base,name);env=rt.make_env(s);key=jax.random.PRNGKey(2026101121)
   try:
    tag=f's{si}_{name}_compact64'
    if tag in old:desc=old[tag];reuse+=1
    else:
     class Query:
      def __getattr__(self,key):return getattr(rt,key)
     desc=np.asarray(compact_field(np.concatenate([controller_context(rt,s),field_context(Query(),s)],-1)))
    features['R3'][ci,si]=desc.ravel()
    js=[]
    for anchor,eta in [('zero',np.zeros(3)),('eta13',np.array(p['eta'][13]))]:
     tag=f's{si}_{name}_{anchor}_basis'
     if tag in old:J=old[tag];a0=old[f's{si}_{name}_{anchor}_action'];reuse+=1
     else:
      def action(v):
       nonlocal queries
       queries+=1;return rt.action(env,key,v,'field')[0].ravel()
      a0=action(eta);J=np.stack([(action(eta+np.eye(3)[j]*.001)-action(eta-np.eye(3)[j]*.001))/.002 for j in range(3)],axis=1)
     js.append(local(J))
     if anchor=='zero':zero=a0
    jb=np.concatenate(js);features['R4'][ci,si,:48]=jb;features['R4_action'][ci,si,:56]=np.r_[jb,local(zero)]
    tag=f's{si}_{name}_zero_RB'
    if tag in old:RB=old[tag];reuse+=1
    else:
     _,_,_,safe,_,_=rt.prepare(env,key,np.zeros(3));B=np.stack([rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).correction(np.eye(3)[j]).ravel() for j in range(3)],axis=1);RB=rt.unprojected_response()@B
    features['RB'][ci,si,:24]=local(RB)
   except Exception as ex:
    for k in ['R3','R4','R4_action','RB']:features[k][ci,si,255]=1
    fail.append(dict(state=si,controller=name,type=type(ex).__name__,message=str(ex)))
  if si%8==7:print(si+1,'states',time.time()-start,flush=True)
 np.savez_compressed(OUT/'C2_STATIC_FEATURES.npz',**features,controller_names=names)
 write(OUT/'C2_STATIC_FEATURES_AUDIT.json',dict(physical_steps=0,new_rollouts=0,outcomes_read=False,states=32,controllers=5,reused_static_arrays=reuse,new_FF_action_queries=queries,errors=fail,seconds=time.time()-start,code_sha256=sha(__file__),file_sha256=sha(OUT/'C2_STATIC_FEATURES.npz')))
if __name__=='__main__':run()
