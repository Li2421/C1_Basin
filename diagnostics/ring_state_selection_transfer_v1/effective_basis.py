"""Static mechanism validity audit; no simulator stepping and no outcome inputs."""
import os,sys,time
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
 p=read(OUT/'c1/protocol.json');base=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
 def forbidden(*a,**k):raise RuntimeError('Static mechanism audit must not step environment')
 base.env_cls.step=forbidden
 names=list(p['kappa']);rows=[];arrays={};t=time.time();fail=[]
 for i,s in enumerate(p['states'][:8]):
  for name in names:
   rt=base if name=='flow' else make(base,name);env=rt.make_env(s);key=jax.random.PRNGKey(2026101121)
   try:
    # Current project static descriptor, only first two states suffice for identifiability audit.
    if i<2:
     class Query:
      def __getattr__(self,key):return getattr(rt,key)
     nom=controller_context(rt,s);field=field_context(Query(),s);desc=np.asarray(compact_field(np.concatenate([nom,field],-1)));arrays[f's{i}_{name}_compact64']=desc;arrays[f's{i}_{name}_nominal14']=nom
    for anchor,eta in [('zero',np.zeros(3)),('eta13',np.array(p['eta'][13]))]:
     def action(eta):return rt.action(env,key,eta,'field')[0].ravel()
     a0=action(eta);der=[]
     for eps in [.001,.01]:
      J=np.stack([(action(eta+np.eye(3)[j]*eps)-action(eta-np.eye(3)[j]*eps))/(2*eps) for j in range(3)],axis=1);der.append(J)
     J=der[0];norm=max(np.linalg.norm(J),1e-12);errors={}
     for h in [.01,.1,.5]:
      v=np.array([1.,-1.,1.])/np.sqrt(3)*h;target=action(eta+v);errors[str(h)]=dict(relative=float(np.linalg.norm(target-a0-J@v)/max(np.linalg.norm(target-a0),1e-12)),absolute=float(np.linalg.norm(target-a0-J@v)))
     rec=dict(state_index=i,controller=name,anchor=anchor,basis_norm=float(np.linalg.norm(J)),basis_rank=int(np.linalg.matrix_rank(J,tol=1e-4)),step_size_relative_difference=float(np.linalg.norm(der[1]-J)/norm),finite_displacement_error=errors)
     arrays[f's{i}_{name}_{anchor}_basis']=J;arrays[f's{i}_{name}_{anchor}_action']=a0
     if name!='flow':
      obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta);B=np.stack([rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).correction(np.eye(3)[j]).ravel() for j in range(3)],axis=1);R=rt.unprojected_response();RB=R@B
      a,_=rt.integrate(env,obs,key,delta,safety,project=False);z,_=rt.integrate(env,obs,key,np.zeros_like(delta),safety,project=False)
      rec.update(unprojected_affine_error=float(np.linalg.norm(a.ravel()-z.ravel()-RB@eta)),projected_derivative_vs_RB_relative=float(np.linalg.norm(J-RB)/max(np.linalg.norm(RB),1e-12)))
      arrays[f's{i}_{name}_{anchor}_RB']=RB
     rows.append(rec)
   except Exception as ex:fail.append(dict(state_index=i,controller=name,type=type(ex).__name__,message=str(ex)))
  print('state',i,'seconds',time.time()-t,flush=True)
 alias=[]
 for i in range(2):
  nom=[arrays[f's{i}_{n}_nominal14'] for n in names[:-1]];desc=[arrays[f's{i}_{n}_compact64'] for n in names[:-1]]
  alias.append(dict(state_index=i,nominal_max_difference=float(np.max(np.abs(np.array(nom)-nom[0]))),compact64_max_difference=float(np.max(np.abs(np.array(desc)-desc[0])))))
 np.savez_compressed(OUT/'EFFECTIVE_BASIS_QUERIES.npz',**arrays);write(OUT/'EFFECTIVE_BASIS_AUDIT.json',dict(status='COMPLETED STATIC VALIDITY AUDIT',rows=rows,errors=fail,descriptor_alias=alias,physical_steps=0,rollouts=0,labels_used=False,root_key=2026101121,seconds=time.time()-t,code_sha256=sha(__file__),interpretation='Exact only for unprojected affine integration up to float roundoff; projected finite differences local in eta/state/key, not complete closedloop representation.'))
if __name__=='__main__':run()
