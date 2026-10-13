"""Non-outcome contract tests, deterministic metric checks, prefix audit."""
from pathlib import Path
import json,datetime
import numpy as np
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def main():
 from . import metrics
 from .pilot import components
 from .factor_runtime import config,state,Env,Base,Field
 import jax
 jax.config.update('jax_enable_x64',True)
 y=np.zeros((2,8,2,4,4))
 for s in range(8):y[:,s,:,s%4,:]=1
 r,_=metrics.evaluate(y,y.mean(-1),np.ones((2,4))*.25)
 assert r['success']==1 and r['regret']==0 and r['state_eta']['unbiased_skill']==1 and r['matched_minus_mismatched']>0 and r['global_fixed']==.25
 x=np.random.default_rng(1).normal(size=(3,7,2,4));assert np.allclose(sum(components(x).values()),x-x.mean())
 cases=[]
 for top in ['single','parallel']:
  for n in [2,4]:
   cfg=config(.65,n,top);physical=state(0,n);env=Env(cfg,physical);base=Base(cfg)
   nom=[]
   for c in ['isotropic','anisotropic']:
    rt=Field(base,c);key=jax.random.PRNGKey(5);out=rt.action(env,key,[.5,.1,.01],'field');assert np.isfinite(out[0]).all();nom.append(rt.nominal(env,key)[1]);assert out[1]['max_violation']<1e-8
    cases.append({'topology':top,'N':n,'controller':c,'projection_certificate':out[1]})
   np.testing.assert_array_equal(*nom);assert state(0,4)['positions'][:2]==state(0,2)['positions'] and state(0,4)['goals'][:2]==state(0,2)['goals']
 # H400 native execution versus corresponding complete H850 source.
 p=HERE/'prefix_audit';src=HERE/'broad_pilot';old=json.loads((src/'cases.json').read_text());matches=[]
 for c in json.loads((p/'cases.json').read_text()):
  a=json.loads((p/'raw'/(c['rollout_uid']+'.json')).read_text());c0=next(v for v in old if all(v[k]==c[k] for k in ['width','state_index','controller_name','eta_index','seed']));b=json.loads((src/'raw'/(c0['rollout_uid']+'.json')).read_text())
  assert a['success']==int(b['success'] and b['summary']['first_success_step']<=400);np.testing.assert_array_equal(a['first_action'],b['first_action']);assert a['episode_length']==min(b['episode_length'],400)
  if b['episode_length']<=400:assert a['summary']==b['summary'] and a['outcome']==b['outcome']
  else:assert a['outcome']=='timeout'
  matches.append({'H400_record':c['rollout_uid'],'H850_record':c0['rollout_uid'],'equal':True})
 write(HERE/'contract_tests.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metric_oracle_and_ANOVA_passed':True,'factor_static_action_cases':cases,'nested_N_tasks_verified':True,'H400_prefix_reproduction':matches,'simulated_factor_episodes':0})
 print('PASS: metrics, N nesting, endpoint match, safety certificates, four H400 native prefix reproductions')
if __name__=='__main__':main()
