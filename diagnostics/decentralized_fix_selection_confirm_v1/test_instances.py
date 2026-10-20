from support import *
G=guard()
def main():
 prereg();assert (HERE/'ALL_CONTROLLERS_FROZEN.receipt.json').exists();p=read(HERE/'CONTROLLERS.json');s=read(HERE/'PROBE_STATES.json')[0];rows=[];cache={}
 for n in p['controller_order']:
  rt=runtime(n,p,cache);env=rt.make_env(s);key=key_for(s,0,0);guard();a=rt.nominal(env,key)[1];b=rt.nominal(env,key)[1];checks=dict(shape=np.asarray(a).shape==(4,2),finite=bool(np.isfinite(a).all()),speed_bound=bool((np.linalg.norm(a,axis=-1)<=env.config.max_speed+1e-12).all()),same_key_bitwise=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),x64_false=not jax.config.x64_enabled);rows.append(dict(controller=n,checks=checks,passed=all(checks.values())));print(n,checks,flush=True)
 write(HERE/'UNIT_TESTS.json',dict(passed=all(r['passed'] for r in rows),controllers=len(rows),tests=rows,simulation_steps=0,task_outcomes_read=False,guard=guard()));source_inventory('imports_unit_tests.json');assert all(r['passed'] for r in rows)
if __name__=='__main__':main()
