import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.event_history import event_margin
jax.config.update('jax_enable_x64',True)


class EventHistoryTests(unittest.TestCase):
    def test_later_recovery_cannot_erase_earlier_deadlock(self):
        u=jnp.array([1.,1.,1.,-1.,-1.,-1.])
        result=event_margin(u,jnp.ones(6),jnp.ones(6),jnp.ones(6,bool),1.,hold_samples=3)
        self.assertGreater(float(result['robustness']),0.)
        self.assertLess(float(result['stalled_margin']),0.)

    def test_later_stagnation_after_success_is_not_a_deadlock(self):
        u=jnp.array([1.,-1.,1.,1.,1.,1.])
        result=event_margin(u,jnp.ones(6),jnp.ones(6),jnp.ones(6,bool),1.,hold_samples=3)
        self.assertLess(float(result['robustness']),0.)

    def test_event_equivalence_with_first_event_oracle(self):
        rng=np.random.default_rng(2026091865)
        fn=jax.jit(lambda u,p,v,ready,s:event_margin(u,p,v,ready,s,hold_samples=3)['robustness'])
        for _ in range(100):
            u,p,v=rng.uniform(-1,1,(3,12));ready=np.arange(12)>=2;s=float(rng.uniform(-1,1))
            first=None;consecutive=0
            for i in range(12):
                if u[i]<=0:first='success';break
                consecutive=consecutive+1 if ready[i] and p[i]>0 and v[i]>0 else 0
                if consecutive>=3:first='deadlock';break
            expected=first=='deadlock' or (first is None and s>0)
            self.assertEqual(bool(fn(u,p,v,ready,s)>0),expected)

    def test_no_jump_when_first_deadlock_witness_changes(self):
        # Fixed full trajectory reductions stay continuous across witness ties.
        u=jnp.ones(6);v=jnp.ones(6);ready=jnp.ones(6,bool)
        f=lambda x:event_margin(u,jnp.array([x,.2,.2,.2,.2,.2]),v,ready,-1.,hold_samples=3)['robustness']
        self.assertLess(abs(float(f(-1e-7)-f(1e-7))),1e-6)
        self.assertTrue(np.isfinite(float(jax.grad(f)(.1))))


if __name__=='__main__':unittest.main()
