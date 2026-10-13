"""Cheap endpoint-matched controllers with different FF response operators.

Every family returns the same unprojected nominal action at any physical state.
Only the path/vector field differs. Existing FF projection, eta lifting, physical
dynamics, horizon and success evaluator remain unchanged.
"""
import numpy as np
from diagnostics.controller_q_diversity_v1.numpy_backend import NumpyAnalyticField

FAMILIES = ('isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative', 'coupled')
REFERENCE = [.85, .5, 0., .25, .2, 2.5]


class EndpointMatchedField(NumpyAnalyticField):
    def __init__(self, base, family):
        assert family in FAMILIES
        super().__init__(base, REFERENCE)
        self.family = family

    def matrix(self, env):
        n = len(env.positions);a = 2.5*np.eye(2*n)
        if self.family == 'anisotropic':
            d = env.goals-env.positions
            norm=np.linalg.norm(d,axis=1,keepdims=True)
            d /= np.maximum(norm,1e-12)
            d[norm[:,0]<1e-12]=[1.,0.]
            for i,e in enumerate(d):
                r = np.stack((e,[-e[1],e[0]]),axis=1)
                a[2*i:2*i+2,2*i:2*i+2] = r@np.diag([5.,.8])@r.T
        elif self.family in ('rotate_positive','rotate_negative'):
            sign = 1. if self.family == 'rotate_positive' else -1.
            a += np.kron(np.eye(n),sign*np.array([[0.,-2.5],[2.5,0.]]))
        elif self.family == 'coupled':
            d = env.positions[:,None]-env.positions[None,:]
            weights = np.exp(-np.sum(d*d,axis=-1)/.8**2)*(1-np.eye(n))
            degree = weights.sum(1)
            laplacian = (np.diag(degree)-weights)/max(float(degree.max()),1e-12)
            a += 1.5*np.kron(laplacian,np.eye(2))
        assert np.linalg.eigvalsh((a+a.T)/2).min() > 0
        assert np.linalg.norm(np.eye(2*n)-a/self.steps,2) < 1.
        return a

    def nominal(self, env, key, traced=False):
        import jax
        self._env = env;reference = self.target(env)
        z0 = np.asarray(jax.random.normal(key,(1,self.agent.config['joint_act_dim']))).reshape(-1,2)
        h0 = self.to_world(self.mean+self.scale*z0,env)
        self._matrix = self.matrix(env)
        transition = np.eye(h0.size)-self._matrix/self.steps
        power = np.linalg.matrix_power(transition,self.steps)
        self._center = np.linalg.solve(np.eye(h0.size)-power,reference.ravel()-power@h0.ravel())
        h = h0.ravel().copy();path = [h.reshape(h0.shape).copy()]
        for _ in range(self.steps):
            h += self._matrix@(self._center-h)/self.steps
            path.append(h.reshape(h0.shape).copy())
        np.testing.assert_allclose(h.reshape(h0.shape),reference,atol=2e-12,rtol=2e-12)
        # Returning the shared reference makes nominal fingerprints identical
        # by construction; numerical path error remains checked above.
        return self.observation(env),reference.copy(),np.asarray(path) if traced else None

    def velocity_fn(self, obs, z, time):
        h = self.to_world(self.mean+self.scale*np.asarray(z),self._env)
        w = (self._matrix@(self._center-h.ravel())).reshape(h.shape)
        return (self.to_local(w,self._env)/self.scale).ravel()

    def unprojected_response(self):
        transition = np.eye(len(self._center))-self._matrix/self.steps
        return np.linalg.solve(self._matrix,np.eye(len(self._center))-
                               np.linalg.matrix_power(transition,self.steps))
