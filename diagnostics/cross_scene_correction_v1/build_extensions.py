"""Static, auditable adapters for parameterized new scene walls (no monkeypatches)."""
from pathlib import Path
import json
import hashlib
import difflib

H = Path(__file__).resolve().parent


def main():
    records = []
    def derive(source, target, code):
        before = source.read_text()
        target.write_text(code)
        target.with_suffix('.py.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), code.splitlines(True),
            fromfile=str(source), tofile=str(target))))
        records.append(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            target=str(target), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    src = H / 'backend/controllers_layout_user_fix_20261023.py'
    code = src.read_text()
    at = "    if scene.startswith('lit_swap8_'):"
    branch = '''    if scene.startswith('xsc_'):
        walls=xp.asarray(env.walls)
        if not len(walls):return zero,zero
        a=walls[:,0];ab=walls[:,1]-walls[:,0];d=p[...,None,:]-a
        t=xp.clip(xp.sum(d*ab,axis=-1)/xp.sum(ab*ab,axis=-1),0,1)
        delta=d-t[...,None]*ab
        distance=xp.sqrt(xp.maximum(xp.sum(delta*delta,axis=-1),1e-30))
        return xp.maximum(margin-xp.min(distance,axis=-1),0)**2,zero
'''
    assert code.count(at) == 1
    code = code.replace(at, branch + at)
    code = code.replace('def optimizer(horizon,iterations,rate,scene,dt,radius,margin,speed):',
                        'def optimizer(horizon,iterations,rate,scene,dt,radius,margin,speed,wall_key=()):')
    code = code.replace('wall_radius=0.))', 'wall_radius=0.),walls=np.asarray(wall_key,float).reshape(-1,2,2))')
    old = "c.scene,c.dt,c.agent_radius,c.wall_collision_margin,c.max_speed)("
    new = "c.scene,c.dt,c.agent_radius,c.wall_collision_margin+c.wall_radius,c.max_speed,tuple(tuple(v) for v in np.asarray(env.walls).reshape(-1,4)))("
    assert old in code
    code = code.replace(old, new)
    derive(src, H / 'controllers_new.py', code)
    src = H / 'backend/swap_runtime_user_fix_20261023.py'
    code = src.read_text().replace('from swap_geometry_user_fix_20261023 import SwapEnvRevision as SwapEnv,LiteratureConfig',
        'from scene_environment import NewEnv as SwapEnv\nfrom legacy_states import LiteratureConfig')
    code = code.replace('from nominal_geometry_user_fix_20261023 import Nominal', 'from scene_environment import NewNominal as Nominal')
    derive(src, H / 'runtime_new.py', code)
    (H / 'EXTENSION_BUILD.json').write_text(json.dumps(dict(files=records,
        modifications='New files only: environment/nominal factory, wall accessor and immutable optimizer geometry key; step, solver, correction, RNG and stopping unchanged.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
