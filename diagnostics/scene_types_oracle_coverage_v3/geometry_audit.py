"""Zero-step check that the actual production environment loads each frozen map."""
from core import *

def main():
    checks=[]
    for e in read(HERE/'SCENES.json')['entries']:
        s=e['scene'];cfg=s['config']
        env=RN.SwapEnv(s['states'][0],RN.LiteratureConfig(**cfg))
        a=np.asarray(env.walls,dtype=float).reshape(-1,2,2)
        b=np.asarray(s['walls'],dtype=float).reshape(-1,2,2)
        wall_equal=a.shape==b.shape and a.tobytes()==b.tobytes()
        physical_margin=getattr(env.instance.geometry,'margin',None)
        solver_margin=cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin']
        margin_equal=physical_margin is None or abs(physical_margin-solver_margin)<=1e-12
        checks.append(dict(type_id=e['type_id'],level=e['level'],scene=s['name'],
            physical_wall_segments=len(a),frozen_wall_segments=len(b),wall_arrays_identical=wall_equal,
            physical_margin=physical_margin,solver_margin=solver_margin,margin_equal=margin_equal,
            agent_radius=cfg['agent_radius'],dt=cfg['dt'],max_speed=cfg['max_speed'],
            state_embeds_geometry=bool(s['states'][0].get('bottleneck_config')),
            passed=wall_equal and margin_equal))
    passed=all(x['passed']for x in checks)
    dump(HERE/'GEOMETRY_CONSISTENCY.json',dict(passed=passed,scenes=checks,environment_steps=0,
        meaning='Actual frozen SwapEnv walls compared byte-for-byte with SCENES.json; original nominal/safety/dynamics unchanged.'))
    print('ACTUAL GEOMETRY',passed,len(checks),'scenes; zero env.step calls',flush=True)
    assert passed

if __name__=='__main__':main()
