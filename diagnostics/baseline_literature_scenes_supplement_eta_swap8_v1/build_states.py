import bootstrap
from experiment import *
from audit_supplement import evidence,V2
from swap_geometry import *
from scipy.ndimage import label

def reachable(s):
    geom=Geometry(config(s['scene']));p=np.asarray(s['positions']);g=np.asarray(s['goals']);ii,jj=np.triu_indices(8,1)
    xx=np.arange(-2,2.0001,.025);gx,gy=np.meshgrid(xx,xx);grid=geom.valid_points(np.stack([gx,gy],-1));labels,_=label(grid,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    def components(point):
        ix=np.argmin(abs(xx-point[0]));iy=np.argmin(abs(xx-point[1]));out=set()
        for y in range(max(0,iy-1),min(len(xx),iy+2)):
            for x in range(max(0,ix-1),min(len(xx),ix+2)):
                if labels[y,x] and geom.visible(point,np.array([xx[x],xx[y]])):out.add(int(labels[y,x]))
        return out
    paths=[bool(components(a)&components(b)) for a,b in zip(p,g)]
    straight=[geom.visible(a,b) for a,b in zip(p,g)]
    row=dict(index=s['index'],valid_initial=bool(geom.valid_points(p).all() and (np.linalg.norm(p[ii]-p[jj],axis=-1)>.1).all()),valid_goals=bool(geom.valid_points(g).all() and (np.linalg.norm(g[ii]-g[jj],axis=-1)>.1).all()),single_robot_paths=paths,straight_paths=straight,grid_step=.025,capacity_lower_bound=int(geom.valid_points(np.array([[2*x,2*y] for x in (-1,0,1) for y in (-1,0,1)])).sum()),joint_solvable='not_certified')
    row['eligible']=row['valid_initial'] and row['valid_goals'] and all(paths) and row['capacity_lower_bound']>=8
    if s['scene']=='lit_swap8_moving':
        points,pieces=moving_path(s,config(s['scene']).max_steps);row['moving_path_sha256']=uid('moving',dict(points=points,pieces=pieces));row['moving_initial_min_center_distance']=float(np.linalg.norm(p-points[0],axis=-1).min());row['single_robot_paths_note']='Static geometric reachability only; moving temporal feasibility tested by per-actor solo capability gate'
        write(HERE/'MOVING_PATHS'/f"{s['index']:02d}.json",dict(points=points,pieces=pieces,identity=row['moving_path_sha256']))
    return row

def main():
    assert not (HERE/'SCENES.receipt.json').exists()
    free=None
    for scene,seed,r in [('lit_swap8_free',2026102311,1.),('lit_swap8_obstacles',2026102312,1.4),('lit_swap8_moving',2026102313,1.)]:
        if scene=='lit_swap8_moving':states=[dict(s,scene=scene) for s in free]
        else:
            theta=np.arange(8)*2*np.pi/8;p=r*np.stack([np.cos(theta),np.sin(theta)],-1);g=-p;rng=np.random.default_rng(seed);states=[]
            for index in range(32):
                while True:
                    q=p.copy() if index==0 else p+rng.uniform(-.1,.1,p.shape)
                    ii,jj=np.triu_indices(8,1)
                    if (np.linalg.norm(q[ii]-q[jj],axis=-1)>.1).all():break
                s=dict(scene=scene,index=index,N=8,positions=q.tolist(),goals=g.tolist(),velocities=np.zeros_like(p).tolist())
                s['parent_physical_identity']=uid('swap_robot_state',dict(index=index,positions=s['positions'],goals=s['goals']))
                states.append(s)
            if scene=='lit_swap8_free':free=states
        rows=[reachable(s) for s in states];assert all(r['eligible'] for r in rows),(scene,rows)
        write(HERE/f'FLOW_STATES_{scene}.json',states);write(HERE/f'GEOMETRY_VALIDATION_{scene}.json',rows)
        write(HERE/f'LEARNING_SPLIT_{scene}.json',dict(training=states[:16],evaluation=states[16:]))
    paths=[HERE/f'{prefix}_{scene}.json' for scene in ('lit_swap8_free','lit_swap8_obstacles','lit_swap8_moving') for prefix in ('FLOW_STATES','GEOMETRY_VALIDATION','LEARNING_SPLIT')]
    seal('SCENES.receipt.json',paths,before_any_new_research_rollout=True,seeds=[2026102311,2026102312,2026102313],moving_robot_samples='Exactly free states; third seed controls obstacle RNG only',layout='Prompt-declared four L arms, not calibrated reproduction of Figure 7 I-shaped obstacle')
    print('96 scenes built and individually reachable; static arms obstruct some straight paths (reported)',flush=True)
if __name__=='__main__':main()
