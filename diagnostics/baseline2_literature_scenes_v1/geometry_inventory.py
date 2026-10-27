"""Memoized geometry-only inventory; no simulator or controller execution."""
from experiment import *
from flow_scene import LiteratureGeometry,validate
from states import config
from functools import lru_cache
from scipy.ndimage import label
import numpy as np

@lru_cache(None)
def circle_grid():
    c=config('lit_circle_five',5);geom=LiteratureGeometry(c);step=.1
    x=np.arange(-11.1,11.1+step*.5,step);y=x.copy();xx,yy=np.meshgrid(x,y)
    # Chunk the 128-wall calculation to bound peak memory.
    points=np.stack([xx,yy],-1).reshape(-1,2);good=np.concatenate([geom.valid_points(chunk) for chunk in np.array_split(points,32)]).reshape(xx.shape)
    labels,_=label(good,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    space=2*c.agent_radius+2*geom.margin*.001+.01;gx,gy=np.meshgrid(np.arange(-11.1,11.1,space),np.arange(-11.1,11.1,space));slots=np.stack([gx,gy],-1).reshape(-1,2)
    return geom,x,y,labels,int(geom.valid_points(slots).sum())

def circle_validate(s):
    geom,x,y,labels,capacity=circle_grid();p=np.asarray(s['positions']);g=np.asarray(s['goals']);ii,jj=np.triu_indices(5,1)
    starts=bool(geom.valid_points(p).all() and np.all(np.linalg.norm(p[ii]-p[jj],axis=-1)>4))
    goals=bool(geom.valid_points(g).all() and np.all(np.linalg.norm(g[ii]-g[jj],axis=-1)>=4-1e-12))
    def components(point):
        ix=int(np.argmin(abs(x-point[0])));iy=int(np.argmin(abs(y-point[1])));result=set()
        for jy in range(max(0,iy-1),min(len(y),iy+2)):
            for jx in range(max(0,ix-1),min(len(x),ix+2)):
                if labels[jy,jx] and geom.visible(point,np.array([x[jx],y[jy]])):result.add(int(labels[jy,jx]))
        return result
    paths=[bool(components(a)&components(b)) for a,b in zip(p,g)]
    return dict(scene=s['scene'],N=5,index=s['index'],valid_initial=starts,valid_goals=goals,single_robot_paths=paths,grid_step=.1,capacity_lower_bound=capacity,capacity_enough=capacity>=5,joint_solvable='not_certified',tangent_goal_pairs=[],eligible=bool(starts and goals and all(paths) and capacity>=5))

def main(scene):
    prereg();states=[s for s in read(HERE/'FLOW_STATES_UNCHECKED.json') if s['scene']==scene]
    rows=[circle_validate(s) if scene=='lit_circle_five' else validate(s) for s in states];eligible=[s for s,r in zip(states,rows) if r['eligible']]
    write(HERE/f'GEOMETRY_VALIDATION_{scene}.json',rows);write(HERE/f'FLOW_STATES_{scene}.json',eligible)
    seal(f'GEOMETRY_VALIDATION_{scene}.receipt.json',[HERE/f'GEOMETRY_VALIDATION_{scene}.json',HERE/f'FLOW_STATES_{scene}.json',HERE/'flow_scene.py',HERE/'geometry_inventory.py'],before_scene_flow=True)
    print(scene,'eligible',len(eligible),'rejected',len(states)-len(eligible))
if __name__=='__main__':main(sys.argv[1])
