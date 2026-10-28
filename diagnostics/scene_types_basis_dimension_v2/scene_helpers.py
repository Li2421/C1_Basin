"""Geometry-only forest proposals, independent of every controller outcome."""
from core import *

def forest_layout(count):
    path=HERE/'forest_layouts'/f'{count}.json'
    if path.exists():
        r=read(path);return [tuple(x)for x in r['rectangles']],r['seed']
    failures=[]
    for seed in range(2026102803,2026103803):
        rng=np.random.default_rng(seed);centers=[]
        for j in range(count):
            for attempt in range(2000):
                q=rng.uniform(-1.05,1.05,2)
                if centers:
                    delta=np.maximum(np.abs(np.asarray(centers)-q)-.3,0.)
                    if np.min(np.linalg.norm(delta,axis=1))<.3:continue
                centers.append(q);break
            else:break
        if len(centers)==count:
            rects=[(q[0]-.15,q[1]-.15,q[0]+.15,q[1]+.15)for q in centers]
            dump(path,dict(count=count,seed=seed,rectangles=rects,failed_seeds=failures,pair_clearance=.3,wall_clearance=.3,max_proposals_per_block=2000,max_seeds=1000,selection='Geometric sequential rejection only; no controller/correction/paper outputs observed.'))
            return rects,seed
        failures.append(seed)
    raise ValueError('Forest geometric proposal budget exhausted for count '+str(count))
