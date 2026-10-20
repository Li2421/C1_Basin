from common import *
from geometry import make_geometry,task

def identical(a,b):
    a=np.asarray(a);b=np.asarray(b)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
    records=[]
    for family in 'ABCDEFG':
        a,ra=make_geometry(family,2026103999);b,rb=make_geometry(family,2026103999)
        identical(a.walls,b.walls);identical(a.nodes,b.nodes);assert ra==rb
        for seed in range(202610394000,202610395000):
            try:x=task(a,family,seed);break
            except ValueError:continue
        else:raise AssertionError('no legal deterministic task fixture')
        y=task(b,family,seed)
        identical(x[0],y[0]);identical(x[1],y[1]);assert x[3]==y[3]
        for p,q in zip(x[2],y[2]):identical(p,q)
        records.append(dict(family=family,geometry_seed=2026103999,task_seed=seed,geometry_bytes=True,start_goal_bytes=True,path_bytes=True,horizon_equal=True))
    write(HERE/'GENERATOR_REPRODUCTION_TEST.json',dict(passed=True,records=records,new_rollouts=0))

if __name__=='__main__':main()
