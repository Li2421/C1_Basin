"""Freeze the same prescribed 64 candidate points, independently of outcomes."""
from core import *
from scipy.stats import qmc
from shared_control.basis_dictionary import ALL,NEW,SIGNED


def main():
    p=HERE/'FAMILIES_64.json'
    if p.exists():return read(p)
    old=ROOT/'diagnostics/scene_types_basis_dimension_v1'
    original=read(old/'FAMILIES_FROZEN.json');ranges=read(old/'GOAL_RANGE_FROZEN.json')
    families=[]
    for j,f in enumerate(original['families']):
        points=json.loads(json.dumps(ranges['pools']['narrow32']if j==0 else f['candidate_pool']))
        seen={db.canonical(sp)for sp in points};columns=[x for x in ALL if x in f['columns']]
        for sample in qmc.Sobol(len(columns),scramble=True,seed=2026102805).random_base2(10):
            values={}
            for name,u in zip(columns,sample):
                lo,hi=(.5,1.25)if name=='G'else(-.5,.5)if name=='P'else(0.,.75)if name=='R'else(-1.,1.)if name in SIGNED else(0.,1.)
                values[name]=float(lo+(hi-lo)*u)
            sp=dict(eta=[values.get(x,0.)for x in ALL[:3]],coefficients={x:values[x]for x in NEW if x in values})
            if db.canonical(sp)not in seen:points.append(sp);seen.add(db.canonical(sp))
            if len(points)==64:break
        assert len(points)==64 and points[:16]==f['candidate_pool']
        families.append(dict(index=j,columns=f['columns'],dimension=f['dimension'],candidate_pool=points))
    result=dict(families=families,K=64,seed=2026102805,original_sha256=sha(old/'FAMILIES_FROZEN.json'),
                range_sha256=sha(old/'GOAL_RANGE_FROZEN.json'),rule='Frozen sequential order; first success stops; no generator and no selection.')
    dump(p,result);return result


if __name__=='__main__':main()
