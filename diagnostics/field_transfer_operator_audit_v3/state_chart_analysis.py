"""Remove the Ring radial-frame coordinate change from fixed-noise state comparisons."""
from collections import defaultdict
import numpy as np
from cache_analysis import PARENT,CACHE,read,save,stats

def dispersion(m):
    m=np.stack(m);mean=m.mean(0)
    return float(np.sqrt(np.mean(np.sum((m-mean)**2,axis=(1,2))))/np.linalg.norm(mean))

def main():
    states=read(PARENT/'confirmation_manifest.json')['states'];out={}
    for sc in ['toy_give_way','ring_exchange']:
        full=defaultdict(list);groups=defaultdict(list)
        for st in [s for s in states if s['scenario']==sc]:
            z=np.load(CACHE/'state_dependence_control'/f"{st['uid']}.npz")
            p=np.array(st['physical']['positions']);d=p.size;r=np.eye(d)
            if sc=='ring_exchange':
                for i,q in enumerate(p):
                    a=q/np.linalg.norm(q);r[2*i:2*i+2,2*i:2*i+2]=np.array([[a[0],-a[1]],[a[1],a[0]]])
            mm={ch:z['center0_M_exec'][i,2] for i,ch in enumerate(['TT','FT','TF','FF'])}
            mm['flow']=z['center0_M_flow'][2]
            for ch,m in mm.items():full[ch].append(r.T@m@r)
            sig=z['field_projection_activity_packed'][1].tobytes().hex()
            groups[sig].append(dict(uid=st['uid'],M=r.T@mm['FF']@r,flow=r.T@mm['flow']@r))
        out[sc]=dict(local_physical_chart_dispersion={ch:dispersion(m) for ch,m in full.items()},
            distinct_field_signatures=len(groups),same_signature_groups=[dict(states=[r['uid'] for r in g],
                FF_dispersion=dispersion([r['M'] for r in g]),flow_dispersion=dispersion([r['flow'] for r in g])) for g in groups.values() if len(g)>=2])
    save('state_chart_control.json',out)

if __name__=='__main__':main()
