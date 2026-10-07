import json,copy
import numpy as np
import jax
from . import representation as r,build

def run():
    rows=build.rows();items=[r.entities(build.scene(x)) for x in rows]
    enc=r.Encoder();params=enc.init(jax.random.PRNGKey(0),r.batch(items[:1]))
    encode=jax.jit(lambda x:enc.apply(params,x));metrics=[]
    for sc in sorted({x['scenario'] for x in rows}):
        cohort=sorted([x for x in rows if x['scenario']==sc],key=lambda x:x['state_uid'])[:12]
        for row in cohort:
            scene=build.scene(row);e=r.entities(scene);h=np.asarray(encode(r.batch([e])))[0]
            n=len(scene['positions']);m=len(scene['obstacles']);perm=np.roll(np.arange(n),1);op=np.arange(m)[::-1]
            for k in range(4):
                s=r.transform(scene,k*np.pi/2,(1.25,-.75),perm,op);ee=r.entities(s)
                inv=np.argsort(perm);oi=np.argsort(op)
                raw=max(np.max(np.abs(e['agents']-ee['agents'][inv])),np.max(np.abs(e['pairs']-ee['pairs'][inv][:,inv])),
                        np.max(np.abs(e['obstacles']-ee['obstacles'][inv][:,oi])))
                error=float(np.max(np.abs(h-np.asarray(encode(r.batch([ee])))[0])))
                assert raw<2e-6,(sc,k,raw);assert error<2e-5,(sc,k,error)
                metrics.append({'scenario':sc,'state_uid':row['state_uid'],'rotation':90*k,'raw_error':float(raw),'h_error':error})
    variable=[];synthetic=[];base=build.scene(rows[0])
    for n in (2,3,4,5):
        for m in (0,1,7,len(base['obstacles'])):
            s=copy.deepcopy(base)
            for key in ('positions','goals','velocities','flow','radius','goal_tolerance'):
                v=np.asarray(s[key]);s[key]=v[np.arange(n)%len(v)].tolist()
            if n==5:s['positions'][-1][0]+=.2;s['goals'][-1][0]+=.2
            s['obstacles']=s['obstacles'][:m]
            x=r.batch([r.entities(s)]);h=np.asarray(encode(x))
            assert h.shape==(1,128) and np.isfinite(h).all()
            synthetic.append((r.entities(s),h[0]))
            variable.append({'N':n,'M':m,'shape':list(h.shape),'passed':True})
    mixed=np.asarray(encode(r.batch([x for x,h in synthetic])))
    padding_error=float(np.max(np.abs(mixed-np.stack([h for x,h in synthetic]))))
    assert padding_error<2e-5,padding_error
    allh=np.asarray(encode(r.batch(items)));delta=np.max(np.abs(allh[:,None]-allh[None,:]),axis=-1)
    aliases=np.argwhere(np.triu(delta<1e-7,1));assert not len(aliases),aliases
    result={'passed':True,'tests':metrics,'variable_count':variable,'initial_embedding_exact_alias_pairs':aliases.tolist(),
        'max_h_error':max(x['h_error'] for x in metrics),'no_claim_unseen_N_policy_generalization':True,
        'mixed_batch_padding_error':padding_error,
        'Four_active_rotation_invariance_not_required_by_contract':True}
    build.dump('structural_tests.json',result);np.save(build.OUT/'initial_embeddings.npy',allh)
    return {k:v for k,v in result.items() if k not in ('tests','variable_count')}

if __name__=='__main__':print(json.dumps(run(),indent=2))
