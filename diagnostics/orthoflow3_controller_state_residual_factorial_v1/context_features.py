"""Two horizons of the same source-only physical response probe."""
from __future__ import annotations
import argparse,json
import numpy as np
import jax
from .pipeline import OUT,OLD,read,write,CONTROLLERS
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20


def main(controller,shard,shards):
    jax.config.update('jax_enable_x64',False)
    assert not jax.config.x64_enabled
    protocol=read(OUT/'protocol.json')
    profile=next(p for p in protocol['profiles'] if p['name']==controller)
    rt=h20.rc.RichRuntime('ring_exchange',profile['path'])
    # RichRuntime.core is always the committed base controller. Its separate
    # alt_sha identifies the frozen future-controller intervention.
    assert profile['sha256']==(rt.alt_sha if profile['path'] else rt.core.checkpoint_sha)
    h20.rc.OUT=OUT
    pairs=read(OUT/'pairs.json');physical=read(OUT/'physical.json')
    old_pairs=read(OLD/'pairs.json');old_dataset=np.load(OLD/'dataset.npz')
    oldc=CONTROLLERS.index(controller)
    first_24={(r['state_uid'],r['eta_uid']):i for i,r in enumerate(old_pairs) if r['split']=='train'}
    h20_rows=[];h80_rows=[]
    for i,p in enumerate(pairs):
        if i%shards!=shard:continue
        state=physical[p['state_index']]
        assert state['state_uid']==p['state_uid']
        key=(p['state_uid'],p['eta_uid'])
        if key in first_24:
            short=np.asarray(old_dataset['context'][oldc,first_24[key]],float)
            assert np.isfinite(short).all()
            h20_rows.append({'pair_index':i,'context':short.tolist(),'valid':True,'historical_reuse':True})
        else:
            h20.rc.PROTOCOL={**h20.rc.PROTOCOL,'horizon_steps':20}
            res=h20.rc.cached(rt,state,p['eta'])
            h20_rows.append({'pair_index':i,'context':res['features']['mean'] if res['valid'] else None,
                             'valid':res['valid'],'error':res.get('error'),'historical_reuse':False})
        h20.rc.PROTOCOL={**h20.rc.PROTOCOL,'horizon_steps':80}
        res=h20.rc.cached(rt,state,p['eta'])
        h80_rows.append({'pair_index':i,'context':res['features']['mean'] if res['valid'] else None,
                         'valid':res['valid'],'error':res.get('error'),'steps':80})
        if (len(h80_rows))%24==0:print(json.dumps({'controller':controller,'shard':shard,'context_pairs':len(h80_rows)}),flush=True)
    write(OUT/f'features_{controller}_{shard}of{shards}.json',h20_rows)
    write(OUT/f'features_h80_{controller}_{shard}of{shards}.json',h80_rows)
    print({'controller':controller,'shard':shard,'H20':len(h20_rows),'H80':len(h80_rows),
           'H20_historical_reuse':sum(r['historical_reuse'] for r in h20_rows)},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--controller',choices=CONTROLLERS,required=True)
    ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=2)
    a=ap.parse_args();main(a.controller,a.shard,a.shards)
