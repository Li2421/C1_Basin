"""Pure inference/serialization probes: never instantiate a DB-writing runtime."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
from audit import ROOT,OUT,read,decode,save,sha,digest,conn
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp
from flax import serialization
from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as t
from diagnostics.orthoflow3_ring_revision_v1.run_fresh import context_vector
from new_benchmark_common.safety_eta3 import ScenarioRuntime,SCENARIOS,state_token
from new_benchmark_common.basin_dataset_v1 import CONDITIONING_FLOW_ROOT,_environment_descriptor
from new_benchmark_common.macflow import load_checkpoint

def readonly_runtime(sc):
    # __new__ intentionally bypasses constructors that register state/controller records.
    r=ScenarioRuntime.__new__(ScenarioRuntime)
    r.name=sc;r.kind=SCENARIOS[sc]['kind']
    manifest=read(SCENARIOS[sc]['dataset']/'manifest.json')
    r.config_dict=manifest['scenario_config']
    r.agent,_=load_checkpoint(SCENARIOS[sc]['checkpoint'],expected_environment_fingerprint=manifest['environment_fingerprint'])
    if r.kind=='ring':
        from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
        r.config=LocalFrameConfig(**r.config_dict);r.make_env=lambda:RingExchangeEnv(r.config)
    else:
        from four_way_intersection.environment import Config,FourWayIntersectionEnv
        r.config=Config(**r.config_dict);r.make_env=lambda:FourWayIntersectionEnv(r.config)
    return r

def main():
    manifest=read(OUT/'hypothesis_test_manifest.json')
    norm=read(ROOT/'diagnostics/orthoflow3_generator_critic_v1/normalization.json')
    dims={sc:len(norm['scenarios'][sc]['h_mean']) for sc in t.SCENARIOS}; cdim=len(norm['scenarios']['ring_exchange']['c_mean'])
    gm=t.Generator();cm=t.Critic()
    gp=serialization.from_bytes(t.merge_initialized(gm,dims,cdim),(ROOT/'diagnostics/orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack').read_bytes())
    cp=serialization.from_bytes(t.merge_initialized(cm,dims,cdim,critic=True),(ROOT/'diagnostics/orthoflow3_generator_critic_v1/critic/seed23/checkpoint.msgpack').read_bytes())
    v2={r['state_uid']:r for r in pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet').to_pylist()}
    a={r['state_uid']:r for r in read(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/proposals.json')['states']}
    tests=[]
    for r in manifest['train_dev_inference_panel']:
        st=v2[r['state_uid']];p=a[r['state_uid']]
        tests.append((r['scenario'],r['state_uid'],decode(st['conditioning'])['flat'],decode(st['environment_descriptor']),p['etas'],p['old_scores'],p['old_index'],'train_dev'))
    frozen=read(ROOT/'diagnostics/orthoflow3_ring_k16_diagnostic_v1/frozen_proposals.json')
    originals={r['state_uid']:r for r in read(ROOT/'diagnostics/orthoflow3_ring_revision_v1/fresh_test_manifests.json')['states']}
    rt=readonly_runtime('ring_exchange')
    for p in frozen['states']:
        st=originals[p['state_uid']];env=rt.make_env();rt.reset(env,{'physical':st['physical']})
        key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(p['state_uid']))
        flow=rt.flow_world(env,key);flat=np.r_[rt.observation(env).ravel(),flow.ravel()]
        tests.append(('ring_exchange',p['state_uid'],flat,_environment_descriptor(rt),[p['mean'],*p['samples']],p['critic_scores'],p['critic_index'],'frozen_artifact_only'))
    # No continuation is requested here. Log cache-first semantics explicitly.
    with conn() as c: count=c.execute('SELECT COUNT(*) FROM experiment').fetchone()[0]
    save('H3_cache_preflight.json',{'requested_continuations':0,'truly_missing':0,'new_rollouts':0,
        'inference_only':True,'DB_readonly_opened':True,'experiment_rows_seen':count})
    rows=[];noise_hashes=[];proposal_seeds=[]
    for sc,sid,flat,env,expected,scores0,index0,pop in tests:
        n=norm['scenarios'][sc]
        h=(np.asarray(flat,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
        ctx=context_vector(env,norm,sc)
        raw=np.asarray(gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(ctx[None]),method=getattr(gm,t.METHOD[sc])))[0]
        seed=t.stable_int('generator-v1-proposals',41,sc,sid);proposal_seeds.append(seed)
        # Independent draws within one deterministic stream; no reseeding per proposal.
        noise=np.random.default_rng(seed).standard_normal((16,3));noise_hashes.append(digest(noise.tolist()))
        samples=np.asarray(t.eta_from_noise(jnp.asarray(raw)[None].repeat(16,0),jnp.asarray(noise,np.float32)),float)
        etas=np.array([np.asarray(t.eta_mean(jnp.asarray(raw)),float),*samples])
        logits=np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(ctx[None].repeat(17,0)),
                      jnp.asarray((etas-t.CENTER)/t.RADIUS),method=getattr(cm,t.METHOD[sc])))
        scores=1/(1+np.exp(-logits))
        permutation=np.random.default_rng(1616).permutation(17)
        shuffled=np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(ctx[None].repeat(17,0)),
                      jnp.asarray((etas[permutation]-t.CENTER)/t.RADIUS),method=getattr(cm,t.METHOD[sc])))
        one=np.asarray([cm.apply(cp,jnp.asarray(h[None]),jnp.asarray(ctx[None]),jnp.asarray(((etas[j]-t.CENTER)/t.RADIUS)[None]),method=getattr(cm,t.METHOD[sc]))[0] for j in range(17)])
        err=float(np.max(np.abs(etas-np.array(expected))));scoreerr=float(np.max(np.abs(scores-np.array(scores0))))
        permerr=float(np.max(np.abs(shuffled-logits[permutation])));batcherr=float(np.max(np.abs(one-logits)))
        assert err<=2e-6 and scoreerr<=2e-6 and permerr<=2e-6,(sid,err,scoreerr,permerr)
        # CPU matmul reduction ordering may differ between batch1 and17; report logits separately.
        assert np.max(np.abs(1/(1+np.exp(-one))-scores))<=2e-6
        # The in-progress Phase-A manifest ranks logits; the historical K16
        # manifest ranks float32 sigmoid scores. Audit both without conflating.
        probability_pick=int(np.argmax(scores));logit_pick=int(np.argmax(logits))
        expected_pick=logit_pick if pop=='train_dev' else probability_pick
        assert expected_pick==index0,(sid,pop,index0,probability_pick,logit_pick)
        same_argmax=int(permutation[np.argmax(shuffled)])==logit_pick
        tied=np.flatnonzero(logits==logits.max()).tolist()
        assert same_argmax or len(tied)>1
        rows.append({'scenario':sc,'state_uid':sid,'population':pop,'input_h_shape':list(h.shape),'input_c_shape':list(ctx.shape),
            'h_hash':digest(h.tolist()),'c_hash':digest(ctx.tolist()),'eta_max_error':err,'score_max_error':scoreerr,
            'permutation_max_logit_error':permerr,'batch1_vs17_max_logit_error':batcherr,'same_top1':True,
            'permuted_same_top1':same_argmax,'tied_max_indices':tied,'distinct_etas':len({digest(e.tolist()) for e in etas}),
            'probability_pick':probability_pick,'logit_pick':logit_pick,
            'probability_tie_indices':np.flatnonzero(scores==scores.max()).tolist(),
            'logits':logits.tolist(),'probabilities':scores.tolist(),
            'distinct_noise_vectors':len({digest(v.tolist()) for v in noise}),'proposal_seed':seed,
            'eta_domain_pass':bool(np.all(etas>=t.LOW) and np.all(etas<=t.HIGH)),
            'canonical_input_used':True})
    summary={'states':len(rows),'train_dev':sum(r['population']=='train_dev' for r in rows),
        'frozen_artifacts':sum(r['population']=='frozen_artifact_only' for r in rows),
        'exact_eta_reproductions':sum(r['eta_max_error']==0 for r in rows),
        'max_eta_error':max(r['eta_max_error'] for r in rows),'max_score_error':max(r['score_max_error'] for r in rows),
        'top1_match':sum(r['same_top1'] for r in rows),'permutation_top1_match':sum(r['permuted_same_top1'] for r in rows),
        'duplicate_seed_count':len(proposal_seeds)-len(set(proposal_seeds)),
        'duplicate_noise_cloud_count':len(noise_hashes)-len(set(noise_hashes)),
        'minimum_distinct_candidates':min(r['distinct_etas'] for r in rows),
        'new_rollouts':0,'classification':'NOT_REPRODUCED',
        'note':'No candidate identity, transform, context, ordering, reseeding or training/inference mismatch found in tested frozen code paths. Shared-stream pseudorandom draws are not empirical proof of population independence.'}
    save('H3_inference_results.json',{'summary':summary,'states':rows});print(summary)
if __name__=='__main__': main()
