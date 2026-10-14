from core import *
import copy,pickle
from shared_rollout_db.src.rollout_db import uid

def param_leaves(x,prefix=''):
    if isinstance(x,dict):
        out={}
        for k,v in sorted(x.items()):out.update(param_leaves(v,prefix+'/'+str(k)))
        return out
    return {prefix:np.asarray(x)}

def main():
    prereg();audit=read(HERE/'CHECKPOINT_NOVELTY_AUDIT.json');assert audit['DB_identity_audit_passed'];byname={r['name']:r for r in audit['checkpoints']};old=read(R10/'protocol.json');profiles={};rows=[];good=[]
    import jax
    jax.config.update('jax_enable_x64',False)
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    probe_state=read(R5/'stage_2/probes/protocol.json')['states'][0]
    for n in old['controller_order']:
        rec=byname[n];base=old['base_profiles'][n];payload=copy.deepcopy(base['payload']);row=dict(run=n,controller=n+'__final',best_sha256=rec['best_sha256'],final_sha256=rec['final_sha256'],compatible=False,error=None,physical_steps=0)
        try:
            assert sha(rec['best_path'])==rec['best_sha256'] and sha(rec['final_path'])==rec['final_sha256'];assert rec['best_sha256']==payload['flow_checkpoint_sha256']
            manifest=payload.get('adapter_manifest_path')
            if manifest is None:manifest=str(FIELD/'assets/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')
            frame=payload.get('observation_frame','local');m=read(manifest)
            with Path(rec['best_path']).open('rb') as f:best=pickle.load(f)
            with Path(rec['final_path']).open('rb') as f:final=pickle.load(f)
            assert best['schema']==final['schema'];bc=best['config'];fc=final['config'];assert set(bc)==set(fc)
            normalization=['obs_mean','obs_scale','act_mean','act_scale']
            for k in normalization:assert np.array_equal(np.asarray(bc[k]),np.asarray(fc[k])),('normalizer mismatch',k)
            for k in bc:
                if k not in normalization:assert clean(bc[k])==clean(fc[k]),('config mismatch',k)
            bp=param_leaves(best['agent']['network']['params']);fp=param_leaves(final['agent']['network']['params']);assert set(bp)==set(fp)
            diff=[];norm=[];shape_map={}
            for k in bp:
                assert bp[k].shape==fp[k].shape and bp[k].dtype==fp[k].dtype;d=(fp[k].astype(float)-bp[k].astype(float)).ravel();diff.append(d);norm.append(bp[k].astype(float).ravel());shape_map[k]=dict(shape=list(bp[k].shape),dtype=str(bp[k].dtype))
            d=np.concatenate(diff);w=np.concatenate(norm);row.update(parameter_L2=float(np.linalg.norm(d)),parameter_relative_L2=float(np.linalg.norm(d)/np.linalg.norm(w)),parameter_max_abs=float(np.max(abs(d))),parameter_count=len(w),network_shapes=shape_map,normalizers_identical=True,architecture_identical=True)
            rt=RingFlowAdapter(rec['final_path'],m,frame);env=rt.make_env(probe_state);obs=np.asarray(rt.observation(env));key=jax.random.PRNGKey(20261013120);a=np.asarray(rt.native_fn(obs,key));a2=np.asarray(rt.native_fn(obs,key));v=np.asarray(rt.velocity_fn(obs,np.zeros(8),np.asarray(.5)));v2=np.asarray(rt.velocity_fn(obs,np.zeros(8),np.asarray(.5)));assert a.tobytes()==a2.tobytes() and v.tobytes()==v2.tobytes();assert np.isfinite(a).all() and np.isfinite(v).all()
            chart_check=None
            if n=='base_u_v10_local_macflow':
                best_rt=RingFlowAdapter(rec['best_path'],m,frame);native=native_runtime('ring_exchange');e=best_rt.make_env(probe_state);ne=native.make_env(probe_state);aa=np.asarray(best_rt.native_fn(best_rt.observation(e),key));bb=np.asarray(native.native_fn(native.observation(ne),key));assert aa.tobytes()==bb.tobytes();assert clean(best_rt.config.to_dict())==clean(native.config.to_dict());chart_check='best adapter/native sampler bytes and physical config identical; projected32-case gate required before main'
            payload.update(flow_checkpoint_path=rec['final_path'],flow_checkpoint_sha256=rec['final_sha256'],adapter_manifest_path=manifest,adapter_manifest_sha256=sha(manifest),observation_frame=frame,checkpoint_environment_fingerprint=m['environment_fingerprint'],execution_callable=str(P1/'flow_adapter.py'),execution_callable_sha256=sha(P1/'flow_adapter.py'),controller_source_sha256=sha(P1/'flow_adapter.py'),parent_best_controller_uid=base['controller_uid'],checkpoint_role='never-closed-loop-evaluated final sibling',compatibility_audit_sha256_pending=True)
            name=n+'__final';profiles[name]=dict(controller_uid=uid('ctl',payload),payload=payload,parent_best_name=n);good.append(name);row.update(compatible=True,manifest_path=manifest,manifest_sha256=sha(manifest),observation_frame=frame,forward_reproducible=True,native_v10_chart_check=chart_check,final_training_step=final.get('metadata',{}).get('step'),best_training_step=best.get('metadata',{}).get('step'))
        except Exception as exc:row['error']=dict(type=type(exc).__name__,message=str(exc));print(json.dumps(dict(run=n,error=row['error'])),flush=True)
        rows.append(row)
    freeze(HERE/'FINAL_BASE_PROFILES.json',dict(profiles=profiles,eligible_controller_order=good,full_controller_order=[n+'__final' for n in old['controller_order']],best_controller_order=old['controller_order']))
    write(HERE/'COMPATIBILITY_AUDIT.json',dict(status='PASS' if len(good)==14 else 'PARTIAL_INCOMPATIBLE',usable=len(good),unavailable=14-len(good),checks=rows,success_labels_generated=False,physical_steps=0,adapter_sha256=sha(P1/'flow_adapter.py')))
    assert good,'No compatible final checkpoints'
    print(json.dumps(dict(status='PASS' if len(good)==14 else 'PARTIAL',usable=len(good),unavailable=14-len(good))))
if __name__=='__main__':main()
