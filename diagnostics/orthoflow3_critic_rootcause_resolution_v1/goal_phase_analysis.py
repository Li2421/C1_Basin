"""Causal phase test: cached baseline versus stationary near-goal blend."""
import os,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .goal_phase_intervention import OUT,EXP,SOURCE,read,write,sha,csvwrite
from .goal_response_cv import OUT as MODELS,model
from shared_rollout_db.src.rollout_db import connect,canonical,ROOT as DBROOT


def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    protocol=read(OUT/'protocol.json');pairs=read(OUT/'pairs.json');rows=[];s=np.zeros((2,16));f=s.copy();num=s.copy()
    with connect(True) as db:
      for i,p in enumerate(pairs):
        records=[]
        for ci,c in enumerate((protocol['parent_profile'],protocol['profiles'][0])):
            rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
            rr=[rr[canonical({'future_index':k})] for k in range(16)]
            assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
            records.append(rr);s[ci,i]=sum(r['success'] for r in rr if not r['numerical_failure']);f[ci,i]=sum(not r['success'] for r in rr if not r['numerical_failure']);num[ci,i]=sum(r['numerical_failure'] for r in rr)
        matched=[(a,b) for a,b in zip(*records) if not a['numerical_failure'] and not b['numerical_failure']]
        aa=np.array([a['success'] for a,b in matched]);bb=np.array([b['success'] for a,b in matched]);rescue=int(((aa==0)&(bb==1)).sum());br=int(((aa==1)&(bb==0)).sum())
        rows.append(dict(pair_index=i,state_uid=p['state_uid'],eta_index=p['eta_index'],parent_success=int(s[0,i]),blend_success=int(s[1,i]),parent_numerical=int(num[0,i]),blend_numerical=int(num[1,i]),paired_valid=len(matched),paired_Q_delta=float(np.mean(bb-aa)),rescue=rescue,breaks=br,exact_p=float(binomtest(rescue,rescue+br,.5).pvalue) if rescue+br else 1.))
    pp=np.array([r['exact_p'] for r in rows]);order=np.argsort(pp);adjusted=np.empty(16);adjusted[order]=np.minimum(1,np.maximum.accumulate(pp[order]*(16-np.arange(16))))
    for r,p in zip(rows,adjusted):r['Holm_p']=float(p)
    csvwrite(OUT/'paired_controller_Q.csv',rows)
    sourcepairs=read(SOURCE/'pairs.json');lookup={(p['state_uid'],p['eta_uid']):i for i,p in enumerate(sourcepairs)};ii=np.array([lookup[(p['state_uid'],p['eta_uid'])] for p in pairs])
    d=np.load(SOURCE/'dataset.npz');x=dict(np.load(SOURCE/'frozen_model_entities.npz'))
    ga=np.load(MODELS/'inputs_alt.npz')['goal_response'][ii];gb=np.load(MODELS/'inputs_second.npz')['goal_response'][ii];frozen=read(MODELS/'models_frozen.json')
    assert sha(MODELS/'models_frozen.json')==protocol['goal_models_frozen_sha256']
    predictions={};out=[];truth=s/(s+f);delta=truth[1]-truth[0]
    for kind in ('eta_only','H20_only','H20_goal'):
      m=model(kind);apply=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
      for entry in [e for e in frozen['models'] if e['kind']==kind]:
        path=Path(entry['path']);assert sha(path/entry['checkpoint'])==entry['checkpoint_sha256'];norm=read(path/'normalization.json');par=serialization.msgpack_restore((path/entry['checkpoint']).read_bytes())
        eta=(d['eta'][ii]-norm['eta_center'])/norm['eta_scale'];cc=(d['context'][0,ii]-norm['context_center'])/norm['context_scale'];ar=(d['agent_response'][0,ii]-norm['agent_center'])/norm['agent_scale']
        zz=[]
        for goal in (ga,gb):
            ctx=np.concatenate([cc,ar.reshape(16,-1),(goal-norm['goal_center'])/norm['goal_scale']],-1)
            zz.append(np.asarray(apply(par,gather(x,d['state_index'][ii]),jnp.asarray(eta,jnp.float32),jnp.asarray(ctx,jnp.float32))))
        z=np.array(zz);pr=expit(z);dp=pr[1]-pr[0];key=f"{entry['variant']}__{entry['seed']}";predictions[key]=z
        if kind!='H20_goal':assert np.array_equal(z[0],z[1])
        out.append(dict(variant=entry['variant'],seed=entry['seed'],parent_NLL=float((s[0]*np.logaddexp(0,-z[0])+f[0]*np.logaddexp(0,z[0])).sum()/(s[0]+f[0]).sum()),
            blend_NLL=float((s[1]*np.logaddexp(0,-z[1])+f[1]*np.logaddexp(0,z[1])).sum()/(s[1]+f[1]).sum()),
            mean_predicted_Q_change=float(dp.mean()),mean_true_Q_change=float(delta.mean()),delta_MAE=float(abs(dp-delta).mean()),
            delta_correlation=float(np.corrcoef(delta,dp)[0,1]) if dp.std()>1e-8 and delta.std()>1e-8 else None,
            large_effect_sign_correct=int(((np.sign(delta)==np.sign(dp))&(abs(delta)>=.25)).sum()),large_effect_cells=int((abs(delta)>=.25).sum()),
            max_probability_difference=float(abs(dp).max())))
    csvwrite(OUT/'frozen_model_causal_predictions.csv',out);np.savez_compressed(OUT/'frozen_predictions.npz',**predictions)
    raw=[]
    for path in (DBROOT/'journals'/EXP).glob('*.jsonl'):
      for line in path.read_text().splitlines():raw.append(json.loads(line)['record'])
    active=[r for r in raw if r['goal_blend_active_steps']>0]
    proof=read(OUT/'input_identity_proof.json')
    write(OUT/'causal_adjudication.json',dict(cells=16,paired_valid_trials=sum(r['paired_valid'] for r in rows),parent_numerical=int(num[0].sum()),blend_numerical=int(num[1].sum()),
        significant_Holm=int((adjusted<.05).sum()),mean_absolute_Q_change=float(abs(delta).mean()),mean_signed_Q_change=float(delta.mean()),
        parent_B15=int((s[0]>=15).sum()),blend_B15=int((s[1]>=15).sum()),robust_to_complete_failure=int(((s[0]>=15)&(s[1]==0)&(num[1]==0)).sum()),
        intervention_active_rollouts=len(active),total_rollouts=len(raw),first_active_step_range=[min(r['goal_blend_first_active_step'] for r in active),max(r['goal_blend_first_active_step'] for r in active)] if active else None,
        H20_exact_same=True,H80_also_exact_same_by_speed_bound=bool(proof['minimum_initial_agent_goal_distance']-80*.05*.52>1),goal_response_different=True,
        model_results=out,scope='Stationary physical-phase controller-program intervention on sourceVALfamilies; not naturalMLP or crossscene confirmation. Nointerventionlabels enteredtraining.',new_continuations=256,generator_modified=False))
    print(read(OUT/'causal_adjudication.json'))


if __name__=='__main__':main()
