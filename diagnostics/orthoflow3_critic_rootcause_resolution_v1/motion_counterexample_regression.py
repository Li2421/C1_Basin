"""Verify the proposed repair on already-cached input-alias counterexamples.

These eight families were excluded from model training but their outcomes have
already been inspected for mechanism discovery. This is a post-hoc regression
test, NOT a new independent confirmation or model-selection set. Zero new task
continuations; short feature probes have no outcome labels or terminal target.
"""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit
from .motion_factorial_train import OUT as MATRIX,model,read,write,sha,SEEDS,csvwrite
from .motion_freeze_train import OUT as FREEZE
from .goal_motion_confirmation import OUT as OLD
OUT=MATRIX.parent/'motion_counterexample_regression'


def prepare():
    assert not (OUT/'protocol.json').exists()
    states,pairs=read(OLD/'states.json'),read(OLD/'pairs.json')
    trainuids={s['uid'] for s in read(MATRIX/'alt_to_second/states.json')}
    assert not trainuids & {s['uid'] for s in states}
    choices=read(FREEZE/'source_selection.json')['selection']
    rest=read(MATRIX/'source_selection.json')['selection']['native_repeated']['rest_only']
    entries=[]
    for fold in range(3):
        for seed in SEEDS:
            for arm in ('old_rest_only','full_update','trunk_only'):
                kind='rest_only' if arm=='old_rest_only' else 'rest_motion'
                if arm=='old_rest_only':path=MATRIX/'cv'/f'fold{fold}'/'native_repeated'/kind/f'seed{seed}';step=rest['step']
                elif arm=='full_update':path=MATRIX/'cv'/f'fold{fold}'/'motion_intervention'/kind/f'seed{seed}';step=choices[arm]['step']
                else:path=FREEZE/'cv'/f'fold{fold}'/arm/f'seed{seed}';step=choices[arm]['step']
                checkpoint=path/f'step{step}.msgpack';normalization=path/'normalization.json'
                entries.append(dict(arm=arm,kind=kind,fold=fold,seed=seed,step=step,path=str(path),
                    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),normalization=str(normalization),normalization_sha256=sha(normalization)))
    source=read(OLD/'protocol.json')
    profiles=[source['parent_profile'],source['second_profile']]
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    write(OUT/'protocol.json',dict(experiment='posthoc_motion_alias_repair_regression',
        states_sha256=sha(OLD/'states.json'),pairs_sha256=sha(OLD/'pairs.json'),profiles=profiles,
        outcome_controllers=[source['parent_profile'],source['profiles'][0]],models=entries,
        family_overlap_TRAIN_VAL=0,new_task_rollouts=0,old_outcomes_already_opened=True,
        model_selection_uses_only_source_VAL=True,independent_generalization_confirmation=False,
        purpose='Check actual repair of the independently discovered input alias, without selecting or tuning models on it.'))
    print(dict(models=len(entries),states=len(states),new_rollouts=0),flush=True)


def inputs():
    import jax
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    from .agent_response import instrument
    from .goal_response import measurement as rest_measurement
    from .goal_velocity_response import measurement as motion_measurement
    assert jax.default_backend()=='gpu'
    native.OUT=OUT;native.physical()
    proto=read(OUT/'protocol.json');physical=read(OUT/'physical.json');pairs=read(OUT/'pairs.json')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20}
    for profile in proto['profiles']:
        dest=OUT/f'inputs_{profile["name"]}.npz'
        if dest.exists():read(OUT/f'audit_{profile["name"]}.json');continue
        assert sha(profile['path'])==profile['sha256']
        rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
        # Keep instantaneous stencil measurements on a noninstrumented runtime.
        point=rc.RichRuntime('ring_exchange',profile['path'])
        contexts=[];agents=[];rests=[];motions=[];errors=[]
        for i,p in enumerate(pairs):
            state=physical[p['state_index']];assert state['state_uid']==p['state_uid']
            trace['runs']=[]
            try:
                value=rt.features(state['physical'],p['eta'])
                assert len(trace['runs'])==4 and all(len(v)==20 for v in trace['runs'])
                means=np.array([np.array(v).mean(0) for v in trace['runs']])
                ctx=value['mean'];agent=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],-1)
                rest=rest_measurement(point,state['physical'],np.array(p['eta'],float))
                motion=motion_measurement(point,state['physical'],np.array(p['eta'],float))
                assert all(np.isfinite(v).all() for v in (ctx,agent,rest,motion))
            except Exception as exc:
                errors.append(dict(index=i,error=f'{type(exc).__name__}: {exc}'))
                ctx=np.full(24,np.nan);agent=np.full((4,16),np.nan);rest=np.full(16,np.nan);motion=rest.copy()
            contexts.append(ctx);agents.append(agent);rests.append(rest);motions.append(motion)
        valid=np.isfinite(contexts).all(1)&np.isfinite(rests).all(1)&np.isfinite(motions).all(1)
        np.savez_compressed(dest,context=np.array(contexts,np.float32),agent_response=np.array(agents,np.float32),
            goal_response=np.array(rests,np.float32),goal_motion_response=np.array(motions,np.float32),valid=valid)
        write(OUT/f'audit_{profile["name"]}.json',dict(invalid=errors,controller_sha256=profile['sha256'],
            labels_read=False,new_task_rollouts=0,probe_horizon=20,goal_probe_environment_steps=0))
        print(dict(profile=profile['name'],pairs=len(pairs),invalid=len(errors)),flush=True)


def evaluate():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    from shared_rollout_db.src.rollout_db import connect,canonical
    proto=read(OUT/'protocol.json');pairs=read(OUT/'pairs.json');x=dict(np.load(OUT/'entities.npz'))
    measured=[dict(np.load(OUT/f'inputs_{p["name"]}.npz')) for p in proto['profiles']]
    assert all(v['valid'].all() for v in measured), 'Do not silently remove numerical input failures'
    source=dict(np.load(MATRIX/'dataset.npz'));tr=np.flatnonzero(source['split']=='train')
    prior_source=(source['success'][[0,14]][:,tr]/(source['success'][[0,14]][:,tr]+source['failure'][[0,14]][:,tr])).reshape(2,64,2).mean(1)
    success=[];failure=[];keys=[]
    with connect(True) as db:
        for ctl in proto['outcome_controllers']:
            ss=[];ff=[]
            for pair in pairs:
                rows={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],ctl['controller_uid']))}
                rr=[rows[canonical({'future_index':k})] for k in range(16)]
                assert all(r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'] and not r['numerical_failure'] for r in rr)
                count=sum(r['success'] for r in rr);ss.append(count);ff.append(16-count)
                keys.append(dict(state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],controller_uid=ctl['controller_uid'],rollout_uids=[r['rollout_uid'] for r in rr]))
            success.append(ss);failure.append(ff)
    q=np.array(success)/16.;good=np.array(success)>=15
    write(OUT/'reused_DB_keys.json',keys)
    si=np.array([p['state_index'] for p in pairs]);eta=np.array([p['eta'] for p in pairs],np.float32)
    results=[];perpair=[];predictions={}
    for entry in proto['models']:
        assert sha(entry['checkpoint'])==entry['checkpoint_sha256'] and sha(entry['normalization'])==entry['normalization_sha256']
        norm=read(entry['normalization']);pp=serialization.msgpack_restore(Path(entry['checkpoint']).read_bytes())
        m=model(entry['kind']);predict=jax.jit(lambda xx,e,c:m.apply(pp,xx,e,c))
        zz=[]
        for condition in ('parent','motion_correct','motion_wrong_parent'):
            a=measured[0];b=measured[1] if condition=='motion_correct' else measured[0]
            cc=np.concatenate([(a['context']-norm['context_center'])/norm['context_scale'],
                ((a['agent_response']-norm['agent_center'])/norm['agent_scale']).reshape(len(pairs),-1),
                (a['goal_response']-norm['goal_center'])/norm['goal_scale'],
                (b['goal_motion_response']-norm['goal_motion_center'])/norm['goal_motion_scale']],-1)
            z=np.asarray(predict(gather(x,si),jnp.asarray((eta-norm['eta_center'])/norm['eta_scale'],jnp.float32),jnp.asarray(cc,jnp.float32)))
            zz.append(z)
        zz=np.array(zz);p=expit(zz);dq=q[1]-q[0];dp=p[1]-p[0]
        if entry['kind']=='rest_only':assert np.array_equal(zz[0],zz[1])
        # Same-controller program input differs only in the newly measured motion block.
        assert np.array_equal(zz[0],zz[2])
        chosen=zz[:2].reshape(2,8,2).argmax(-1);ci,hi=np.arange(2)[:,None],np.arange(8)[None,:]
        gp=good.reshape(2,8,2);qq=q.reshape(2,8,2)
        nll=(q*np.logaddexp(0,-zz[:2])+(1-q)*np.logaddexp(0,zz[:2])).mean()
        wrong_nll=(q[1]*np.logaddexp(0,-zz[2])+(1-q[1])*np.logaddexp(0,zz[2])).mean()
        meta={k:entry[k] for k in ('arm','fold','seed','step')}
        results.append(dict(**meta,constituents_seen=entry['fold']!=2,NLL=float(nll),
            B15=int(gp[ci,hi,chosen].sum()),oracle_B15=int(gp.any(-1).sum()),cases=16,
            selected_Q=float(qq[ci,hi,chosen].mean()),causal_delta_MAE=float(abs(dp-dq).mean()),
            causal_delta_correlation=float(np.corrcoef(dp,dq)[0,1]) if dp.std()>1e-9 else None,
            correct_motion_NLL=float((q[1]*np.logaddexp(0,-zz[1])+(1-q[1])*np.logaddexp(0,zz[1])).mean()),
            wrong_motion_NLL=float(wrong_nll),example_true_parent=float(q[0,1]),example_true_motion=float(q[1,1]),
            example_pred_parent=float(p[0,1]),example_pred_motion=float(p[1,1]),example_pred_delta=float(dp[1])))
        for i,pair in enumerate(pairs):
            perpair.append(dict(**meta,state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],
                parent_success=int(success[0][i]),motion_success=int(success[1][i]),parent_prediction=float(p[0,i]),
                motion_prediction=float(p[1,i]),true_delta=float(dq[i]),predicted_delta=float(dp[i])))
        predictions[f"{entry['arm']}__fold{entry['fold']}__seed{entry['seed']}"]=zz
    csvwrite(OUT/'metrics.csv',results);csvwrite(OUT/'per_pair_predictions.csv',perpair)
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    write(OUT/'audit.json',dict(new_task_rollouts=0,reused_exact_rollouts=512,model_TRAIN_VAL_family_overlap=0,
        labels_previously_opened=True,independent_confirmation=False,model_selection_on_these_labels=False,
        scope='Posthoc regression on original counterexamples, not promotion evidence or a strict LOSO result.',
        privileged_TRAIN_prior_probabilities=prior_source.tolist()))
    print([r for r in results if r['constituents_seen']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','inputs','evaluate'));a=p.parse_args();globals()[a.action]()
