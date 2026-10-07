"""Independent raw-bypass confirmation; unchanged audited rollout executor."""
import argparse,shutil
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from . import motion_confirmation as run
from . import motion_confirmation_eval as ev
from . import motion_confirmation_authorize as auth
from . import raw_bypass_final as final
from .motion_replication_adjudication import paired
ROOT=Path(__file__).resolve().parent
SOURCE,RULE=final.OUT,final.RULE
read,write,sha=run.read,run.write,run.sha
CONDITIONS=(*ev.CONDITIONS,'zero_raw_bypass')


def folder(replica):
    return ROOT/f'motion_independent_confirmation_{read(RULE)["target_rule"][replica]["controller_seed"]}'


def bridge():
    assert (SOURCE/'models_frozen.json').exists()
    for name in ('protocol.json','states.json','pairs.json'):
        a,b=run.ASSETS/name,SOURCE/name
        if not b.exists():shutil.copyfile(a,b)
        assert sha(a)==sha(b)
    a,b=RULE,SOURCE/'independent_confirmation_protocol.json'
    if not b.exists():shutil.copyfile(a,b)
    assert sha(a)==sha(b)
    gate=dict(passed=True,evidence_sha256=sha(ROOT/'context_raw_bypass/selection_frozen.json'),
        reason='Source-only raw skip improves all3seedB15 against matched expanded reference;correct-context and wrong-state controls matter. Independent confirmation still required.',
        source_only=True,target_labels_used=False)
    dest=SOURCE/'source_gate.json'
    if dest.exists():assert read(dest)==gate
    else:write(dest,gate)
    deps=[Path(__file__),Path(final.__file__),Path(run.__file__),Path(ev.__file__),
        ROOT/'context_raw_bypass.py',ROOT/'function_confirmation.py',ROOT/'goal_response.py',ROOT/'goal_velocity_response.py',
        ROOT.parent/'orthoflow3_controller_training_repair_v1/data.py',
        ROOT.parent/'orthoflow3_controller_intervention_generalization_v1/rich_context.py']
    guard=dict(models_sha256=sha(SOURCE/'models_frozen.json'),rule_sha256=sha(RULE),
        dependencies={str(p):sha(p) for p in deps},target_labels_used=False)
    dest=SOURCE/'confirmation_bridge.json'
    if dest.exists():assert read(dest)==guard,'Frozen dependency changed'
    else:write(dest,guard)


def configure(replica):
    run.SOURCE,run.RULE,run.bridge=SOURCE,RULE,bridge
    ev.SOURCE,ev.RULE,ev.folder=SOURCE,RULE,folder
    auth.SOURCE,auth.folder=SOURCE,folder
    run.configure(replica)


def forward(par,x,eta,context,variant,zero=False):
    if variant=='eta_mle':
        prior=read(SOURCE/'models_frozen.json')['eta_prior']
        return np.asarray(eta,np.float32)@np.asarray(prior['normalized_eta_coefficient'],np.float32)+np.float32(prior['intercept'])
    if variant not in ('raw_bypass','expanded_reference'):
        return ev.forward(par,x,eta,context,variant)
    cpu=ev.cpu
    x={k:np.asarray(v,np.float32) for k,v in x.items()}
    eta,context=np.asarray(eta,np.float32),np.asarray(context,np.float32)
    agent=context[:,24:88].reshape(len(eta),x['agents'].shape[1],16)
    mean=agent.mean(1)
    xx={**x,'agents':np.concatenate((x['agents'],np.broadcast_to(mean[:,None],agent.shape)),-1)}
    cc=np.concatenate((context[:,:24],context[:,88:120]),-1)
    h=cpu.physical(xx,par['physical_encoder'])
    e=cpu.silu(cpu.dense(eta,par['eta_encoder']))
    c=cpu.silu(cpu.dense(cc,par['context_encoder']))
    cid=cpu.silu(cpu.dense(np.zeros((len(eta),3),np.float32),par['id_encoder']))
    a=cpu.dense(np.concatenate((h,e,c,cid),-1),par['trunk1'])
    if not zero:a+=np.concatenate((cc,mean),-1)@par['raw_context_skip']['kernel']
    return cpu.dense(cpu.silu(cpu.dense(cpu.silu(a),par['trunk2'])),par['out'])[...,0]


def source_parity(entries):
    d=dict(np.load(ROOT/'state_breadth_training/dataset.npz'))
    x=dict(np.load(ROOT/'state_breadth_training/entities.npz'));va=np.arange(320,352)
    xx={k:v[d['state_index'][va]] for k,v in x.items()};checks=[]
    for entry in entries:
        par,norm=ev.load_entry(entry);eta=(d['eta'][va]-norm['eta_center'])/norm['eta_scale']
        zs=[]
        for ci in range(16):
            values={k:d[k][ci,va] for k in ('context','agent_response','goal_response','goal_motion_response')}
            zs.append(forward(par,xx,eta,ev.normalized_context(values,values,norm),entry['variant']))
        zz=np.array(zs)
        if entry['variant']=='eta_mle':
            expected=np.broadcast_to(logit(read(SOURCE/'models_frozen.json')['eta_prior']['probabilities']),(16,16,2)).reshape(16,32)
        else:expected=np.load(Path(entry['path'])/'predictions.npz')[f'step{entry["steps"]}']
        assert np.allclose(zz,expected,atol=2e-5,rtol=2e-5),(entry,float(abs(zz-expected).max()))
        assert np.array_equal(zz.reshape(16,16,2).argmax(-1),expected.reshape(16,16,2).argmax(-1))
        checks.append(dict(variant=entry['variant'],seed=entry['seed'],max_abs_logit_error=float(abs(zz-expected).max()),top1_exact=True,
            comparison='analyticMLE' if entry['variant']=='eta_mle' else 'cachedGPU'))
    return checks


def predict(replica):
    out=folder(replica);assert not (out/'prediction_freeze.json').exists()
    frozen=read(SOURCE/'models_frozen.json');entries=frozen['models']
    assert read(out/'protocol.json')['models_frozen_sha256']==sha(SOURCE/'models_frozen.json')
    checks=source_parity(entries)
    pairs,states=read(out/'pairs.json'),read(out/'states.json')
    assert len(states)==64 and len(pairs)==128
    si=np.array([p['state_index'] for p in pairs]);eta=np.array([p['eta'] for p in pairs],np.float32)
    assert np.array_equal(si,np.repeat(np.arange(64),2))
    assert np.array_equal(eta.reshape(64,2,3),np.broadcast_to(eta[:2],(64,2,3)))
    entities=dict(np.load(out/'entities.npz'))
    inp={name:dict(np.load(out/f'inputs_{name}.npz')) for name in ('held','alt')}
    response={name:dict(np.load(out/f'goal_both_{name}.npz')) for name in inp}
    assert all(a['valid'].all() for a in (*inp.values(),*response.values()))
    shift=np.roll(np.arange(64),-1);pshift=np.column_stack((2*shift,2*shift+1)).ravel();pred={}
    for entry in entries:
        par,norm=ev.load_entry(entry);ee=(eta-norm['eta_center'])/norm['eta_scale']
        contexts={name:ev.normalized_context(inp[name],response[name],norm) for name in inp}
        for cond in CONDITIONS:
            c=contexts['alt' if cond=='wrong_controller_alt' else 'held']
            if cond in ('joint_state_context_shuffle','all_context_wrong_state'):c=c[pshift]
            ix=shift[si] if cond in ('joint_state_context_shuffle','state_shuffle') else si
            xx={k:v[ix] for k,v in entities.items()}
            z=forward(par,xx,ee,c,entry['variant'],zero=cond=='zero_raw_bypass').reshape(64,2)
            assert np.isfinite(z).all()
            pred[f'{entry["variant"]}__{entry["seed"]}__{cond}']=z
        if entry['variant'] in ('eta_only','eta_mle'):
            z=pred[f'{entry["variant"]}__{entry["seed"]}__correct']
            assert np.allclose(z,z[:1],atol=2e-6,rtol=2e-6) and np.all(z.argmax(1)==z[0].argmax())
            assert all(np.array_equal(z,pred[f'{entry["variant"]}__{entry["seed"]}__{cond}']) for cond in CONDITIONS)
    np.savez_compressed(out/'frozen_predictions.npz',**pred)
    names=['protocol.json','pairs.json','states.json','entities.npz','planned_rollouts.json',
        'inputs_held.npz','inputs_alt.npz','goal_both_held.npz','goal_both_alt.npz']
    write(out/'prediction_freeze.json',dict(models_frozen_sha256=sha(SOURCE/'models_frozen.json'),protocol_sha256=sha(RULE),
        evaluator_sha256=sha(__file__),numpy_forward_sha256=sha(ev.cpu.__file__),input_sha256={k:sha(out/k) for k in names},
        predictions_sha256=sha(out/'frozen_predictions.npz'),source_parity=checks,
        target_labels_read=False,new_rollouts=0,conditions=CONDITIONS))
    print(dict(target=read(out/'protocol.json')['target_controller_seed'],predictions_frozen=True),flush=True)


def evaluate(replica):
    out=folder(replica);guard=read(out/'prediction_freeze.json')
    assert guard['evaluator_sha256']==sha(__file__) and guard['numpy_forward_sha256']==sha(ev.cpu.__file__)
    assert guard['models_frozen_sha256']==sha(SOURCE/'models_frozen.json') and guard['protocol_sha256']==sha(RULE)
    for k,v in guard['input_sha256'].items():assert sha(out/k)==v
    assert guard['predictions_sha256']==sha(out/'frozen_predictions.npz')
    state=read(out/'working_state.json');assert state['phase']=='postflight_complete'
    d=np.load(out/'dataset.npz');s,f,num=[d[k].reshape(64,2) for k in ('success','failure','numerical')]
    assert d['valid'].all() and np.all(s+f+num==16)
    scores=dict(np.load(out/'frozen_predictions.npz'));rows=[];comparisons=[]
    arms=('raw_bypass','expanded_reference','old_trunk','eta_only')
    for arm in arms:
        for cond in CONDITIONS:scores[f'{arm}__ensemble__{cond}']=logit(np.mean([expit(scores[f'{arm}__{seed}__{cond}']) for seed in final.base.SEEDS],0))
    for arm in (*arms,'eta_mle'):
        for seed in (*final.base.SEEDS,'ensemble') if arm!='eta_mle' else (0,):
            for cond in CONDITIONS:rows.append(dict(arm=arm,seed=seed,condition=cond,**ev.measures(scores[f'{arm}__{seed}__{cond}'],s,f)))
            if arm in ('eta_only','eta_mle'):continue
            aa=scores[f'{arm}__{seed}__correct']
            controls={c:scores[f'{arm}__{seed}__{c}'] for c in CONDITIONS[1:]}
            controls.update(eta_only=scores[f'eta_only__{seed}__correct'],eta_mle=scores['eta_mle__0__correct'],
                old_trunk=scores[f'old_trunk__{seed}__correct'],expanded_reference=scores[f'expanded_reference__{seed}__correct'])
            for name,zz in controls.items():comparisons.append(dict(arm=arm,seed=seed,reference=name,**paired(aa,zz,s,f)))
    ev.csvwrite(out/'metrics.csv',rows);ev.csvwrite(out/'paired_uncertainty.csv',comparisons)
    write(out/'evaluation_audit.json',dict(target=read(out/'protocol.json')['target_controller_seed'],
        cases=64,oracle_B15=int((s>=15).any(1).sum()),fixed_B15=(s>=15).sum(0).tolist(),
        eta0_only=int(((s[:,0]>=15)&(f[:,1]>=2)).sum()),eta1_only=int(((s[:,1]>=15)&(f[:,0]>=2)).sum()),
        numerical=int(num.sum()),collision=read(out/'alignment_audit.json')['collision'],
        models_sha256=sha(SOURCE/'models_frozen.json'),dataset_sha256=sha(out/'dataset.npz'),
        predictions_frozen_before_labels=True,target_labels_used_for_selection=False,
        scope=read(RULE)['scope'],new_evaluation_rollouts=0,
        statistical_scope='Family bootstrap conditional on this controller;trainingseedsnotindependentcohorts;numericalunknown retained.'))
    print([r for r in rows if r['condition']=='correct'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--replicate',type=int,default=0);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();configure(a.replicate)
    if a.action=='predict':predict(a.replicate)
    elif a.action=='evaluate':evaluate(a.replicate)
    elif a.action=='authorize':auth.main(a.replicate)
    else:run.main(a.action,a.replicate,a.index)
