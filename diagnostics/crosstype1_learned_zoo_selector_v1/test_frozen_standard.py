"""Recompute archived predictions only; no training or task outcomes are read."""
from common import *
from frozen_standard import predict,OLD_ARMS
import jax


def main(backend):
    assert all(d.platform==backend for d in jax.devices())
    archive=D/'ring_controller_critic_v1';checks=[]
    ge=read(archive/'predictions/G_E_FROZEN_CHOICES.json')['external']
    sources={'T1':D/'ring_independent_flow_confirmation_v2',
             'T3':D/'controller_new_checkpoint_confirmation_v1/fp32_n8'}
    for tag,folder in sources.items():
        with np.load(folder/'R5_FINAL_FEATURES.npz',allow_pickle=False) as z:
            names=list(z['controller_names']);r5=z['features'].astype(np.float32)
        with np.load(archive/f'predictions/{tag}_A0_seed17.npz',allow_pickle=False) as z:wanted=list(z['controller_names'])
        r5=r5[[names.index(n) for n in wanted]]
        choices,scores=predict(read(folder/'FRESH_STATES.json'),r5)
        for model in ('A0','A3'):
            references=[]
            for seed in (17,23,41):
                with np.load(archive/f'predictions/{tag}_{model}_seed{seed}.npz',allow_pickle=False) as z:
                    assert list(z['controller_names'])==wanted;references.append(z['probability'])
            reference=np.mean(references,axis=0).mean(1)
            delta=float(np.max(np.abs(scores[model]-reference)))
            same=np.array_equal(choices[model],np.asarray(OLD_ARMS)[reference.argmax(-1)])
            checks.append(dict(cohort=tag,method=model,controllers=len(wanted),max_probability_error=delta,choices_identical=bool(same),passed=bool(same and delta<=2e-6)))
        reference={r['controller']:r['action'] for r in ge[tag]}
        same=all(choices['G_E'][i]==reference[n] for i,n in enumerate(wanted))
        checks.append(dict(cohort=tag,method='G_E',controllers=len(wanted),choices_identical=bool(same),passed=bool(same)))
    if backend=='gpu':
        sealed=read(HERE/'DEVELOPMENT_COMPARATOR_CHOICES.json')['methods']
        with np.load(HERE/'DEVELOPMENT_FEATURES.npz',allow_pickle=False) as z:
            names=list(z['names']);r5=z['r5']
        actual,_=predict(read(HERE/'EVAL_STATES.json'),r5)
        for method,values in actual.items():
            same=all(str(a)==sealed[method]['choices'][str(n)] for n,a in zip(names,values))
            checks.append(dict(cohort='current_development_frozen_choices',method=method,controllers=len(names),choices_identical=same,passed=same))
    result=dict(utc=utc(),passed=all(c['passed'] for c in checks),choices_identical=all(c['choices_identical'] for c in checks),
        backend=backend,checks=checks,training_updates=0,physical_steps=0,task_outcomes_read=False)
    target=HERE/f'FROZEN_STANDARD_REPRODUCTION_{backend.upper()}.json';assert not target.exists()
    write(target,result);print(json.dumps(result,indent=2))
    assert result['passed']


if __name__=='__main__':main(sys.argv[1])
