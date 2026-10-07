"""Final immutable-input and paired uncertainty checks; never changes model choice."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from .study import ROOT,OLD,SCENES,SEEDS,read,write,sha

def main():
    protocol=read(ROOT/'protocol.json');assert protocol['code_sha256']==sha(ROOT/'study.py')
    assert all(sha(p)==h for p,h in protocol['source_hashes'].items())
    fits=list((ROOT/'models').glob('*/seed*/complete.json'));assert len(fits)==6
    for p in fits:
        r=read(p);assert all(r[k] for k in ('matched_init','matched_pair_order','matched_batch_draws'))
        assert r['steps']==2000 and not r['TEST_labels_read'] and r['code_sha256']==sha(ROOT/'study.py')
    d=np.load(OLD/'dataset.npz');t=np.load(OLD/'test_truth.npz');z=np.load(ROOT/'predictions_frozen.npz');o=t['outcomes'][1]
    s=((o[...,0]==1)&(o[...,1]==0)).sum(-1);f=16-o[...,1].sum(-1)-s;intervals=[];rng=np.random.default_rng(461)
    for scene in SCENES:
        idx=np.flatnonzero((d['scene']==scene)&(d['split']=='test'));ii=[int(np.flatnonzero(t['indices']==i)[0]) for i in idx]
        def per_state(kind,condition='correct'):
            loss=[]
            for seed in SEEDS:
                zz=z[f'{scene}__{kind}__{seed}__test__{condition}'];loss.append((s[ii]*np.logaddexp(0,-zz)+f[ii]*np.logaddexp(0,zz)).sum(1)/(s[ii]+f[ii]).sum(1))
            return np.mean(loss,0)
        for label,delta in [('eta_minus_shrunk',per_state('eta_only')-per_state('full_shrunk')),
                            ('full_minus_shrunk',per_state('full')-per_state('full_shrunk')),
                            ('shuffled_minus_correct_context',per_state('full_shrunk','context_shuffle')-per_state('full_shrunk'))]:
            draw=rng.integers(len(idx),size=(20000,len(idx)))
            intervals.append(dict(scene=scene,contrast=label,mean_state_NLL_difference=float(delta.mean()),family_bootstrap95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist()))
    metrics=list(csv.DictReader((ROOT/'metrics.csv').open()))
    for sc in SCENES:
        eta=[int(r['B15']) for r in metrics if r['scene']==sc and r['kind']=='eta_only' and r['condition']=='correct']
        shrunk=[int(r['B15']) for r in metrics if r['scene']==sc and r['kind']=='full_shrunk' and r['condition']=='correct']
        assert eta==shrunk and len(eta)==3
    result=dict(status='PASS',original_assets_unchanged=True,matched_small_fits=6,new_rollouts=0,
        classification='PRIOR_SHRINKAGE_HELPFUL_ON_REGRESSION_PANEL',no_h_selection_gain=False,
        probability_formula='p=(1-alpha)*p_eta+alpha*p_full; Toy alpha=.75, Ring alpha=.25 selected by VAL NLL only',
        conclusions=['Retains eta-prior top1 ceiling while improving probability NLL; no evidence of beating eta-only B15',
                     'Explicit h removal alone does not solve selection; C retains state-conditioned response',
                     'Three training seeds are not three independent test cohorts',
                     'Previously examined TEST is a regression check; no new independent generalization claim'],
        paired_probability_uncertainty=intervals,models_modified_in_main_pipeline=False,
        next_action='STOP_FOR_USER_REVIEW; no new training, rollout, architecture search, or dataset expansion queued',
        completion_jobs={'training_array':4786,'evaluation':4790,'both_blocking_wait_exit_codes':0},
        reproduction=['python -m diagnostics.orthoflow3_field_transfer_small_v1.evidence',
                      'python -m diagnostics.orthoflow3_field_transfer_small_v1.verify'],
        protocol_sha256=sha(ROOT/'protocol.json'),selection_sha256=sha(ROOT/'selection_frozen.json'),
        evidence_sha256=sha(ROOT/'paper_evidence.json'),results_sha256=sha(ROOT/'final_decision.json'))
    write(ROOT/'verification.json',result);write(ROOT/'working_state.json',dict(status='completed_for_user_review',new_rollouts=0,live_jobs=[]))
    print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
