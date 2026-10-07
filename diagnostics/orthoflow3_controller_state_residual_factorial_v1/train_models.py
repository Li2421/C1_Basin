"""Matched source-family H20/H80 and family-count comparisons."""
import argparse,csv
from pathlib import Path
import shutil
import numpy as np
from .pipeline import OUT,read,write
from diagnostics.orthoflow3_controller_training_repair_v1 import train as prior
from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic

SEEDS=(17,23,41)
MODELS=[('small_20','eta_only'),('large_20','eta_only'),
        ('small_20','physical_context'),('small_80','physical_context'),
        ('large_20','physical_context'),('large_80','physical_context'),
        ('small_20','context_only'),('small_80','context_only'),
        ('large_20','context_only'),('large_80','context_only')]
MODELS += [('small_20','additive'),('large_20','additive')]

def variant_root(variant,kind):
    return OUT/('additive_variants' if kind=='additive' else 'variants')/variant

def prepare_additive():
    """Make excluded, never-trained rows satisfy the legacy additive check."""
    excluded=set(read(OUT/'context_failure_audit.json')['excluded_state_indices'])
    for variant in ('small_20','large_20'):
        source=OUT/'variants'/variant;target=variant_root(variant,'additive')
        target.mkdir(parents=True,exist_ok=True)
        d=dict(np.load(source/'dataset.npz'))
        unused=np.isin(d['state_index'],list(excluded))
        assert np.all(d['split'][unused]=='unused')
        d['context'][:,unused,:10]=0.
        np.savez_compressed(target/'dataset.npz',**d)
        for name in ('entities.npz','pairs.json','protocol.json','controller_path_audit.json',
                     'alignment_audit_all.json','variant.json'):
            shutil.copy2(source/name,target/name)
    print({'prepared_additive_control':True,'excluded_input_quality_state_indices':sorted(excluded)})

def train(index):
    variant,kind=MODELS[index//3];seed=SEEDS[index%3]
    prior.OUT=variant_root(variant,kind)
    assert read(prior.OUT/'alignment_audit_all.json')['all_checks_passed']
    if kind=='context_only':
        original=prior.model_for
        prior.model_for=lambda requested: Critic(False,True,False) if requested=='context_only' else original(requested)
        try:prior.train(kind,seed)
        finally:prior.model_for=original
    else:prior.train(kind,seed)
    path=prior.OUT/'models'/kind/f'seed{seed}/summary.json'
    result=read(path)
    result['architecture']='same original physical encoder + eta encoder + context encoder + trunk; h masked by design' if kind=='context_only' else 'same original architecture'
    result['experiment_variant']=read(prior.OUT/'variant.json')
    write(path,result)

def compare():
    vals={}
    rows=[]
    for variant,kind in MODELS:
        for seed in SEEDS:
            r=read(variant_root(variant,kind)/'models'/kind/f'seed{seed}/summary.json')
            m=r['correct'];vals[(variant,kind,seed)]=r
            rows.append({'variant':variant,'kind':kind,'seed':seed,'TRAIN_families':r['experiment_variant']['TRAIN_states'],
                         'VAL_families':r['experiment_variant']['VAL_states'],
                         'horizon':20 if variant.endswith('20') else 80,
                         'selected_step':r['best_step'],'VAL_NLL':m['NLL'],'VAL_MAE':m['MAE'],
                         'oracle_eligible':m['eligible'],'B15_selected':m['selected_B15'],
                         'unknown_selected':m['selected_unknown'],'severe_false_positive':m['severe_false_positive'],
                         'selected_observed_Q':m['selected_observed_Q'],'observed_Q_regret':m['observed_Q_regret'],
                         'controller_reversal_correct':m['controller_reversals']['correct'],
                         'controller_reversal_total':m['controller_reversals']['total'],
                         'state_reversal_correct':m['state_reversals']['correct'],
                         'state_reversal_total':m['state_reversals']['total'],
                         'wrong_C_NLL':r['wrong_controller']['NLL'],
                         'wrong_C_B15':r['wrong_controller']['selected_B15'],
                         'state_shuffle_NLL':r['state_shuffle']['NLL'],
                         'state_shuffle_B15':r['state_shuffle']['selected_B15']})
    path=OUT/'source_validation_metrics.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paired=[]
    for seed in SEEDS:
        for a,b in [(('small_80','physical_context'),('small_20','physical_context')),
                    (('large_80','physical_context'),('large_20','physical_context')),
                    (('large_20','physical_context'),('small_20','physical_context')),
                    (('large_80','physical_context'),('small_80','physical_context')),
                    (('small_20','physical_context'),('small_20','eta_only')),
                    (('large_20','physical_context'),('large_20','eta_only')),
                    (('large_80','physical_context'),('large_20','eta_only')),
                    (('small_20','physical_context'),('small_20','context_only')),
                    (('small_80','physical_context'),('small_80','context_only')),
                    (('large_20','physical_context'),('large_20','context_only')),
                    (('large_80','physical_context'),('large_80','context_only')),
                    (('small_20','physical_context'),('small_20','additive')),
                    (('large_20','physical_context'),('large_20','additive')),
                    (('large_80','context_only'),('large_20','context_only'))]:
            aa=vals[(*a,seed)]['correct'];bb=vals[(*b,seed)]['correct']
            picks_a={(r['controller'],r['state_uid']):r for r in aa['picks']}
            picks_b={(r['controller'],r['state_uid']):r for r in bb['picks']}
            assert picks_a.keys()==picks_b.keys()
            a_rescue=sum(picks_a[k]['B15'] and not picks_b[k]['B15'] for k in picks_a)
            a_break=sum(picks_b[k]['B15'] and not picks_a[k]['B15'] for k in picks_a)
            families=sorted({k[1] for k in picks_a})
            effect=np.array([sum(int(picks_a[k]['B15'])-int(picks_b[k]['B15']) for k in picks_a if k[1]==uid) for uid in families])
            den=np.array([sum(k[1]==uid for k in picks_a) for uid in families])
            rng=np.random.default_rng(20261004)
            draw=rng.integers(0,len(families),(10000,len(families)))
            ci=np.quantile(effect[draw].sum(1)/den[draw].sum(1),[.025,.975])
            paired.append({'seed':seed,'model_A':a,'model_B':b,'A_rescue':int(a_rescue),'A_break':int(a_break),
                           'net':int(a_rescue-a_break),'family_bootstrap_95pct_rate_difference':ci.tolist(),
                           'NLL_difference_A_minus_B':aa['NLL']-bb['NLL'],
                           'same_candidates_and_VAL_labels':True})
    write(OUT/'paired_comparisons.json',paired)
    print({'models':len(rows),'VAL_eligible':rows[0]['oracle_eligible'],'comparisons':len(paired)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('train','compare','prepare_additive'))
    ap.add_argument('--index',type=int);a=ap.parse_args()
    train(a.index) if a.action=='train' else compare() if a.action=='compare' else prepare_additive()
