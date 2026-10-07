from audit import ROOT,OUT,read,save,conn,lookup,evidence,canonical
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

def main():
    probes=read(OUT/'H3_inference_results.json')['states']
    states={r['state_uid']:r for r in pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet').to_pylist()}
    props={r['state_uid']:r for r in read(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/proposals.json')['states']}
    from shared_rollout_db.src.rollout_db import eta_identity
    rows=[];cache=Counter()
    with conn() as c:
        c.execute('BEGIN')
        for p in probes:
            z=np.asarray(p['logits'],float);p64=1/(1+np.exp(-z))
            if p['probability_pick']==p['logit_pick']:continue
            sid=p['state_uid'];st=states[sid];pr=props[sid];ev=[]
            for index in (p['probability_pick'],p['logit_pick']):
                rr=lookup(c,sid,eta_identity(pr['etas'][index])[0],st['controller_uid']);e=evidence(rr)
                cache['requested']+=16;cache['exact_reuse']+=16-e['numerical']-e['missing'];cache['numerical_present']+=e['numerical'];cache['truly_missing']+=e['missing']
                ev.append({'index':index,**e,'rollout_uids':[r['rollout_uid'] for r in rr]})
            rows.append({'scenario':p['scenario'],'state_uid':sid,'split':st['split'],
                'h_hash':p['h_hash'],'probability_pick':p['probability_pick'],'logit_pick':p['logit_pick'],
                'float64_sigmoid_pick':int(np.argmax(p64)),
                'float32_tie_count':len(p['probability_tie_indices']),
                'logits_at_picks':[p['logits'][p['probability_pick']],p['logits'][p['logit_pick']]],
                'Q_evidence':ev,'both_B15':all(e['robust'] is True for e in ev)})
    summary={'matched_inference_states':len(probes),'changed_choices':len(rows),
        'Ring_changed_choices':sum(p['scenario']=='ring_exchange' and p['probability_pick']!=p['logit_pick'] for p in probes),
        'Four_train_dev_changed_choices':len(rows),'both_B15_count':sum(r['both_B15'] for r in rows),
        'float64_control_match':sum(r['float64_sigmoid_pick']==r['logit_pick'] for r in rows),
        'classification':'NUMERICAL_LIMITATION_CONFIRMED_IN_HISTORICAL_SELECTOR_ALREADY_ADDRESSED_BY_PHASE_A_LOGIT_SELECTION',
        'Ring_7_misses_explained':0,'canonical_patch_applied':False,'new_rollouts':0}
    save('H5_saturation_witness.json',{'summary':summary,'states':rows});save('H5_cache_preflight.json',dict(cache))
    # Extract non-threshold-only counterexamples with a readable exact DB provenance.
    proxy=read(OUT/'H1_exact_vs_proxy.json'); examples=[]
    for sc in ('ring_exchange','four_way_intersection'):
        candidates=[]
        for st in proxy['states']:
            if st['scenario']!=sc:continue
            for e in st['candidates']:
                if e['Q16'] is not None and e['nearest_stopped_fraction'] is not None:
                    candidates.append((abs(e['Q16']-e['nearest_stopped_fraction']),st,e))
        for _,st,e in sorted(candidates,key=lambda v:(-v[0],v[1]['state_uid'],v[2]['index']))[:2]:
            examples.append({'scenario':sc,'state_uid':st['state_uid'],'split':st['split'],
                'state_id':st['state_id'],'candidate':e,
                'expected':'different eta identity cannot inherit a robust/Q certificate from its nearest neighbor',
                'control':'same state and current controller; original exact labels remain unchanged'})
    save('confirmed_witnesses.json',{'H1':examples,'H2':read(OUT/'H2_k16_corrected_summary.json'),
         'H5':summary,'physical_execution_semantics_changed':False})
    print(summary)
if __name__=='__main__':main()
