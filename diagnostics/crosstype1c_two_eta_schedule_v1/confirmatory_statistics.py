"""Five frozen endpoints; evaluation cannot change any model or fallback."""
from common import *
from statistics_pipeline import evaluate_group,equivalence
def main():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    data['uid_class']=equivalence(data['uids']);types=data['types'];reports={}
    for group in [*sorted(set(types)),'combined']:
        mask=np.ones(len(types),bool) if group=='combined' else types==group;subset={k:v[mask] for k,v in data.items()};selected={k:v[mask] for k,v in methods.items()};reports[str(group)]={'type_equal':evaluate_group(subset,selected,True,5000)}
        if group=='combined':reports[str(group)]['controller_equal']=evaluate_group(subset,selected,False,5000)
    summary=reports['combined']['type_equal'];pair=summary['paired_comparisons'];primary=pair['pipeline__minus__best_human_schedule'];frozen=read(HERE/'FROZEN_PIPELINE.json');records=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];switches={};base=read(HERE/'BASE_PROTOCOL.json');eta13=np.asarray(base['eta'][13]);eta10=np.asarray(base['eta'][10]);direction=eta10-eta13;descriptions=[]
    for kind in sorted(set(types)):
        switches[str(kind)]={}
        for method in ('self_schedule','fixed500','pipeline'):
            aa=[r['methods'][method] for r in records if r['type']==kind];diagonal=sum(np.array_equal(a['eta_a'],a['eta_b']) for a in aa);counts={str(t):sum(not np.array_equal(a['eta_a'],a['eta_b']) and a['switch_step']==t for a in aa) for t in (200,300,400,500,600)};switches[str(kind)][method]=dict(diagonal_no_effective_switch=diagonal,non_diagonal_counts=counts,total=len(aa),all_effective_switches500=bool(sum(counts.values())) and counts['500']==sum(counts.values()))
    for r in records:
        a=r['methods']['pipeline'];early=np.asarray(a['eta_a']);late=np.asarray(a['eta_b']);delta=late-early;norm=float(np.linalg.norm(delta));cos=float(np.clip(np.dot(delta,direction)/(norm*np.linalg.norm(direction)),-1,1)) if norm>0 else None
        descriptions.append(dict(controller=r['controller'],type=r['type'],state_index=r['state_index'],eta_early=early,eta_late=late,switch_step=a['switch_step'] if norm>0 else None,diagonal=norm==0,distance_early_eta13=float(np.linalg.norm(early-eta13)),distance_early_eta10=float(np.linalg.norm(early-eta10)),distance_late_eta13=float(np.linalg.norm(late-eta13)),distance_late_eta10=float(np.linalg.norm(late-eta10)),direction_cosine=cos))
    endpoints=dict(endpoint1=dict(**primary,primary_method=frozen['primary_method'],noninferior=primary['CI95_pp'][0]>-1,superior=primary['CI95_pp'][0]>0,versus_step500=pair['pipeline__minus__all_S500'],versus_progress=pair['pipeline__minus__all_Sprog']),endpoint2=dict(**pair['self_schedule__minus__fixed500'],switch_distributions=switches),endpoint3=dict(**pair['pipeline__minus__no_schedule'],passed=pair['pipeline__minus__no_schedule']['CI95_pp'][0]>0,selected_non_diagonal_fraction=summary['methods']['pipeline']['selected_non_diagonal_fraction']),endpoint4=pair['pipeline__minus__legacy_pipeline'],endpoint5=dict(descriptive=True,rows=str(HERE/'SELECTED_ETA_DESCRIPTIONS.json'),diagonal_direction_undefined=sum(r['diagonal'] for r in descriptions)))
    write(HERE/'SELECTED_ETA_DESCRIPTIONS.json',dict(utc=utc(),rows=descriptions,human_reference_only=dict(eta13=eta13,eta10=eta10)))
    write(HERE/'CONFIRMATORY_RESULTS.json',dict(utc=utc(),evidence='CONFIRMATORY_HELD_TYPE_NEW_STATE',reports=reports,endpoints=endpoints,primary_method=frozen['primary_method'],no_controller_comparison=pair['pipeline__minus__no_controller'],all_types_reference_evidence='SEEN_TYPES_UNSEEN_STATES_REFERENCE',legacy_pretraining_caveat='Original cross-type1b warm branch previously saw native Flow; this comparator retains its original pretraining range. New schedule critics use scratch only.',oracle_scope='Only actually evaluated non-nominal actions; not all320 candidates',bootstrap_replicates=5000,new_states=24,controllers=62,types=9))
    seal('CONFIRMATORY_RESULTS.receipt.json',[HERE/n for n in ('CONFIRMATORY_RESULTS.json','SELECTED_ETA_DESCRIPTIONS.json','confirmatory_statistics.py','statistics_pipeline.py','NEW_EVAL_BUNDLE.receipt.json','NEW_EVAL_CHOICES.receipt.json')],confirmation_not_used_for_retuning=True)
    print(json.dumps(dict(primary=endpoints['endpoint1'],schedule_used=endpoints['endpoint3'],versus_legacy=endpoints['endpoint4']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
