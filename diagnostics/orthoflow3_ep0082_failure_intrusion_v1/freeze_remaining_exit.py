from core import *
def main():
 d=json.load(open(H/'remaining_region_design_diagnostic.json'));r=d['best'][0];name='normal_escape4';cid='normal_escape4_region0';ee=[np.array(e) for e in r['eta']]
 assert d['current_new_eta']==135 and r['new_eta']==11
 defs=json.load(open(H/'corridor_definitions.json'));assert not any(x.get('corridor_id')==cid for x in defs)
 defs.append(dict(corridor_id=cid,region_id=0,purpose='Independent normal escape from already sampled representative-connected negative component',start_key=r['start_key'],status='FROZEN',ordered_keys=[key(e) for e in ee],endpoint_on_domain=True,maximum_segment_sample_gap=.05,selection_diagnostic='remaining_region_design_diagnostic.json'))
 rows=[dict(corridor_id=cid,region_id=0,edge=0,position=j,eta_key=key(e),eta=json.dumps(e.tolist()),outer_endpoint=j==len(ee)-1,round=name) for j,e in enumerate(ee)]
 write('candidate_failure_corridors.csv',read(H/'candidate_failure_corridors.csv')+rows);dump('corridor_definitions.json',defs)
 cost=plan(name,[dict(eta=e,role=f'corridor:{cid}:position={j}') for j,e in enumerate(ee)],'Distinguish unresolved failure pocket from a missed normal-direction exterior intrusion, after nearest-facet paths were interrupted.')
 dump(f'rounds/{name}/design_insufficiency_review.json',dict(d,chosen_rank=0,projected_total_new=146,additional_continuations=704,reason='A single independent normal-direction corridor is separated from observed positives by >=0.148 normalized distance; it tests a different escape direction rather than resubmitting a failed nearest-facet route.',no_scope_expansion='One state, one straight exit, max0.05 sample gap, exactQ64; no basin fitting or sweep. If unresolved after this test, report remaining topology ambiguity.'))
 print(json.dumps(cost))
if __name__=='__main__':main()
