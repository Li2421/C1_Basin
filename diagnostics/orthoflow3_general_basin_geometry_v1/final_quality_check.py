#!/usr/bin/env python3
from audit import *
from families import contains
from evidence_roles import validation_memberships
import xml.etree.ElementTree as ET
def main():
 inv=read(HERE/'exact_q64_inventory.csv');lookup={(r['state_id'],r['eta_key']):r for r in inv};assert len(lookup)==len(inv)
 for r in inv:
  assert int(r['trials'])==64 and (r['B63']=='True')==(int(r['successes'])>=63)
  assert abs(float(r['Q64'])-int(r['successes'])/64)<1e-14
  assert inside((eta(r)-AFF)/SCALE)[0]==(r['in_E_bridge']=='True')
 outside=[r for r in inv if r['in_E_bridge']=='False'];assert len(outside)==3 and all(r['phases']=='test64' for r in outside)
 roles=validation_memberships()
 assert len(roles)==402
 assert all(k in lookup and set(lookup[k]['retained_validation_batches'].split(';'))==v for k,v in roles.items())
 assert all(lookup[k]['in_E_bridge']=='True' for k in roles)
 primary={k for k,r in lookup.items() if r['geometry_role']=='heldout' and 'retained_validation' not in r['phases']}
 assert not primary.intersection(roles)
 Z=np.random.default_rng(2026092901).uniform([-7/6,-.5,-.5],[.5,.5,.5],(10000,3))
 models=json.load(open(HERE/'family_models_minimal_polyhedral_refined_r7_stable.json'))
 for r in models:assert np.all(~contains(r['model'],Z,True)|contains(r['model'],Z))
 g=json.load(open(HERE/'representation_gate.json'));s=json.load(open(HERE/'scientific_saturation_audit.json'))
 assert abs(g['corrected_nonfitting_positive_metrics']['median_retained_recall']-s['rounds'][-1]['retained_recall_fixed_holdout'])<1e-12
 assert abs(g['corrected_nonfitting_positive_metrics']['median_full_recall']-s['rounds'][-1]['full_recall_fixed_holdout'])<1e-12
 for p in (HERE/'figures').glob('*.svg'):ET.parse(p)
 manifest=json.load(open(HERE/'manifest.json'));conflict=[n for n,h in manifest['files'].items() if sha(HERE/n)!=h]
 # This checker may have been added after the previous manifest; final regeneration follows.
 assert not conflict,conflict
 dump('final_quality_checks.json',dict(exact_inventory_rows=len(inv),B63_threshold_correct=True,eta_domain_flags_correct=True,historical_out_of_domain_records_excluded_from_geometry=3,all_new_validation_in_domain=True,exact_dedup=True,all402_validation_tuples_classified=True,validation_excluded_from_primary_holdout=True,retained_subset_full_random_checks=120000,corrected_gate_and_saturation_metrics_agree=True,SVG_XML_valid=True,manifest_hashes_consistent=True,training_runs=0))
 print('Final offline checks passed; no scientific rollout/training executed.')
if __name__=='__main__':main()
