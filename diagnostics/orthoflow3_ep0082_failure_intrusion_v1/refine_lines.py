#!/usr/bin/env python3
from core import *
def main():
 number=int(sys.argv[1]);assert 1<=number<=4
 assert all(r['complete']=='True' for r in read(H/'transition_statistics.csv')),'Wait for registered line outcomes'
 brackets=[r for r in read(H/'opposite_label_brackets.csv') if float(r['width'])>.0100000001]
 brackets.sort(key=lambda r:(r['repeated_line']!='True',int(r['region_id']),r['line_id'],-float(r['width']),int(r['transition'])))
 name=f'refine{number}';lookup={r['eta_key']:r for r in read(H/'exact_q64_ep0082.csv')};manifest=read(H/'conditional_slice_manifest.csv');defs={r['line_id']:r for r in json.load(open(H/'line_definitions.json'))};requests=[];newrows=[]
 for b in brackets:
  e=(eta(lookup[b['left_key']])+eta(lookup[b['right_key']]))/2;k=key(e);d=defs[b['line_id']];origin=(np.array(d['origin'])-AFF)/SCALE;t=float(((e-AFF)/SCALE-origin)@np.array(d['direction']))
  if any(r['line_id']==b['line_id'] and r['eta_key']==k for r in manifest):continue
  newrows.append(dict(line_id=b['line_id'],region_id=d['region_id'],axis=b['line_id'].split('_e')[-1],t=t,eta_key=k,eta=json.dumps(e.tolist()),round=name))
  requests.append(dict(eta=e,role=f'line:{b["line_id"]}:t={t:.17g}:opposite_bracket_width={float(b["width"]):.17g}'))
 if not requests:dump(f'rounds/{name}/no_acquisition.json',dict(reason='All registered opposite-label brackets at or below0.01; no further bisection necessary'));print('No refinement needed');return
 already={r['eta_key'] for r in read(H/'adaptive_probe_manifest.csv') if r['cached_Q64']=='False'};new={key(r['eta']) for r in requests if key(r['eta']) not in lookup};projected=len(already|new)
 if projected>100:
  dump(f'rounds/{name}/above100_diagnostic.json',dict(previous_new_eta=len(already),additional_eta=len(new),projected_total=projected,unresolved_question='Do the identified exact opposite-label brackets retain a single sharp transition down to0.01, or reveal additional robust alternation?',brackets=brackets,justification='Midpoints only on registered exact lines, no grid/state expansion; max4bisections per original short bracket. Count remains below150 diagnostic threshold.'))
 assert projected<=150,'Full design-insufficiency review required before approaching/exceeding150'
 write('conditional_slice_manifest.csv',manifest+newrows)
 cost=plan(name,requests,'Refine the registered exact opposite-label intervals toward0.01 and detect hidden additional transitions without new lines/states.')
 print(json.dumps(dict(round=name,projected_new_eta_total=projected,cost=cost)))
if __name__=='__main__':main()
