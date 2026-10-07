#!/usr/bin/env python3
from families import *
def main(tag):
 inv=read(HERE/'exact_q64_inventory.csv');models=json.load(open(HERE/f'family_models_{tag}.json'));rows=[]
 for item in models:
  sid=item['state_id'];m=item['model'];rr=[r for r in inv if r['state_id']==sid];Z=np.array([(eta(r)-AFF)/SCALE for r in rr]);pos=np.array([r['B63']=='True' for r in rr]);full=contains(m,Z);ret=contains(m,Z,True)
  for idx in np.flatnonzero(full&~pos):
   nearest=int(np.argmin(np.linalg.norm(Z[pos]-Z[idx],axis=1)));pr=[r for r in rr if r['B63']=='True'][nearest]
   rows.append(dict(state_id=sid,family=m['family'],eta_key=rr[idx]['eta_key'],Q64=float(rr[idx]['Q64']),inside_retained=bool(ret[idx]),role=rr[idx]['geometry_role'],
    nearest_B63_key=pr['eta_key'],nearest_B63_distance=float(np.linalg.norm((eta(pr)-eta(rr[idx]))/SCALE)),failure_mode='deadlock' if int(rr[idx]['deadlock']) else 'timeout_or_other'))
 write(f'candidate_falsifications_{tag}.csv',rows,['state_id','family','eta_key','Q64','inside_retained','role','nearest_B63_key','nearest_B63_distance','failure_mode'])
 logs=HERE/'property_diagnosis_log.md'
 section=f'\n## {tag}\n\n'+f'{len(rows)} full-set cached negative inclusions across tested candidate instances; {sum(r["inside_retained"] for r in rows)} retained inclusions. These are counterexamples to these fitted instances, not impossibility results for their mathematical classes.\n'
 section+='The initial caps must extrapolate unmeasured exclusion boundaries. The conditional-line and failure-corridor acquisitions test whether missing structure is a boundary intrusion, interval splitting or an enclosed pocket. No new complex family is warranted before those observations return.\n'
 if not logs.exists():logs.write_text('# Property diagnosis and justified search history\n')
 text=logs.read_text()
 if f'## {tag}\n' not in text:logs.write_text(text+section)
 print('Falsifications',len(rows),'retained',sum(r['inside_retained'] for r in rows))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'initial')
