from common import *
stage=sys.argv[1];p=read(HERE/f'{stage}_protocol.json');cases=read(HERE/f'{stage}_CASES.json');rates={};eligible=[];ambiguous=[];excluded=[]
for name in p['controller_order']:
 rates[name]={}
 for arm in ['F10','S500']:
  rr=[read(HERE/stage/'raw'/f"{c['rollout_uid']}.json") for c in cases if c['base_controller_name']==name and c['arm']==arm];assert len(rr)==16;lo=sum(r['success'] for r in rr)/16;hi=lo+sum(r['numerical_failure'] for r in rr)/16;rates[name][arm]=[lo,hi]
 a,b=rates[name]['F10'],rates[name]['S500'];reject=(a[1]<.15 and b[1]<.15) or (a[0]>.98 and b[0]>.98);uncertain=((a[0]<.15 and b[0]<.15) or (a[1]>.98 and b[1]>.98)) and not reject
 if reject:excluded.append(name)
 elif uncertain:ambiguous.append(name)
 else:eligible.append(name)
bytype={t:[n for n in eligible if p['types'][n]==t] for t in dict.fromkeys(p['types'].values())};write(HERE/f'{stage}_CAPABILITY.json',dict(utc=utc(),eligible=eligible,excluded=excluded,ambiguous=ambiguous,rates=rates,eligible_by_family=bytype,family_status={t:'PASS' if len(v)>=4 else 'INCONCLUSIVE' for t,v in bytype.items()},no_replacement_or_tuning=True));print({k:len(v) for k,v in bytype.items()})
