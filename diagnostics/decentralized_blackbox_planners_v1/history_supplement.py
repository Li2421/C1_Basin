from common import *
G=guard()
new={n:read(HERE/f'{n}_STATES.json') for n in ['EVAL','PILOT','TUNING','PROBE']};files=[D/'blackbox_families_v1/main_protocol.json',D/'blackbox_addendum_v1/right_hand_evaluation_protocol.json',D/'blackbox_addendum_v1/right_hand_tune_protocol.json'];rows=[]
for f in files:
 p=read(f);states=p['states'];overlap={k:sum(any(x.get('physical')==y.get('physical') and x['uid']==y['uid'] and x.get('rng_namespace')==y.get('rng_namespace') for y in states) for x in v) for k,v in new.items()};rows.append(dict(path=str(f),sha256=sha(f),state_and_rng_overlap=overlap,compatibility='INCOMPATIBLE',reason='Round37/38 separate state/RNG namespaces and centralized TT/FF safety/other nominal architectures. No local distributed safety trajectory or exact nominal request is reusable. Read for compatibility only; no imports, weights, or labels reused.'))
assert all(all(v==0 for v in r['state_and_rng_overlap'].values()) for r in rows)
write(HERE/'HISTORY_BROAD_SEARCH_SUPPLEMENT.json',dict(utc=utc(),before_main_evaluation=True,records=rows,compatible_physical_rollouts=0,guard=guard()))
