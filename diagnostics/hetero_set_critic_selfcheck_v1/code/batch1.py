import json,run_any
from run import KEYS
if __name__=='__main__':
 S=json.load(open('scenes_set.json'))+json.load(open('scenes_test.json'));json.dump(S,open('scenes_all.json','w'))
 PT=json.load(open('GEN_ENS_LOSO_TRAIN_PROPOSALS.json'));PV=json.load(open('GEN_ENS_VAL_PROPOSALS.json'));TE=json.load(open('scenes_test.json'))
 tasks=[('ens_train.jsonl',v,ci,si,f'p{k}',e) for v,P in PT.items() for k,e in enumerate(P) for ci in range(3) for si in range(16)]
 tasks+=[('ens_val.jsonl',v,ci,si,f'p{k}',e) for v,g in PV.items() for k,e in enumerate(g['proposals']) for ci in range(3) for si in range(8,24)]
 tasks+=[('test_gate.jsonl',s['name'],ci,si,'menu:'+K,None) for s in TE for ci in range(3) for si in range(8) for K in KEYS]
 run_any.go(tasks,23)
