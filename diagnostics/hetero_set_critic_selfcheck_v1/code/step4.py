import json,run_any
from run import KEYS
if __name__=='__main__':
 Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test']
 tasks=[('test_menu.jsonl',v,ci,si,'menu:'+K,None) for v in TE for ci in range(3) for si in range(8,24) for K in KEYS]
 tasks+=[('test_props.jsonl',v,ci,si,f'p{k}',e) for v in TE for k,e in enumerate(Z['choices'][v]['proposals']) for ci in range(3) for si in range(8,24)]
 run_any.go(tasks,23)
