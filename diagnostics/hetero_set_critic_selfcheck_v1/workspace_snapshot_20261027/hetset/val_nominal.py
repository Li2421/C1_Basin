"""Nominal only (no safety projection) for the 7 capable classical controllers on the 8 frozen validation scenes, states 8-23."""
import json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run8
if __name__=='__main__':
 V=json.load(open('FINAL_FROZEN.json'))['val'];out='val_nominal.jsonl'
 tasks=[(v,ci,si,'nominal',None) for v in V for ci in [0,1,2,4,5,6,7] for si in range(8,24)]
 with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=run8._init) as ex,open(out,'w') as f:
  for fu in as_completed([ex.submit(run8._run,t) for t in tasks]):f.write(json.dumps(fu.result())+'\n')
 print('done',len(tasks))
