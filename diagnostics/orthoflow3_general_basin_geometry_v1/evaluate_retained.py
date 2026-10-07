#!/usr/bin/env python3
from audit import *
from families import contains
def main():
 name=sys.argv[1];root=HERE/'targeted_probe_rounds'/name;manifest=read(root/'manifest.csv');models={r['state_id']:r['model'] for r in json.load(open(root/'frozen_parameters.json'))};lookup={(r['state_id'],r['eta_key']):r for r in read(HERE/'exact_q64_inventory.csv')};out=[]
 for p in manifest:
  sid=p['state_id'];m=models[sid];v=np.array(json.loads(p['eta']));assert hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()==p['model_sha256'];assert contains(m,(v-AFF)/SCALE,True)[0]
  r=lookup.get((sid,key(v)));out.append(dict(state_id=sid,eta_key=key(v),eta=json.dumps(v.tolist()),Q64=r['Q64'] if r else '',B63=r['B63'] if r else '',deadlock=r['deadlock'] if r else '',timeout=r['timeout'] if r else '',collision=r['collision'] if r else '',exact_available=r is not None,confirmed_false_inclusion=r is not None and r['B63']=='False',model_sha256=p['model_sha256']))
 write(f'targeted_probe_rounds/{name}/results.csv',out);false=sum(r['confirmed_false_inclusion'] for r in out);complete=all(r['exact_available'] for r in out)
 result=dict(family=next(iter(models.values()))['family'],expected=len(out),exact_Q64=sum(r['exact_available'] for r in out),false_inclusions=false,complete=complete,
  status='REJECTED' if false else 'PASSED_THIS_FROZEN_BATCH_ONLY' if complete else 'PENDING',generalized_gate_pass=False)
 dump(f'targeted_probe_rounds/{name}/gate.json',result);print(json.dumps(result))
if __name__=='__main__':main()
