from core import *
def main(k):
 rows=read(HERE/f'wave{k}_MISSING_CASES.json');raw=HERE/f'wave{k}'/'raw';journaled=HERE/f'wave{k}'/'journaled';missing=[c['rollout_uid'] for c in rows if not (raw/f"{c['rollout_uid']}.json").exists() or not (journaled/f"{c['rollout_uid']}.json").exists()]
 worker=[read(HERE/f'wave{k}'/f'worker_{i}.json') for i in range(18)];assert not missing and all(x['complete'] for x in worker),(k,len(missing))
 assert all(read(journaled/f"{c['rollout_uid']}.json")['raw_sha256']==sha(raw/f"{c['rollout_uid']}.json") for c in rows)
 elapsed=[x['episode_cpu_seconds']/x['new'] for x in worker if x['new']]
 write(HERE/f'wave{k}_VERIFY.json',dict(utc=utc(),wave=k,complete=True,requests=len(rows),missing=0,workers=18,mean_seconds=float(np.mean(elapsed)),total_new=sum(x['new'] for x in worker),worker_reports=worker))
 print('wave',k,'complete',len(rows))
if __name__=='__main__':main(int(sys.argv[1]))
