import json,hashlib,shutil,sys
from pathlib import Path
root=Path('/home/zhihan/research/Basin_C1'); audit=Path('/mnt/ssd/zhihan/Basin_C1/audits/storage_20261110')
verified=[]
for name in ['storage_migration_20261110.json','storage_migration_20261110_pip.json']:
 plan=json.loads((root/'docs'/name).read_text()); ar=Path(plan['audit_root'])
 overall=json.loads((ar/'summary.json').read_text()); assert overall['status']=='complete'
 for rel in plan['directories']:
  src=Path(plan['source_root'])/rel; dst=Path(plan['destination_root'])/rel
  summary=json.loads((ar/rel/'summary.json').read_text())
  assert summary['status']=='complete' and src.is_symlink() and src.resolve()==dst
  assert not src.with_name(src.name+'.migration-local-copy').exists()
  with (ar/rel/'sha256.jsonl').open() as f:
   sample=json.loads(next(f)); count=1+sum(1 for _ in f)
  assert count==summary['files']==summary['verified_sha256_files']
  h=hashlib.sha256()
  with (src/sample['path']).open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
  assert h.hexdigest()==sample['sha256']
  verified.append({'source':str(src),'files':count,'bytes':summary['bytes'],'original_path_hash_sample':'passed'})
sql=json.loads((audit/'sqlite/summary.json').read_text()); assert sql['status']=='complete'
for n in sql['sha256']:
 p=root/'shared_rollout_db'/n; assert p.is_symlink() and p.resolve()==Path(sql['destination'])/n
 assert not p.with_name(p.name+'.migration-local-copy').exists()
for n in ['results','pip','sqlite']: assert (audit/(n+'.exitcode')).read_text().strip()=='0'
before=json.loads((audit/'preflight.json').read_text())
after={d:shutil.disk_usage(d)._asdict() for d in ['/','/mnt/ssd','/mnt/hdd']}
result={'status':'passed','directories':len(verified),'verified_files':sum(r['files'] for r in verified)+sql['files'],'content_bytes':sum(r['bytes'] for r in verified)+sql['bytes'],'system_free_gain_bytes':after['/']['free']-before['disk_usage']['/']['free'],'disk_usage_after':after,'directory_checks':verified,'sqlite':sql,'note':'Every file SHA256 verified during migration; final checks verify every link and manifest count plus one original-path SHA256 sample per directory. Migration is not an independent backup.'}
(audit/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['directory_checks','sqlite']},indent=2))
