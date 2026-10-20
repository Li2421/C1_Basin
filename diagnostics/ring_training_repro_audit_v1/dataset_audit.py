from pathlib import Path
import json,hashlib,datetime,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];p=ROOT/'diagnostics/ring_exchange_stage1/base_u_v2_dataset';manifest=json.loads((p/'manifest.json').read_text());cfg=json.loads((ROOT/'diagnostics/ring_exchange_stage1/base_u_v2_macflow/config.json').read_text());assert (HERE/'base_u_v2_macflow_STEP1.json').exists();out=[]
for row in manifest['files']:
 f=p/row['file'];raw=hashlib.sha256(f.read_bytes()).hexdigest();h=hashlib.sha256()
 with np.load(f,allow_pickle=False) as z:
  for k in ('states','observations','actions'):
   a=np.ascontiguousarray(z[k]);h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
  meta=json.loads(str(z['metadata_json'].item()));length=len(z['actions'])
 out.append(dict(file=str(f),sha256=raw,manifest_file_sha256_available='sha256' in row,trajectory_digest=h.hexdigest(),digest_matches_manifest=h.hexdigest()==row['trajectory_digest'],digest_matches_metadata=h.hexdigest()==meta['trajectory_digest'],length=length,mtime_utc=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat(),mtime_after_original_day=f.stat().st_mtime>datetime.datetime(2026,10,1,tzinfo=datetime.timezone.utc).timestamp()))
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),'original_run_manifest_sha256':cfg['dataset_manifest_sha256'],'manifest_matches_original_config':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==cfg['dataset_manifest_sha256'],'files':out,'file_count':len(out),'all_content_digest_match':all(x['digest_matches_manifest'] and x['digest_matches_metadata'] for x in out),'raw_file_sha_not_in_manifest':True,'mtime_after_original_day_count':sum(x['mtime_after_original_day'] for x in out),'note':'mtime is not proof of changed content; trajectory digests cover states/observations/actions, original raw archive SHA unavailable'};(HERE/'STEP2_DATASET.json').write_text(json.dumps(r,indent=2)+'\n');print(r['file_count'],r['all_content_digest_match'],r['manifest_matches_original_config'],r['mtime_after_original_day_count'])
