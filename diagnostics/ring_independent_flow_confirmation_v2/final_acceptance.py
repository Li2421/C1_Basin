from core import *
import subprocess
prereg();labels_allowed();gate();p=read(HERE/'protocol.json');r=read(HERE/'results.json');workers=[read(q) for q in (HERE/'main').glob('worker_*.json')];assert len(workers)==23 and all(q['complete'] for q in workers) and sum(q['new'] for q in workers)==21504
assert sha(HERE/'core.py')==p['core_sha256'] and sha(HERE/'run.py')==p['io_binding_sha256'] and sha(R9/'run.py')==p['runner_sha256']
assets=read(HERE/'HISTORICAL_ASSET_INDEX.json')['assets'];assert all(sha(path)==h for path,h in assets.items())
for row in read(HERE/'SOURCE_SEMANTICS_AUDIT.json')['checks']:assert sha(row['path'])==row['sha256']
for ctl in read(HERE/'FINAL_BASE_PROFILES.json')['profiles'].values():assert sha(ctl['payload']['flow_checkpoint_path'])==ctl['payload']['flow_checkpoint_sha256']
opening=read(HERE/'ENDPOINT_OPENING.receipt.json');assert opening['records']==21504
start=[read(q) for q in (HERE/'main').glob('START_AUTHORIZATION_*.json')];assert len(start)==23;freeze_time=read(HERE/'SELECTOR_FROZEN.receipt.json')['utc'];assert all(q['utc']>freeze_time for q in start)
from shared_rollout_db.src.rollout_db import connect
with connect(True) as db:
 count=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];unknown=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=? AND numerical_failure=1',(EXP,)).fetchone()[0];probe_success=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP+'_unlabeled',)).fetchone()[0]
assert count==21504 and unknown==r['unknown'] and probe_success==0
post=read(HERE/'OFFICIAL_POSTFLIGHT.json');assert post['actual_unobserved_requests']==0 and post['known_unknown_retained']==unknown
videos=read(HERE/'videos/manifest.json');checked=[]
for v in videos['videos']:
 assert sha(v['file'])==v['sha256'];actual=int(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','default=nokey=1:noprint_wrappers=1',v['file']]));assert actual==v['frames'];checked.append(dict(file=v['file'],sha256=v['sha256'],actual_frames=actual))
assert not (HERE/'COMPLETE').exists() if not videos['complete'] else True
files=['REPORT.md','HANDOFF.md','results.json','protocol.json','R5_FROZEN.receipt.json','SELECTOR_FROZEN.receipt.json','TRAINING_REPRODUCTION_GATE.json','OFFICIAL_POSTFLIGHT.json','videos/manifest.json']
write(HERE/'FINAL_ACCEPTANCE.receipt.json',dict(utc=utc(),passed=True,numerical_complete=True,video_complete=videos['complete'],main_new_rollouts=21504,gate_new_rollouts=0,gate_exact_reused=32,unknown=unknown,unlabeled_short_probes=448,unlabeled_success_rows=probe_success,worker_new_sum=21504,old_assets_SHA_unchanged=True,postflight_all_records_present=True,video_checks=checked,files={f:sha(HERE/f) for f in files}))
print('Final acceptance: numerical complete; video complete=',videos['complete'])
