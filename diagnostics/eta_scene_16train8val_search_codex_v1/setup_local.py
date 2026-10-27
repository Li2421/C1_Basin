from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];S=R/'diagnostics/critic_pool_supplement_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):(H/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
files=['experiment.py','bootstrap.py','common.py','execution.py','runtime_new.py','controllers_new.py','scene_environment.py','scenes.py','design.py','rollout_worker.py','MOVING_USER_FIX_UNIT_GATE.json','LAYOUT_REVISION_DESIGN.json','CIRCLE_SPEED.json']
rows=[]
for f in files+[str(p.relative_to(S))for p in (S/'backend').glob('*') if p.is_file()]+[str(p.relative_to(S))for p in (S/'native_resume').glob('*')if p.is_file()]+[p.name for p in S.glob('*.so')]:
 src=S/f;dst=H/f;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,dst);rows.append(dict(source=str(src),target=str(dst),source_sha256=sha(src),copied_sha256=sha(dst)))
for f in ['common.py','backend/reservation_certificate_v2_pa_memo_query_disk_candidate_spatial_mask_20261022.py','native_resume/cbs_kernel_query_disk_candidate_20261022.cpp']:
 p=H/f
 if p.exists():p.write_text(p.read_text().replace('exp_critic_pool_supplement_v1','exp_eta_scene_16train8val_search_codex_v1').replace(str(S),str(H)))
# Every absolute private path must target this round. Shared paths remain read-only imports.
for p in list((H/'backend').glob('*.py'))+list((H/'native_resume').glob('*.cpp')):
 s=p.read_text();t=s.replace(str(S),str(H))
 if s!=t:p.write_text(t)
write('TOOLING_COPY.receipt.json',dict(files=rows))
orig=json.loads((S/'BACKEND_BUILD.json').read_text())['sources'];graph=[]
for r in orig:
 src=Path(r['target']);dst=H/src.relative_to(S)
 if dst.exists():graph.append(dict(source=str(src),source_sha256=sha(src),target=str(dst),target_sha256=sha(dst)))
write('BACKEND_BUILD.json',dict(sources=graph,private_copy=True))
write('REUSED_EVIDENCE_SHA256_START.json',dict(files={r['source']:r['source_sha256']for r in rows}))
write('EVIDENCE_ORIGIN.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),parallel_work_read_only='/home/zhihan/.claude/jobs/8061f3d8/tmp/hetset',own_cpu_workers_max=8,total_user_cpu_limit=18,prospective_stat_only=True))
# Same C++ import shim, relocated cache root, no controller search will be executed.
cmd=['g++','-O3','-std=c++17','-shared','-fPIC',str(H/'native_resume/cbs_kernel_query_disk_candidate_20261022.cpp'),'-o',str(H/'native_resume/cbs_kernel_resume.so'),'-lcrypto']
subprocess.run(cmd,check=True);write('NATIVE_BUILD.json',dict(command=cmd,sha256=sha(H/'native_resume/cbs_kernel_resume.so')))
print('Copied',len(rows),'execution dependencies')
