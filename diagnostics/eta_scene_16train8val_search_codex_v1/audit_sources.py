from common import *
import subprocess
FOLDERS=['shared_control','single_integrator','bottleneck_family','shared_rollout_db/src']
def snapshot():
 files={str(p):sha(p)for f in FOLDERS for p in(ROOT/f).rglob('*.py')}
 files.update({str(ROOT/'AGENTS.md'):sha(ROOT/'AGENTS.md'),str(ROOT/'docs/exact_engineering_acceleration_20261026.md'):sha(ROOT/'docs/exact_engineering_acceleration_20261026.md')})
 for r in read(HERE/'TOOLING_COPY.receipt.json')['files']:files[r['source']]=sha(r['source'])
 return dict(utc=utc(),files=files,tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT)).hexdigest(),prospective_stat=list(os.stat(ROOT/'diagnostics/cross_scene_prospective_v1')))
if __name__=='__main__':
 if sys.argv[1]=='start':write(HERE/'SOURCE_AUDIT_START.json',snapshot())
 else:
  old=read(HERE/'SOURCE_AUDIT_START.json');new=snapshot();changed={p:dict(before=h,after=new['files'].get(p))for p,h in old['files'].items()if new['files'].get(p)!=h};passed=not changed and old['tracked_diff_sha256']==new['tracked_diff_sha256'];write(HERE/'SOURCE_AUDIT_END.json',new);write(HERE/'INTEGRITY_COMPARE.json',dict(passed=passed,changed_inputs=changed,tracked_diff_unchanged=old['tracked_diff_sha256']==new['tracked_diff_sha256'],prospective_stat_only=True));assert passed
