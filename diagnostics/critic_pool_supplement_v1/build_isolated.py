"""Copy provenance is in TOOLING_COPY; compile only the import-required native library."""
from pathlib import Path
import json,hashlib,subprocess,os
H=Path(__file__).resolve().parent;R=H.parents[1];V=R/'diagnostics/hetero_scene_pipeline_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=json.loads((H/'TOOLING_COPY.receipt.json').read_text())['files']
 for row in rows:
  assert sha(row['source'])==row['source_sha256'];row['target_sha256']=sha(row['target']);row['changed']=row['source_sha256']!=row['target_sha256']
 (H/'TOOLING_COPY.receipt.json').write_text(json.dumps(dict(files=rows),indent=2)+'\n')
 graph_names={Path(r['target']).relative_to(V).as_posix()for r in json.loads((V/'BACKEND_BUILD.json').read_text())['sources']if Path(r['target']).is_relative_to(V)}
 graph=[r for r in rows if Path(r['target']).relative_to(H).as_posix()in graph_names]
 (H/'tmp').mkdir(exist_ok=True);command=['g++','-O3','-std=c++17','-shared','-fPIC',str(H/'native_resume/cbs_kernel_query_disk_candidate_20261022.cpp'),'-o',str(H/'native_resume/cbs_kernel_resume.so'),'-lcrypto']
 subprocess.run(command,check=True,env=dict(os.environ,TMPDIR=str(H/'tmp')))
 (H/'BACKEND_BUILD.json').write_text(json.dumps(dict(sources=graph,native_command=command,native_sha256=sha(H/'native_resume/cbs_kernel_resume.so'),native_only_change='cache root; runtime_new imports CDLL eagerly, no search rollout'),indent=2)+'\n')
 d=json.loads((H/'REUSED_EVIDENCE_SHA256_START.json').read_text());d['files'].update({r['source']:r['source_sha256']for r in rows})
 for path in [V/'BACKEND_BUILD.json',V/'REPORT.md',V/'COMPLETE.json',V/'INTEGRITY_COMPARE.json']:
  d['files'][str(path)]=sha(path)
 for folder in ['shared_control','single_integrator','bottleneck_family','shared_rollout_db/src']:
  for path in (R/folder).glob('*.py'):d['files'][str(path)]=sha(path)
 (H/'REUSED_EVIDENCE_SHA256_START.json').write_text(json.dumps(d,indent=2)+'\n');(H/'CIRCLE_SPEED.receipt.json').write_text(json.dumps(dict(files={str(H/'CIRCLE_SPEED.json'):sha(H/'CIRCLE_SPEED.json')}))+'\n')
 print('Private graph ready',len(graph),'recorded copies.')
if __name__=='__main__':main()
