import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MAC=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args])
def snapshot():
 result={'utc':datetime.now(timezone.utc).isoformat()}
 for name,p in [('Basin',ROOT),('MACFlow',MAC)]:
  status=git(p,'status','--porcelain').decode()
  if name=='Basin':status='\n'.join(x for x in status.splitlines() if 'diagnostics/axis4b_agent_count_v1/' not in x)
  result[name]={'HEAD':git(p,'rev-parse','HEAD').decode().strip(),'status':status,'diff_sha256':hashlib.sha256(git(p,'diff')).hexdigest()}
 result['poc_py']={str(p):sha(p) for p in sorted((POC/'poc').rglob('*.py'))}
 result['existing_source_sha256']={str(p):sha(p) for p in sorted(ROOT.rglob('*.py')) if not p.is_relative_to(HERE) and '.git' not in p.parts}
 result['poc_importable_py']={str(p):sha(p) for folder in ['poc','source','validation_stage2','validation_v2'] for p in sorted((POC/folder).rglob('*.py'))}
 result['MACFlow_py']={str(p):sha(p) for p in sorted(MAC.rglob('*.py'))}
 result['tracked_diff_files']={str(ROOT/line):sha(ROOT/line) for line in git(ROOT,'diff','--name-only').decode().splitlines() if (ROOT/line).is_file()}
 return result
if __name__=='__main__':
 stage=sys.argv[1];r=snapshot();(HERE/f'INTEGRITY_{stage}.json').write_text(json.dumps(r,indent=2)+'\n')
 if stage=='END':
  a=json.loads((HERE/'INTEGRITY_BEGIN.json').read_text());changes=[k for k in a if k!='utc' and a[k]!=r[k]]
  (HERE/'INTEGRITY_COMPARISON.json').write_text(json.dumps({'passed':not changes,'changed_sections':changes},indent=2)+'\n')
 print(stage,'audit recorded')
