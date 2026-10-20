import hashlib,json,pathlib,subprocess,sys,datetime
ROOT=pathlib.Path('/home/zhihan/research/Basin_C1'); OUT=ROOT/'diagnostics/teacher_student_macflow_v1f'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def snapshot():
 repos={}
 for name,p in [('basin',ROOT),('official',pathlib.Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official'))]:
  def git(*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(p),*args])
  diff=git('diff'); repos[name]={'head':git('rev-parse','HEAD').decode().strip(),'status':git('status','--porcelain').decode(),'diff_sha256':hashlib.sha256(diff).hexdigest()}
 paths=[pathlib.Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')]+[ROOT/'diagnostics'/('teacher_student_macflow_v1'+s) for s in ['', 'b','c','d']]
 files={str(p):sha(p) for base in paths for p in sorted(base.rglob('*')) if p.is_file()}
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repos':repos,'protected_files':files}
if __name__=='__main__':
 name=sys.argv[1]; data=snapshot(); (OUT/f'INTEGRITY_{name}.json').write_text(json.dumps(data,indent=2))
 print(name,len(data['protected_files']),data['repos'])
 if name=='END':
  start=json.loads((OUT/'INTEGRITY_START.json').read_text()); changes={p:[start['protected_files'].get(p),data['protected_files'].get(p)] for p in start['protected_files'].keys()|data['protected_files'].keys() if start['protected_files'].get(p)!=data['protected_files'].get(p)}
  new_status={k:sorted(set(data['repos'][k]['status'].splitlines())-set(start['repos'][k]['status'].splitlines())) for k in data['repos']}
  removed_status={k:sorted(set(start['repos'][k]['status'].splitlines())-set(data['repos'][k]['status'].splitlines())) for k in data['repos']}
  same_diff_head=all(start['repos'][k]['head']==data['repos'][k]['head'] and start['repos'][k]['diff_sha256']==data['repos'][k]['diff_sha256'] for k in data['repos'])
  result={'protected_file_changes':changes,'repos_equal':start['repos']==data['repos'],'protected_content_pass':not changes and same_diff_head and not any(removed_status.values()),'concurrent_new_git_entries_not_written_by_this_round':new_status,'removed_status_entries':removed_status};(OUT/'INTEGRITY_COMPARISON.json').write_text(json.dumps(result,indent=2)); print({k:v for k,v in result.items() if k!='protected_file_changes'})
