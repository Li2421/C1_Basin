"""Read-only evidence-tree and Git snapshots. Never opens the shared database."""
import pathlib,subprocess,hashlib,json,datetime,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
HISTORY=[ROOT/'diagnostics'/n for n in ['decentralized_pipeline_v1','teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c','teacher_student_macflow_v1d','teacher_student_macflow_v1f']]
SOURCES=[ROOT/'ring_exchange/expert.py',ROOT/'ring_exchange/environment.py',ROOT/'docs/controller_generalization_ring_summary_20261014.md',ROOT/'docs/prompt_round30g_student_line_mechanism_closeout_20261017.md',ROOT/'AGENTS.md']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snapshot(label):
 repos={}
 for name,p in [('Basin_C1',ROOT),('MACFlow_Official',pathlib.Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official'))]:
  def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(p),*a])
  diff=git('diff');repos[name]=dict(head=git('rev-parse','HEAD').decode().strip(),status=git('status','--porcelain').decode(),diff_sha256=hashlib.sha256(diff).hexdigest())
 files={};links={}
 for base in HISTORY:
  for p in sorted(base.rglob('*')):
   if p.is_symlink():links[str(p)]=str(p.readlink())
   if p.is_file():files[str(p)]=sha(p)
  print(label,base.name,'cumulative files',len(files),flush=True)
 for p in SOURCES:files[str(p)]=sha(p)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repos=repos,roots=[str(x) for x in HISTORY],files=files,symlinks=links)
 (HERE/f'INTEGRITY_{label}.json').write_text(json.dumps(r,indent=2));return r
if __name__=='__main__':
 label=sys.argv[1];r=snapshot(label)
 if label=='END':
  a=json.loads((HERE/'INTEGRITY_START.json').read_text());change={p:[a['files'].get(p),r['files'].get(p)] for p in a['files'].keys()|r['files'].keys() if a['files'].get(p)!=r['files'].get(p)};added={k:sorted(set(r['repos'][k]['status'].splitlines())-set(a['repos'][k]['status'].splitlines())) for k in a['repos']};removed={k:sorted(set(a['repos'][k]['status'].splitlines())-set(r['repos'][k]['status'].splitlines())) for k in a['repos']};stable=all(a['repos'][k]['head']==r['repos'][k]['head'] and a['repos'][k]['diff_sha256']==r['repos'][k]['diff_sha256'] for k in a['repos']);same_links=a['symlinks']==r['symlinks'];passed=not change and stable and same_links and not any(removed.values());c=dict(passed=passed,changed_files=change,symlinks_equal=same_links,git_diff_and_HEAD_equal=stable,git_status_equal=all(a['repos'][k]['status']==r['repos'][k]['status'] for k in a['repos']),concurrent_new_git_entries_not_written_by_this_round=added,removed_git_entries=removed,source_files=len(r['files']));(HERE/'INTEGRITY_COMPARISON.json').write_text(json.dumps(c,indent=2));print(c,flush=True)
