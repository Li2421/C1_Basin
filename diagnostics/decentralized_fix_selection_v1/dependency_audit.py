"""Supplemental coverage of dependency directories, with explicit timing evidence."""
from support import *
G=guard()
def main():
 tag=sys.argv[1];roots=[D/n for n in ['ring_exchange_stage1','controller_q_transfer_round3_v1','orthoflow3_tt_ff_matched_learnability_v1']];start=read(HERE/'INTEGRITY_BEGIN.json')['utc'];start_seconds=datetime.fromisoformat(start).timestamp();files={};ctime={}
 for root in roots:
  for p in sorted(root.rglob('*')):
   if p.is_file():files[str(p)]=sha(p);ctime[str(p)]=p.stat().st_ctime_ns
 ck=[]
 for r in read(HERE/'TEACHERS.json'):
  ck.append(dict(path=r['checkpoint'],expected=r['checkpoint_sha256'],actual=sha(r['checkpoint'])))
 for r in read(HERE/'STUDENTS.json').values():ck.append(dict(path=r['checkpoint'],expected=r['checkpoint_sha256'],actual=sha(r['checkpoint'])))
 p=read(HERE/'CONTROLLERS.json')
 for i in range(6):
  v=p['profiles'][f'G{i}']['payload'];ck.append(dict(path=v['flow_checkpoint_path'],expected=v['flow_checkpoint_sha256'],actual=sha(v['flow_checkpoint_path'])))
 later=[p for p,t in ctime.items() if t>int(start_seconds*1e9)]
 r=dict(utc=utc(),roots=[str(p) for p in roots],files=files,ctime_ns=ctime,main_begin_utc=start,files_with_ctime_after_main_begin=later,checkpoints=ck,frozen_checkpoint_hashes_match=all(c['actual']==c['expected'] for c in ck),timing_note='These dependency directories supplement the original full-tree experiment-result inventory. This inventory was read during the run, not at original BEGIN. Filesystem ctime predating BEGIN supplies additional no-write evidence; checkpoint SHA values were frozen in original manifests and checked on construction. This is not a retroactively timestamped snapshot.',guard=guard());write(HERE/f'DEPENDENCY_{tag}.json',r)
 if tag=='END':
  b=read(HERE/'DEPENDENCY_BASELINE.json');changed=[p for p,h in b['files'].items() if files.get(p)!=h];added=[p for p in files if p not in b['files']];write(HERE/'DEPENDENCY_COMPARISON.json',dict(passed=not changed and not added and r['frozen_checkpoint_hashes_match'] and not b['files_with_ctime_after_main_begin'],changed=changed,added=added,baseline_files=len(b['files']),baseline_all_ctime_before_main_begin=not b['files_with_ctime_after_main_begin'],frozen_checkpoint_hashes_match=r['frozen_checkpoint_hashes_match'],timing_note=r['timing_note']));assert not changed and not added and r['frozen_checkpoint_hashes_match']
 print(tag,len(files),'newer_ctime',len(later),'checkpoint_match',r['frozen_checkpoint_hashes_match'],flush=True)
if __name__=='__main__':main()
