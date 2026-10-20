from common import *
OLD=D/'axis4a_geometry_horizon_v1';R36=D/'axis4b_agent_count_v1'
def main():
 files=read(HERE/'ROUND35_FULL_INVENTORY.json')['files'];current={str(f) for f in OLD.rglob('*') if f.is_file()};errors=[]
 for f,v in files.items():
  if not Path(f).exists() or sha(f)!=v['sha256']:errors.append(dict(path=f,problem='missing orchanged'))
 for f in current-set(files):errors.append(dict(path=f,problem='newfile inreadonlyold35'))
 required=read(HERE/'HISTORY_READ_AUDIT.json')['round36_required_files']
 for f,h in required.items():
  if sha(f)!=h:errors.append(dict(path=f,problem='changedold36required'))
 for folder in [OLD,R36]:
  for f,h in read(folder/'NUMERICAL_COMPLETE.receipt.json')['files'].items():
   if sha(f)!=h:errors.append(dict(path=f,problem='oldfinalreceipthash'))
 for f,h in read(HERE/'N2_GATE_AUDIT.json')['historical_evidence'].items():
  if sha(f)!=h['sha256']:errors.append(dict(path=f,problem='N2oldsourcehash'))
 write(HERE/'OLD_SOURCE_IMMUTABILITY.json',dict(utc=utc(),passed=not errors,errors=errors,old35_full_files_verified=len(files),old36_required_files_verified=len(required),old35_36_final_receipts_verified=True,N2_historical_source_hashes_verified=True));assert not errors,errors[:10];print('Readonly source/assets immutable',len(files),'files',flush=True)
if __name__=='__main__':main()
