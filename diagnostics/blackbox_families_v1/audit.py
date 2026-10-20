"""Read-only protected-source audit. Never descend into the excluded study."""
import hashlib,json,subprocess,sys,datetime
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/blackbox_families_v1'
MAC=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official'); POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(root,args):return subprocess.check_output(['git','-C',str(root),*args])
def snapshot():
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 for name,root in [('basin',ROOT),('macflow',MAC)]:
  args=['diff','--','.',' :(exclude)diagnostics/blackbox_families_v1'].copy() if False else ['diff','--','. ',]
  # The prohibited directory is excluded before git opens file contents.
  diffargs=['diff','--','.']+([':(exclude)diagnostics/blackbox_families_v1/**',':(exclude)diagnostics/decentralized_blackbox_planners_v1/**'] if name=='basin' else [])
  out[name]={'head':git(root,['rev-parse','HEAD']).decode().strip(),'status':git(root,['status','--porcelain']).decode(),'protected_diff_sha256':hashlib.sha256(git(root,diffargs)).hexdigest()}
 out['poc_py']={str(p):sha(p) for p in sorted((POC/'poc').rglob('*.py'))}
 sources=[]
 for folder in ['axis2_controller_v1','other_controller_classes_v1','blackbox_addendum_v1','ring_controller_critic_v1','axis4a_geometry_horizon_v1','controller_q_transfer_round3_v1','orthoflow3_tt_ff_matched_learnability_v1']:
  sources.extend((ROOT/'diagnostics'/folder).glob('*.py'))
 for folder in ['shared_rollout_db','ring_exchange','shared_control','new_benchmark_common']:
  sources.extend((ROOT/folder).rglob('*.py'))
 out['source_py']={str(p):sha(p) for p in sorted(set(sources))}
 return out
if __name__=='__main__':
 stage=sys.argv[1];p=HERE/f'INTEGRITY_{stage.upper()}.json';assert not p.exists();p.write_text(json.dumps(snapshot(),indent=2)+'\n');print(p)
