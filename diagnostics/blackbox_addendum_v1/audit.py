from core import *
import subprocess
MAC=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
def git(p):
    def run(*args):return subprocess.check_output(['git','-C',str(p),*args])
    return dict(HEAD=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff')).hexdigest())
def snapshot():
    return dict(utc=utc(),Basin=git(ROOT),MACFlow=git(MAC),existing_sources={str(p):sha(p) for p in sorted(ROOT.rglob('*.py')) if not p.is_relative_to(HERE) and '.git' not in p.parts},PoC={str(p):sha(p) for folder in ['poc','source','validation_v2','validation_stage2'] for p in sorted((FIELD/folder).rglob('*.py'))},MAC_sources={str(p):sha(p) for p in sorted(MAC.rglob('*.py')) if '.git' not in p.parts})
def main(phase):
    x=snapshot();write(HERE/('INTEGRITY_'+phase+'.json'),x)
    if phase=='END':
        old=read(HERE/'INTEGRITY_BEGIN.json');delta={}
        for k in ['MACFlow','PoC','MAC_sources']:
            if old[k]!=x[k]:delta[k]=dict(before=old[k],after=x[k])
        changed={p:dict(before=h,after=x['existing_sources'].get(p)) for p,h in old['existing_sources'].items() if x['existing_sources'].get(p)!=h};new={p:h for p,h in x['existing_sources'].items() if p not in old['existing_sources']}
        for k in ['HEAD','diff_sha256']:
            if old['Basin'][k]!=x['Basin'][k]:delta['Basin_'+k]=dict(before=old['Basin'][k],after=x['Basin'][k])
        before=set(old['Basin']['status'].splitlines());after=set(x['Basin']['status'].splitlines());own=lambda s:'diagnostics/blackbox_addendum_v1/' in s
        write(HERE/'INTEGRITY_COMPARISON.json',dict(utc=utc(),existing_content_pass=not delta and not changed,changed_existing_sources=changed,other_changes=delta,external_added_sources=new,external_added_status=sorted(s for s in after-before if not own(s)),removed_status=sorted(before-after),own_added_status=sorted(s for s in after-before if own(s))))
if __name__=='__main__':main(sys.argv[1])
