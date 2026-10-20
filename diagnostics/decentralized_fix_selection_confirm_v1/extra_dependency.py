"""Protect the legacy physical-feature normalizer stage before its first use."""
from core import *
from guard import guard
G=guard()
def main():
 tag=sys.argv[1];root=D/'state_nuisance_rootcause_v1/real_state_expansion';files={str(p):sha(p) for p in sorted(root.rglob('*')) if p.is_file()};write(HERE/f'NORMALIZER_STAGE_{tag}.json',dict(utc=utc(),root=str(root),files=files,scope='Additional legacy-critic dependency, snapshot before first physical23/normalizer use; primary simulations and training do not read it.',guard=guard()))
 if tag=='END':
  b=read(HERE/'NORMALIZER_STAGE_BEGIN.json');assert b['files']==files;write(HERE/'NORMALIZER_STAGE_COMPARISON.json',dict(passed=True,files=len(files),guard=guard()))
 print(tag,len(files),flush=True)
if __name__=='__main__':main()
