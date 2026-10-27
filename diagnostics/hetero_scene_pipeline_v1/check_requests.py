"""Reconstruct historical literature identities byte-for-byte before registering proposals."""
from common import *
from requests_corrected import original_requests,from_original,request
from design import *

def main():
 rows=[];original=original_requests()
 for n in ['lit_circle_five','lit_swap8_free','lit_swap8_obstacles']:
  found=[c for c in original.values()if c['state']['scene']==n]
  for c in found[:4]:
   rebuilt=from_original(c,c['eta_a']);assert canonical(rebuilt)==canonical(c)
   rows.append(dict(scene=n,state_index=c['state']['index'],controller=c['controller_payload']['nominal']['controller'],rollout_uid=c['rollout_uid'],canonical_bytes_equal=True,eta=c['eta_a']))
 assert len(rows)==12
 write(HERE/'REQUEST_RECONSTRUCTION_CHECK.json',dict(passed=True,cases=len(rows),rows=rows,constructor_sha256=sha(HERE/'requests_corrected.py'),all_original_fields_preserved=True))
 print('Request reconstruction passed:',len(rows),flush=True)
if __name__=='__main__':main()
