#!/usr/bin/env python3
from audit import *
def main():
 tag,newname=sys.argv[1:3];audit=json.load(open(HERE/f'validation_reuse_{tag}.json'));assert audit['reuse_allowed'];oldname=audit['batch'];src=HERE/'targeted_probe_rounds'/oldname;dst=HERE/'targeted_probe_rounds'/newname
 assert not dst.exists(),'Do not overwrite a validation ledger'
 for filename in ['manifest.csv','results.csv']:
  write(f'targeted_probe_rounds/{newname}/{filename}',read(src/filename))
 dump(f'targeted_probe_rounds/{newname}/frozen_parameters.json',json.load(open(src/'frozen_parameters.json')))
 gate=json.load(open(src/'gate.json'));gate.update(reused_from=oldname,new_exact_Q64=0,new_rollouts=0,current_fit_tag=tag,mathematical_identity_audit=f'validation_reuse_{tag}.json',fresh_at_original_parameter_freeze=True)
 dump(f'targeted_probe_rounds/{newname}/gate.json',gate);dump(f'targeted_probe_rounds/{newname}/reuse_audit.json',audit)
 dump(f'targeted_probe_rounds/{newname}/cost_estimate.json',dict(new_exact_q64=0,new_continuations=0,shards=0,estimated_wall_seconds=0,independent_validation_Q64_reused=audit['independent_Q64_reused'],reason='Exact unchanged mathematical instance and fitting-input identity; do not submit jobs for this alias'))
 print('Recorded reused validation alias:',newname,'new rollouts0')
if __name__=='__main__':main()
