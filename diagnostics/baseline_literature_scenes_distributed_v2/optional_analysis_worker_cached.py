"""Posthoc dispatch for optional trajectories, with no rollout or journal write."""
from experiment import *
from atomic_write import write
from diagnosis_alias_cache import reuse
from registry import verify_receipt
import numpy as np

def main(stage):
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json');verify_receipt('DOWNSTREAM_READY.receipt.json');verify_receipt('DYNAMICS_DIAGNOSTICS.receipt.json')
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);groups=0
    for entry in read(HERE/stage/'GROUP_INDEX.json'):
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ['SLURM_JOB_ID'],pid=os.getpid()))
        assert sha(entry['file'])==entry['sha256']
        for row in read(entry['file']):
            dest=HERE/'DIAGNOSTICS'/f"{row['rollout_uid']}.json"
            if dest.exists():continue
            assert sha(row['path'])==row['sha256'];raw=read(row['path']);assert sha(raw['trace_path'])==raw['trace_sha256']
            if raw['case']['execution_engine']=='paper_dynamics':from dynamics_diagnostics import summary
            else:from failure_analysis import summary
            twin=reuse(raw)
            if twin is None:
                with np.load(raw['trace_path']) as z:result=summary(raw,z)
                reuse_info={}
            else:result=twin['result'];reuse_info=dict(alias_diagnosis_reuse=twin['provenance'])
            write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**reuse_info,**result))
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        if groups>=128:os.execv(sys.executable,[sys.executable,'-B',str(HERE/'optional_analysis_worker_cached.py'),stage])

if __name__=='__main__':main(sys.argv[1])
