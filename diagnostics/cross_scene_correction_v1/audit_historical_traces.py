"""Hash all reused raw and saved traces before main-table consumption, with no simulations."""
from common import *


def main():
    output=HERE/'HISTORICAL_TRACE_AUDIT.json'
    if output.exists():
        assert read(output)['passed'];return
    rows=read(HERE/'HISTORY_CASES.json');raws={};traces={};nbytes=0
    for i,c in enumerate(rows):
        p=Path(c['reuse_source']);digest=sha(p);r=read(p)
        assert r['rollout_uid']==c['rollout_uid']
        if c.get('reuse_source_sha256'):assert digest==c['reuse_source_sha256']
        raws[str(p)]=digest;trace=Path(r['trace_path'])
        if str(trace)not in traces:
            traces[str(trace)]=sha(trace);nbytes+=trace.stat().st_size
        assert traces[str(trace)]==r['trace_sha256'],('HISTORICAL_TRACE_CHANGED',p,trace)
        if i%2000==0:print('Historical trace audit',i,'/',len(rows),flush=True)
    write(HERE/'REUSED_EVIDENCE_SHA256_START.json',dict(raw_files=raws,trace_files=traces))
    write(output,dict(passed=True,utc=utc(),requests=len(rows),raw_files=len(raws),trace_files=len(traces),trace_bytes=nbytes,new_rollouts=0))
    print('Historical trace audit passed',len(rows),flush=True)


if __name__=='__main__':main()
