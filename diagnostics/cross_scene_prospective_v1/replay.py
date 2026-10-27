"""Required full historical episodes, compared byte for byte except wall clocks."""
from common import *
from execution import execute
import numpy as np

OLD=ROOT/'diagnostics/cross_scene_correction_v1'

def iter_array(path):
    decoder=json.JSONDecoder(); buf='';pos=0;start=False
    with Path(path).open() as f:
        eof=False
        while True:
            if pos>len(buf)//2:buf=buf[pos:];pos=0
            while pos<len(buf) and buf[pos].isspace():pos+=1
            if not start and pos<len(buf):
                assert buf[pos]=='[';pos+=1;start=True
            while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
            if pos<len(buf) and buf[pos]==']':return
            try:
                value,end=decoder.raw_decode(buf,pos)
            except json.JSONDecodeError:
                if eof:raise
                block=f.read(1024*1024);buf+=block;eof=not block;continue
            pos=end;yield value

def main():
    cases=[]
    for c in read_evidence(OLD/'pilot_CASES.json'):
        if c['state']['scene']=='xsc_double_n4' and c['controller_payload']['nominal']['type_code']=='prioritized_astar' and c['state']['index']==0:
            source=next(p for folder in ('pilot','pilot_tail','main')
                if (p:=OLD/folder/'raw'/f"{c['rollout_uid']}.json").exists())
            c=dict(c,reuse_source=str(source));cases.append(c)
    assert len(cases)==2
    needed={'cross_entropy','prioritized_astar'}
    evidence(OLD/'HISTORY_CASES.json')
    for c in iter_array(OLD/'HISTORY_CASES.json'):
        kind=c['controller_payload']['nominal']['type_code']
        if c['state']['scene']=='lit_crowded_random' and kind in needed and c['requested_action']==('eta13' if kind=='cross_entropy' else 'eta0'):
            cases.append(c);needed.remove(kind)
        if not needed:break
    assert len(cases)==4 and not needed
    source_hashes=read(HERE/'REUSED_EVIDENCE_SHA256_START.json')['files']
    for c in cases:
        p=Path(c['reuse_source']);raw=read(p)
        assert sha(raw['trace_path'])==raw['trace_sha256']
        source_hashes[str(p)]=sha(p);source_hashes[raw['trace_path']]=raw['trace_sha256']
    write(HERE/'REUSED_EVIDENCE_SHA256_START.json',dict(files=source_hashes))
    write(HERE/'replay_CASES.json',cases)
    plan=HERE/'replay_PLANNED_ROLLOUTS.json'
    write(plan,dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')}for c in cases]))
    pf=planner.preflight(plan);write(HERE/'replay_CACHE_PREFLIGHT.json',pf)
    assert pf['summary']['genuinely_missing']==0 and pf['summary']['ambiguous']==0,pf['summary']
    rows=[]
    for c in cases:
        dest=HERE/'replay'/f"{c['rollout_uid']}.json"
        if dest.exists():
            row=read(dest);assert row['passed'];rows.append(row);continue
        raw=read(c['reuse_source']);result,arrays=execute(c)
        with np.load(raw['trace_path'],allow_pickle=False) as stored:
            checks={}
            for name in stored.files:
                if 'seconds' in name:continue
                a=np.asarray(arrays[name]);b=stored[name]
                checks[name]=bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
        fields={k:canonical(v)==canonical(result[k]) for k,v in raw.items() if k in result and 'seconds' not in k and k!='operational_resume'}
        row=dict(scene=c['state']['scene'],rollout_uid=c['rollout_uid'],source=c['reuse_source'],
            trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],arrays=checks,fields=fields,
            passed=all(checks.values()) and all(fields.values()),elapsed_seconds=result['elapsed_seconds'])
        write(dest,row);rows.append(row)
        print('Replay',row['scene'],c['controller_payload']['nominal']['controller'],c['requested_action'],row['passed'],flush=True)
        if not row['passed']:
            write(HERE/'STOP.receipt.json',dict(reason='BITWISE_REPLAY_FAILED',case=row));raise AssertionError(row)
    write(HERE/'REPLAY_CHECK.json',dict(utc=utc(),passed=True,count=len(rows),cases=rows,
        excludes_only_wallclock_timing=True,bytewise_arrays=True,new_test_rollouts=0))

if __name__=='__main__':main()
