"""Exactly two fast-controller historical diametric episodes; no search replay."""
from common import *
from execution import execute
from design import HET as OLD,CONTROL_IDS,CONTROL_TYPES
import numpy as np

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
    chosen=[];targets={'xsc_het_plus_w020':('sE05','centralized_sampling0'),'xsc_het_box_n10_r013_f25':('eta4','gradient_mpc0')}
    evidence(OLD/'main_CASES.json')
    for c in iter_array(OLD/'main_CASES.json'):
        name=c['state']['scene']
        if name not in targets:continue
        key,ctl=targets[name]
        if c['state']['index']!=0 or c['requested_action']!=key or c['controller_payload']['nominal']['controller']!=ctl:continue
        source=OLD/'main'/'raw'/f"{c['rollout_uid']}.json"
        if source.exists():chosen.append((c,source));targets.pop(name)
        if not targets:break
    assert len(chosen)==2, 'Required complete replay records unavailable'
    plan=HERE/'replay_PLANNED_ROLLOUTS.json';write(plan,dict(requests=[{k:c[k]for k in('state_uid','eta_uid','controller_uid','seed_key')}for c,p in chosen]));pf=planner.preflight(plan);write(HERE/'replay_CACHE_PREFLIGHT.json',pf)
    assert pf['summary']['genuinely_missing']==0 and pf['summary']['ambiguous']==0
    results=[]
    for c,source in chosen:
        raw=read_evidence(source);evidence(raw['trace_path']);assert sha(raw['trace_path'])==raw['trace_sha256'];result,arrays=execute(c)
        with np.load(raw['trace_path'],allow_pickle=False)as stored:
            checks={k:np.asarray(arrays[k]).dtype==stored[k].dtype and np.asarray(arrays[k]).shape==stored[k].shape and np.asarray(arrays[k]).tobytes()==stored[k].tobytes()for k in stored.files if'seconds'not in k}
        fields={k:canonical(v)==canonical(result[k])for k,v in raw.items()if k in result and'seconds'not in k and k!='operational_resume'}
        row=dict(scene=c['state']['scene'],controller=c['controller_payload']['nominal']['controller'],rollout_uid=c['rollout_uid'],source=str(source),trace_sha256=raw['trace_sha256'],arrays=checks,fields=fields,passed=all(checks.values())and all(fields.values()))
        write(HERE/'replay'/f"{c['rollout_uid']}.json",row);results.append(row)
        if not row['passed']:
            write(HERE/'STOP.receipt.json',dict(utc=utc(),reason='BITWISE_REPLAY_FAILED',case=row));raise RuntimeError('BITWISE_REPLAY_FAILED')
        print('Replay passed',row['scene'],row['controller'],flush=True)
    write(HERE/'REPLAY_CHECK.json',dict(utc=utc(),passed=True,count=2,cases=results,search_or_astar_replays=0,new_test_labels=0,only_wallclock_excluded=True))
if __name__=='__main__':main()
