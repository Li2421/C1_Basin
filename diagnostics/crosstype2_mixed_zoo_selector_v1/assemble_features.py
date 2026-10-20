from common import *
from features import aggregate,one_trace,NAMES,TIMES


def main():
    assert (HERE/'probe_COMPLETE.receipt.json').exists()
    packet=read(HERE/'PROBE_MANIFEST.json');p=read(HERE/'probe_protocol.json');names=packet['controllers']
    old=D/'crosstype1b_generator_critic_pipeline_v1'
    with np.load(old/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:previous={str(n):v for n,v in zip(z['names'],z['features'])}
    rows=[];gaps=[];index=[]
    for name in names:
        traces={a:[None]*16 for a in ('safety_baseline','F13','F10','RHR')};bad=set()
        for entry in packet['entries']:
            c=entry['case']
            if c['base_controller_name']!=name:continue
            raw=Path(entry['raw_path']);r=read(raw)
            if 'raw_sha256' in entry:assert sha(raw)==entry['raw_sha256']
            assert not r['terminal_labels_read'] and not any(k in r for k in ('success','outcome','termination'))
            assert sha(r['trace_path'])==r['trace_sha256']
            if not r['complete']:bad.add(c['arm'])
            with np.load(r['trace_path'],allow_pickle=False) as z:traces[c['arm']][c['state_index']]={k:z[k] for k in z.files}
            index.append(dict(controller=name,arm=c['arm'],state_index=c['state_index'],source=entry['source'],raw_path=str(raw),raw_sha256=sha(raw),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],complete=r['complete']))
        cfg=p['native_manifest']['scenario_config']
        if name in previous:base=previous[name].copy()
        elif bad&{'safety_baseline','F13','F10'}:base=np.full(24,np.nan)
        else:
            file=HERE/'probe/initial_actions'/f'{name}.npz';assert sha(file)==read(file.with_suffix('.receipt.json'))['sha256']
            with np.load(file,allow_pickle=False) as z:random=z['actions']
            base=aggregate(traces,random,cfg['dt'],cfg['max_speed'],cfg['goal_tolerance'])
        extra=np.full(10,np.nan)
        if 'RHR' not in bad:
            values=[one_trace(t,cfg['dt'],cfg['max_speed'],cfg['goal_tolerance']) for t in traces['RHR']]
            progress=np.mean([v['progress'] for v in values],0)
            extra[:6]=[*progress,float(np.mean([v['freeze'] for v in values])),float(np.mean([v['moving'] for v in values]))]
            if 'F13' not in bad:
                comparison=np.mean([one_trace(t,cfg['dt'],cfg['max_speed'],cfg['goal_tolerance'])['progress'] for t in traces['F13']],0)
                extra[6:]=progress-comparison
        vector=np.r_[base,extra];assert vector.shape==(34,);rows.append(vector)
        if bad:gaps.append(dict(controller=name,incomplete_probe_arms=sorted(bad),decision='Missing feature values remain NaN; no unknown probe rerun or partial averaging'))
    target=HERE/'CONTROLLER_FEATURES.npz'
    extra_names=[*(f'progress_RHR_{t}' for t in TIMES),'strict_freeze_RHR','moving_no_progress_RHR',*(f'RHR_minus_eta13_{t}' for t in TIMES)]
    np.savez_compressed(target,names=names,types=[p['types'][n] for n in names],features=rows,feature_names=[*NAMES,*extra_names])
    write(HERE/'PROBE_TRAJECTORY_INDEX.json',index);write(HERE/'FEATURE_GAPS.json',gaps)
    seal('CONTROLLER_FEATURES.receipt.json',[target,HERE/'PROBE_TRAJECTORY_INDEX.json',HERE/'FEATURE_GAPS.json',HERE/'features.py',HERE/'assemble_features.py'],outcome_labels_read=False,old24_dimensions_reused_exactly=True)
    print('34 mechanism features sealed for',len(names),'controllers;',len(gaps),'with explicitly missing probe features')


if __name__=='__main__':main()
