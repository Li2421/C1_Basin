"""Build features before outcomes, and reject any premature locked-label access."""
from common import *
from features import aggregate,NAMES
from r5 import from_mechanism
from design_execution import ARMS


def features(phase,cohort):
    p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json')
    if cohort=='locked':assert (HERE/'FROZEN_PIPELINE.receipt.json').exists()
    names=p['controller_order'];rows=[];r5=[];index=[];gaps=[]
    for name in names:
        traces={arm:[None]*16 for arm in ('safety_baseline','F13','F10')}
        bad=[]
        for c in cases:
            if c['base_controller_name']!=name:continue
            path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            raw=read(path)
            assert not raw['terminal_labels_read'] and not raw['jax_enable_x64']
            assert not any(k in raw for k in ('success','timeout','collision','outcome'))
            assert sha(raw['trace_path'])==raw['trace_sha256']
            if not raw['complete']:bad.append(c['rollout_uid'])
            with np.load(raw['trace_path'],allow_pickle=False) as z:
                traces[c['arm']][c['state_index']]={k:z[k] for k in z.files}
            index.append(dict(controller=name,probe_uid=c['rollout_uid'],raw_path=str(path),raw_sha256=sha(path),
                              trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],complete=raw['complete']))
        actionfile=HERE/phase/'initial_actions'/f'{name}.npz'
        receipt=read(actionfile.with_suffix('.receipt.json'));assert sha(actionfile)==receipt['sha256']
        with np.load(actionfile) as z:random=z['actions']
        cfg=p['profiles'][name]['payload']
        manifest=read(cfg['adapter_manifest_path']) if 'adapter_manifest_path' in cfg else p['native_manifest']
        config=manifest['scenario_config']
        if bad:
            gaps.append(dict(controller=name,unknown_probes=bad,decision='Feature values unavailable; preserve probes, no rerun'))
            rows.append(np.full(len(NAMES),np.nan));r5.append(np.full(160,np.nan))
        else:
            rows.append(aggregate(traces,random,config['dt'],config['max_speed'],config['goal_tolerance']))
            r5.append(from_mechanism(traces))
    target=HERE/f'{cohort.upper()}_FEATURES.npz'
    assert not target.exists()
    np.savez_compressed(target,features=rows,r5=r5,names=names,types=[p['types'][n] for n in names],feature_names=NAMES)
    write(HERE/f'{cohort.upper()}_PROBE_TRAJECTORY_INDEX.json',index)
    write(HERE/f'{cohort.upper()}_FEATURE_GAPS.json',gaps)
    seal(f'{cohort.upper()}_FEATURES.receipt.json',[target,HERE/f'{cohort.upper()}_PROBE_TRAJECTORY_INDEX.json',
         HERE/f'{cohort.upper()}_FEATURE_GAPS.json',HERE/'features.py',HERE/'r5.py'],outcomes_read=False)
    print(cohort,'features sealed:',len(names),'controllers;',len(gaps),'unknown probe controllers')


def labels(eval_phase,tune_phase,cohort):
    if cohort=='locked':
        receipt=read(HERE/'LOCKED_CHOICES.receipt.json')
        for path,digest in receipt['files'].items():assert sha(path)==digest
    with np.load(HERE/f'{cohort.upper()}_FEATURES.npz',allow_pickle=False) as source:
        features={k:source[k] for k in source.files}
    names=list(features['names']);nc=len(names)
    evaluation=np.full((nc,32,8,2),-2,np.int8);tuning=np.full((nc,16,8),-2,np.int8)
    nominal=np.full((nc,32,2),-2,np.int8)
    events={event:np.zeros((nc,32,8,2),np.int8) for event in ('collision','timeout','unknown')}
    nominal_events={f'nominal_{event}':np.zeros((nc,32,2),np.int8) for event in ('collision','timeout','unknown')}
    evidence=[]
    with rollout_db.connect(True) as db:
        for phase,istuning in ((eval_phase,False),(tune_phase,True)):
            p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json')
            if set(p['controller_order'])!=set(names):raise RuntimeError('Feature/label controller cohort differs')
            for c in cases:
                row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',
                               (c['rollout_uid'],)).fetchone()
                if row is None:continue
                ci=names.index(c['base_controller_name']);si=c['state_index'];q=c['seed']
                value=-1 if row['numerical_failure'] else int(row['success'])
                evidence.append(dict(rollout_uid=c['rollout_uid'],phase=phase,controller=c['base_controller_name'],arm=c['arm']))
                if c['arm']=='nominal_only':
                    assert not istuning;nominal[ci,si,q]=value
                    for event in ('collision','timeout'):nominal_events[f'nominal_{event}'][ci,si,q]=row[event]
                    nominal_events['nominal_unknown'][ci,si,q]=row['numerical_failure']
                    continue
                ai=ARMS.index(c['arm'])
                if istuning:tuning[ci,si,ai]=value
                else:
                    evaluation[ci,si,ai,q]=value
                    for event in ('collision','timeout'):events[event][ci,si,ai,q]=row[event]
                    events['unknown'][ci,si,ai,q]=row['numerical_failure']
    missing=np.sum(evaluation==-2)+np.sum(tuning==-2)+np.sum(nominal==-2)
    if missing:
        write(HERE/f'{cohort.upper()}_LABEL_GAPS.json',dict(utc=utc(),missing=int(missing),
              reason='Incomplete evidence; not unknown or failure. No development bundle emitted.'))
        raise RuntimeError(f'{missing} labels absent')
    target=HERE/f'{cohort.upper()}_BUNDLE.npz'
    assert not target.exists()
    stateuids=[s['state_uid'] for s in read(HERE/'EVAL_STATES.json')]
    np.savez_compressed(target,**features,labels=evaluation,tuning=tuning,nominal=nominal,
                        state_uids=stateuids,arms=ARMS,**events,**nominal_events)
    write(HERE/f'{cohort.upper()}_LABEL_INDEX.json',evidence)
    seal(f'{cohort.upper()}_BUNDLE.receipt.json',[target,HERE/f'{cohort.upper()}_LABEL_INDEX.json'],
         cohort=cohort,unknown_preserved=True,missing_fabricated=False)
    print(cohort,'complete label bundle sealed')

if __name__=='__main__':
    if sys.argv[1]=='features':features(*sys.argv[2:])
    elif sys.argv[1]=='labels':labels(*sys.argv[2:])
