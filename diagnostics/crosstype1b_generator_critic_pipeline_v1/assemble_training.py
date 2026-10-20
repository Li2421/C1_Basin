"""Join reused anchor observations with completed generated-action evidence."""
from common import *

OLD=D/'crosstype1_learned_zoo_selector_v1'
ANCHORS={'safety_baseline':10,'S500':11,'Sprog':12}


def main(which):
    assert which in ('pilot','all')
    phases=['pilot'] if which=='pilot' else ['pilot','remaining']
    if (HERE/'training_seed1_COMPLETE.receipt.json').exists():phases.append('training_seed1')
    for phase in phases:
        for path,digest in read(HERE/f'{phase}_COMPLETE.receipt.json')['files'].items():assert sha(path)==digest
    p=read(HERE/'BASE_PROTOCOL.json');names=p['controller_order'];nc=len(names)
    labels=np.full((nc,48,13,2),-2,np.int8)
    events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')};uids=np.full(labels.shape,'',dtype='U96')
    sources=[]
    with rollout_db.connect(True) as db:
        for cohort in ('development','locked'):
            for split,offset in (('tuning',0),('evaluation',16)):
                cases=read(OLD/f'{cohort}_{split}_CASES.json')
                for c in cases:
                    if c['arm'] not in ANCHORS:continue
                    i=names.index(c['base_controller_name']);s=offset+c['state_index'];a=ANCHORS[c['arm']];q=c['seed']
                    row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    assert row is not None
                    labels[i,s,a,q]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,q]=c['rollout_uid']
                    for e in events:events[e][i,s,a,q]=row['numerical_failure'] if e=='unknown' else row[e]
                sources.extend([OLD/f'{cohort}_{split}_CASES.json',OLD/f'{cohort.upper()}_BUNDLE.receipt.json'])
        for phase in phases:
            for c in read(HERE/f'{phase}_CASES.json'):
                i=names.index(c['base_controller_name']);s=c['state_index'];a=c['candidate_index'];q=c['seed']
                assert labels[i,s,a,q]==-2
                row=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                assert row is not None
                labels[i,s,a,q]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,q]=c['rollout_uid']
                for e in events:events[e][i,s,a,q]=row['numerical_failure'] if e=='unknown' else row[e]
            sources.append(HERE/f'{phase}_COMPLETE.receipt.json')
    types=np.asarray([p['types'][n] for n in names]);mask=np.isin(types,('Flow','diffusion','ring_ippo')) if which=='pilot' else np.ones(nc,bool)
    assert not np.any(labels[mask,:,:,0]==-2),'Seed0 full menu must be complete before development or headroom'
    for cohort in ('DEVELOPMENT','LOCKED'):
        with np.load(OLD/f'{cohort}_BUNDLE.npz',allow_pickle=False) as z:
            for j,name in enumerate(z['names']):
                i=names.index(str(name))
                for old_a,new_a in ((0,10),(5,11),(6,12)):
                    assert np.array_equal(labels[i,:16,new_a,0],z['tuning'][j,:,old_a])
                    assert np.array_equal(labels[i,16:,new_a],z['labels'][j,:,old_a])
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'][:48];c=z['c'][:48];action=z['action'][:48];state_uids=z['state_uids'][:48]
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:
        assert list(z['names'])==names;features=z['features'];r5=z['r5']
    target=HERE/('PILOT_TRAINING_BUNDLE.npz' if which=='pilot' else 'TRAINING_BUNDLE.npz');assert not target.exists()
    np.savez_compressed(target,names=np.asarray(names)[mask],types=types[mask],labels=labels[mask],uids=uids[mask],
        features=features[mask],r5=r5[mask],h=h,c=c,action=action,state_uids=state_uids,**{e:v[mask] for e,v in events.items()})
    seal(target.with_suffix('.receipt.json').name,[target,HERE/'assemble_training.py',HERE/'CANDIDATES.receipt.json',*set(sources)],
        missing_seed_is_minus2=True,unknown_is_minus1=True,seed0_complete=True,
        full_menu_two_seed_controller_states=int(np.all(labels[mask]!=-2,axis=(2,3)).sum()))
    print('Assembled',which,int(mask.sum()),'controllers; complete two-seed controller-state rows:',int(np.all(labels[mask]!=-2,axis=(2,3)).sum()),flush=True)


if __name__=='__main__':main(sys.argv[1])
