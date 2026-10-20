"""Pure training labels: no human eta or human schedule enters this bundle."""
from common import *
OLD=D/'crosstype1b_generator_critic_pipeline_v1'
def main(which):
    assert which in ('pilot','all');phases=['pilot'] if which=='pilot' else ['pilot','remaining'];base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];types=np.asarray([base['types'][n] for n in names]);mask=np.isin(types,['Flow','diffusion','ring_ippo']) if which=='pilot' else np.ones(len(names),bool)
    labels=np.full((62,48,26,1),-2,np.int8);uids=np.full(labels.shape,'',dtype='U96');events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')};trace_records={}
    with rollout_db.connect(True) as db:
        for phase in phases:
            assert (HERE/f'{phase}_COMPLETE.receipt.json').exists()
            for c in read(HERE/f'{phase}_CASES.json'):
                i=names.index(c['base_controller_name']);s=c['state_index'];a=c['training_slot'];row=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None
                assert labels[i,s,a,0]==-2;labels[i,s,a,0]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,0]=c['rollout_uid']
                for e in events:events[e][i,s,a,0]=row['numerical_failure'] if e=='unknown' else row[e]
                paths=[HERE/phase/'raw'/f"{c['rollout_uid']}.json",*[OLD/p/'raw'/f"{c['rollout_uid']}.json" for p in ('pilot','remaining')]]
                path=next((p for p in paths if p.exists()),None)
                if path is not None:
                    r=read(path);trace_records[c['rollout_uid']]=dict(record_path=str(path),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'])
    assert not np.any(labels[mask]==-2)
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices']
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:
        h=z['h'][:48];c=z['c'][:48];action=z['action'][:48][np.arange(48)[None,:,None],indices];proposals=z['proposals'][:48];state_uids=z['state_uids'][:48]
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:features=z['features']
    # The reference file is separate and is never accepted by train_critic.load_data.
    with np.load(OLD/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:
        assert np.array_equal(z['names'],names)
        for i in np.flatnonzero(mask):
            for j in range(5):assert np.array_equal(labels[i,:,j,0],z['labels'][i,:,j*2,0]) and np.array_equal(uids[i,:,j,0],z['uids'][i,:,j*2,0])
        refs={k:z[k][:,:,10:13,:] for k in ('labels','uids','collision','timeout','unknown')}
    target='PILOT_TRAINING_BUNDLE' if which=='pilot' else 'TRAINING_BUNDLE';assert not (HERE/f'{target}.npz').exists()
    np.savez_compressed(HERE/f'{target}.npz',names=np.asarray(names)[mask],types=types[mask],labels=labels[mask],uids=uids[mask],features=features[mask],h=h,c=c,action=action[mask],candidate_indices=indices[mask],proposals=proposals,state_uids=state_uids,**{k:v[mask] for k,v in events.items()})
    refname='PILOT_TRAINING_REFERENCES' if which=='pilot' else 'TRAINING_REFERENCES';np.savez_compressed(HERE/f'{refname}.npz',names=np.asarray(names)[mask],types=types[mask],**{k:v[mask] for k,v in refs.items()})
    write(HERE/f'{target}_TRACE_INDEX.json',trace_records)
    seal(f'{target}.receipt.json',[HERE/f'{target}.npz',HERE/f'{refname}.npz',HERE/f'{target}_TRACE_INDEX.json',HERE/'assemble.py',*[HERE/f'{p}_COMPLETE.receipt.json' for p in phases]],seed0_complete=True,critic_training_data_excludes_human_anchors=True,reference_file_not_training=True)
    print('Assembled',which,int(mask.sum()),'controllers; action shape',action[mask].shape,flush=True)
if __name__=='__main__':main(sys.argv[1])
